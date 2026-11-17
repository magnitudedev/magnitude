//! Minimal dynamically loaded NVML: one device's memory accounting
//! (`nvmlDeviceGetMemoryInfo_v2`), which separates the driver's reserve from
//! what processes use, and what it holds for one process
//! (`nvmlDeviceGetComputeRunningProcesses_v3` and
//! `nvmlDeviceGetGraphicsRunningProcesses_v3`, or their `_v2` where the
//! library predates them). Like the CUDA driver it needs no headers or libraries to
//! build, and is loaded and initialized once per process; a host without
//! NVML or without an entry point yields `None` for what it would read. The
//! device is found by its CUDA UUID, so CUDA and NVML enumeration order and
//! `CUDA_VISIBLE_DEVICES` do not matter.

use libloading::Library;
use seismic_native_target::process::{ProcessFootprint, ProcessId};
use std::ffi::{c_char, c_int, c_uint, c_void, CString};
use std::sync::OnceLock;

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

/// `nvmlProcessInfo_t`, as the `_v3` process queries fill it, and
/// `nvmlProcessInfo_v2_t`, its layout, as their `_v2` do.
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

/// The two lists of the processes on a device: those with a compute
/// context (CUDA) and those with a graphics one (Vulkan, OpenGL). NVML
/// added each version of the two together, so a library has both or
/// neither.
#[derive(Clone, Copy)]
struct ProcessQueries {
    compute: RunningProcesses,
    graphics: RunningProcesses,
}

/// NVML's entry points in an initialized session: loaded and initialized
/// once per process and kept for its life, as the driver is
/// ([`crate::driver::Driver::load`]).
struct Nvml {
    by_uuid: unsafe extern "C" fn(*const c_char, *mut DeviceHandle) -> Return,
    memory_info: unsafe extern "C" fn(DeviceHandle, *mut MemoryV2) -> Return,
    processes: Option<ProcessQueries>,
    _library: Library,
}

// NVML's function pointers and the loaded library are plain data; NVML is
// documented thread-safe.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

static NVML: OnceLock<Option<Nvml>> = OnceLock::new();

impl Nvml {
    /// The process's NVML session; none where the library, an entry point
    /// or its initialization is missing.
    fn get() -> Option<&'static Self> {
        // NVML symbols have the documented C ABI; the library is kept with
        // its function pointers for the life of the process.
        NVML.get_or_init(|| unsafe {
            let library = Library::new("libnvidia-ml.so.1").ok()?;
            let init = *library
                .get::<unsafe extern "C" fn() -> Return>(b"nvmlInit_v2\0")
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
            let query = |symbol: &[u8]| library.get::<RunningProcesses>(symbol).ok().map(|symbol| *symbol);
            let processes = [
                (
                    &b"nvmlDeviceGetComputeRunningProcesses_v3\0"[..],
                    &b"nvmlDeviceGetGraphicsRunningProcesses_v3\0"[..],
                ),
                (
                    &b"nvmlDeviceGetComputeRunningProcesses_v2\0"[..],
                    &b"nvmlDeviceGetGraphicsRunningProcesses_v2\0"[..],
                ),
            ]
            .into_iter()
            .find_map(|(compute, graphics)| {
                Some(ProcessQueries {
                    compute: query(compute)?,
                    graphics: query(graphics)?,
                })
            });
            (init() == 0).then_some(Self {
                by_uuid,
                memory_info,
                processes,
                _library: library,
            })
        })
        .as_ref()
    }
}

/// One device of the process's NVML session.
pub(crate) struct NvmlDevice {
    nvml: &'static Nvml,
    handle: DeviceHandle,
}

impl NvmlDevice {
    /// The NVML device whose UUID is `uuid` (`cuDeviceGetUuid_v2`), as a
    /// whole GPU or a MIG partition.
    pub fn open(uuid: [u8; 16]) -> Option<Self> {
        let nvml = Nvml::get()?;
        let mut handle = std::ptr::null_mut();
        let found = ["GPU", "MIG"].into_iter().any(|prefix| {
            let text = CString::new(uuid_text(prefix, uuid)).expect("UUID text has no NUL");
            // SAFETY: a NUL-terminated UUID and a handle to write.
            unsafe { (nvml.by_uuid)(text.as_ptr(), &mut handle) == 0 }
        });
        found.then_some(Self { nvml, handle })
    }

    /// What the device's driver holds for process `pid` (its context and
    /// allocations), as NVML counts it.
    pub fn process_bytes(&self, pid: u32) -> Result<u64, ProcessMemoryError> {
        let queries = self.nvml.processes.ok_or(ProcessMemoryError::Unsupported)?;
        own_bytes([&self.listed(queries.compute)?, &self.listed(queries.graphics)?], pid)
    }

    /// The processes `query` lists on the device.
    fn listed(&self, query: RunningProcesses) -> Result<Vec<ProcessInfo>, ProcessMemoryError> {
        // A count of zero asks how many processes there are; a process may
        // start between two queries, so the room asked for is the last count.
        let mut processes = Vec::new();
        loop {
            let mut count = c_uint::try_from(processes.len()).expect("a process count NVML gave");
            match unsafe { query(self.handle, &mut count, processes.as_mut_ptr()) } {
                0 => {
                    processes.truncate(count as usize);
                    return Ok(processes);
                }
                INSUFFICIENT_SIZE => processes.resize(count as usize, ProcessInfo::default()),
                code => return Err(ProcessMemoryError::Query(code)),
            }
        }
    }

    pub fn sample(&self) -> Option<MemorySample> {
        let mut memory = MemoryV2 {
            version: MEMORY_V2_VERSION,
            ..MemoryV2::default()
        };
        if unsafe { (self.nvml.memory_info)(self.handle, &mut memory) } != 0 {
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
    /// NVML lacks the compute and graphics process queries, `_v3` and `_v2`.
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
            Self::Unsupported => write!(f, "NVML lacks the compute and graphics process queries (_v3 and _v2)"),
            Self::Query(code) => write!(f, "NVML's process query failed ({code})"),
            Self::Unlisted { pid } => write!(
                f,
                "NVML lists no process {pid} on a device it holds a context on: its PID namespace is not NVML's"
            ),
            Self::Unreported { pid } => write!(f, "NVML does not report process {pid}'s memory"),
        }
    }
}

/// What NVML's compute and graphics `lists` say process `pid` holds.
///
/// Within a list, the sum of its entries: a GPU's handle lists a process
/// once per MIG instance it holds a context in. Across the lists, the most
/// of the two: NVML documents `usedGpuMemory` only as the process's used
/// memory on the device, not which of its contexts it counts, and
/// `nvidia-smi` shows a process with both kinds of context (`C+G`) as one
/// entry with one figure, so a process in both lists may be counted whole
/// in each; adding them could count it twice, and the larger one counts
/// all of it whichever way NVML attributes it.
fn own_bytes(lists: [&[ProcessInfo]; 2], pid: u32) -> Result<u64, ProcessMemoryError> {
    lists
        .into_iter()
        .map(|processes| {
            processes
                .iter()
                .filter(|process| process.pid == pid)
                .try_fold(None, |sum: Option<u64>, process| match process.used_gpu_memory {
                    NOT_AVAILABLE => Err(ProcessMemoryError::Unreported { pid }),
                    bytes => Ok(Some(sum.unwrap_or(0) + bytes)),
                })
        })
        .try_fold(None, |most: Option<u64>, listed| Ok(most.max(listed?)))?
        .ok_or(ProcessMemoryError::Unlisted { pid })
}

/// What `process` holds on the device `uuid` (its context and allocations),
/// as NVML counts it; unobserved where NVML is absent, has no such device,
/// fails, or reports no figure for the process. NVML lists processes by
/// their PIDs in the host's PID namespace, so a process in a container's
/// namespace is not found by the PID it knows itself by, and reads
/// unobserved there.
pub(crate) fn process_footprint(process: ProcessId, uuid: [u8; 16]) -> ProcessFootprint {
    NvmlDevice::open(uuid)
        .ok_or(ProcessMemoryError::Unavailable)
        .and_then(|device| device.process_bytes(process.get()))
        .map_or(ProcessFootprint::Unobserved, ProcessFootprint::Observed)
}

/// What this process's context holds on the device `uuid`, as NVML counts
/// the process there: read once, while the process holds the context and no
/// allocation on the device (`seismic_native_target::process::DriverMemory`).
pub(crate) fn context_bytes(uuid: [u8; 16]) -> Result<u64, ProcessMemoryError> {
    NvmlDevice::open(uuid)
        .ok_or(ProcessMemoryError::Unavailable)?
        .process_bytes(std::process::id())
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
        let compute = |processes: &[ProcessInfo], pid| own_bytes([processes, &[]], pid);
        let listed = [process(7, 100), process(9, NOT_AVAILABLE), process(11, 5)];
        assert_eq!(compute(&listed, 11), Ok(5));
        assert_eq!(compute(&listed, 9), Err(ProcessMemoryError::Unreported { pid: 9 }));
        assert_eq!(compute(&listed, 8), Err(ProcessMemoryError::Unlisted { pid: 8 }));
        assert_eq!(compute(&[], 8), Err(ProcessMemoryError::Unlisted { pid: 8 }));
        // A process listed once per MIG instance holds the sum; one
        // instance without a figure leaves the whole unreported.
        let instances = [process(7, 100), process(11, 5), process(7, 20)];
        assert_eq!(compute(&instances, 7), Ok(120));
        let partial = [process(7, 100), process(7, NOT_AVAILABLE)];
        assert_eq!(compute(&partial, 7), Err(ProcessMemoryError::Unreported { pid: 7 }));
        // A graphics process alone (Vulkan) reads from the graphics list.
        assert_eq!(own_bytes([&[], &[process(7, 64)]], 7), Ok(64));
        assert_eq!(own_bytes([&[process(11, 5)], &[process(7, 64)]], 11), Ok(5));
        // A process in both lists is counted once: the larger list's.
        assert_eq!(own_bytes([&[process(7, 100)], &[process(7, 100)]], 7), Ok(100));
        assert_eq!(own_bytes([&instances, &[process(7, 64)]], 7), Ok(120));
        assert_eq!(own_bytes([&[process(7, 10)], &[process(7, 64)]], 7), Ok(64));
        // Either list without a figure for it leaves it unreported.
        assert_eq!(
            own_bytes([&[process(7, 100)], &[process(7, NOT_AVAILABLE)]], 7),
            Err(ProcessMemoryError::Unreported { pid: 7 })
        );
        assert_eq!(own_bytes([&[], &[]], 7), Err(ProcessMemoryError::Unlisted { pid: 7 }));
    }

    /// A device NVML cannot read for the process, whether the library is
    /// absent or it has no device of the UUID, reads unobserved: no figure is
    /// guessed.
    #[test]
    fn a_device_nvml_cannot_read_is_unobserved() {
        let uuid = *b"no-such-device!!";
        assert!(NvmlDevice::open(uuid).is_none());
        assert_eq!(process_footprint(ProcessId::current(), uuid), ProcessFootprint::Unobserved);
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
