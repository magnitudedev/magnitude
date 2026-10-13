//! Minimal dynamically loaded NVML: one device's memory accounting
//! (`nvmlDeviceGetMemoryInfo_v2`), which separates the driver's reserve from
//! what processes use. Like the CUDA driver it needs no headers or libraries
//! to build; a host without NVML or without the v2 entry point yields `None`.
//! The device is found by its CUDA UUID, so CUDA and NVML enumeration order
//! and `CUDA_VISIBLE_DEVICES` do not matter.

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

/// An initialized NVML session bound to one device; shut down on drop.
pub(crate) struct NvmlDevice {
    shutdown: unsafe extern "C" fn() -> Return,
    memory_info: unsafe extern "C" fn(DeviceHandle, *mut MemoryV2) -> Return,
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
                handle,
                _library: library,
            })
        }
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
