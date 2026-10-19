//! The OS's display adapters on Windows (DXGI): the memory WDDM reports as
//! dedicated to each adapter (`DedicatedVideoMemory` and
//! `DedicatedSystemMemory`), by the adapter's LUID, which Vulkan reports as
//! `VkPhysicalDeviceIDProperties::deviceLUID` and CUDA as `cuDeviceGetLuid`.

use crate::discovery::AdapterReport;
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1, DXGI_ERROR_NOT_FOUND};

/// Every adapter DXGI enumerates.
pub(crate) fn adapters() -> Result<Vec<AdapterReport>, String> {
    let factory: IDXGIFactory1 =
        unsafe { CreateDXGIFactory1() }.map_err(|error| format!("CreateDXGIFactory1: {error}"))?;
    let mut adapters = Vec::new();
    let mut index = 0;
    loop {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(adapter) => adapter,
            Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => return Ok(adapters),
            Err(error) => return Err(format!("IDXGIFactory1::EnumAdapters1({index}): {error}")),
        };
        let description = unsafe { adapter.GetDesc1() }
            .map_err(|error| format!("IDXGIAdapter1::GetDesc1 of adapter {index}: {error}"))?;
        let luid = description.AdapterLuid;
        let mut bytes = [0u8; 8];
        bytes[..4].copy_from_slice(&luid.LowPart.to_le_bytes());
        bytes[4..].copy_from_slice(&luid.HighPart.to_le_bytes());
        // Both figures are memory the OS does not manage: video memory on
        // the adapter, and system memory the firmware reserved for it.
        adapters.push(AdapterReport {
            luid: bytes,
            dedicated: description.DedicatedVideoMemory as u64
                + description.DedicatedSystemMemory as u64,
        });
        index += 1;
    }
}
