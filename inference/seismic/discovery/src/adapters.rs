//! The OS's display adapters on Windows (DXGI): the memory WDDM reports as
//! dedicated to each adapter (`DedicatedVideoMemory` and
//! `DedicatedSystemMemory`) and its per-process local video memory budget
//! (`IDXGIAdapter3::QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL)`),
//! by the adapter's LUID, which Vulkan reports as
//! `VkPhysicalDeviceIDProperties::deviceLUID` and CUDA as `cuDeviceGetLuid`.

use seismic_topology::AdapterReport;
use windows::core::Interface;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter1, IDXGIAdapter3, IDXGIFactory1, DXGI_ERROR_NOT_FOUND,
    DXGI_MEMORY_SEGMENT_GROUP_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO,
};

/// WDDM's local video memory budget and this process's current usage for one
/// adapter (`IDXGIAdapter3::QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LocalVideoMemory {
    pub(crate) budget: u64,
    pub(crate) current_usage: u64,
}

impl LocalVideoMemory {
    /// Local VRAM headroom remaining under WDDM's budget before demotion to
    /// system memory.
    pub(crate) fn headroom(self) -> u64 {
        self.budget.saturating_sub(self.current_usage)
    }
}

fn luid_bytes(luid: windows::Win32::Foundation::LUID) -> [u8; 8] {
    let mut bytes = [0u8; 8];
    bytes[..4].copy_from_slice(&luid.LowPart.to_le_bytes());
    bytes[4..].copy_from_slice(&luid.HighPart.to_le_bytes());
    bytes
}

fn query_adapter_local_memory(adapter: &IDXGIAdapter1) -> Option<LocalVideoMemory> {
    let adapter3: IDXGIAdapter3 = adapter.cast().ok()?;
    let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    unsafe { adapter3.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info) }.ok()?;
    Some(LocalVideoMemory {
        budget: info.Budget,
        current_usage: info.CurrentUsage,
    })
}

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
        let local_budget = query_adapter_local_memory(&adapter)
            .map(|info| info.budget)
            .filter(|budget| *budget > 0);
        // Both figures are memory the OS does not manage: video memory on
        // the adapter, and system memory the firmware reserved for it.
        adapters.push(AdapterReport {
            luid: luid_bytes(description.AdapterLuid),
            dedicated: description.DedicatedVideoMemory as u64
                + description.DedicatedSystemMemory as u64,
            local_budget,
        });
        index += 1;
    }
}

/// Query live WDDM local video memory budget and current usage for the DXGI
/// adapter matching `luid`.
pub(crate) fn local_video_memory(luid: [u8; 8]) -> Result<LocalVideoMemory, String> {
    let factory: IDXGIFactory1 =
        unsafe { CreateDXGIFactory1() }.map_err(|error| format!("CreateDXGIFactory1: {error}"))?;
    let mut index = 0;
    loop {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(adapter) => adapter,
            Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => {
                return Err("no DXGI adapter matched the device LUID".into());
            }
            Err(error) => return Err(format!("IDXGIFactory1::EnumAdapters1({index}): {error}")),
        };
        let description = unsafe { adapter.GetDesc1() }
            .map_err(|error| format!("IDXGIAdapter1::GetDesc1 of adapter {index}: {error}"))?;
        if luid_bytes(description.AdapterLuid) == luid {
            return query_adapter_local_memory(&adapter).ok_or_else(|| {
                "IDXGIAdapter3::QueryVideoMemoryInfo(DXGI_MEMORY_SEGMENT_GROUP_LOCAL) failed".into()
            });
        }
        index += 1;
    }
}
