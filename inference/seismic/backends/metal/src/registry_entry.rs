//! What the device's IORegistry entry states that `MTLDevice` does not: its
//! GPU cores (`gpu-core-count`), which bound the threadgroups a resident
//! launch holds at once (`seismic_lang::registry::BackendName::resident_launches`).

use std::ffi::{c_char, c_void};

type CFTypeRef = *const c_void;
type IoObject = u32;

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IORegistryEntryIDMatching(entry_id: u64) -> *mut c_void;
    fn IOServiceGetMatchingService(main_port: u32, matching: *mut c_void) -> IoObject;
    fn IORegistryEntryCreateCFProperty(entry: IoObject, key: CFTypeRef, allocator: CFTypeRef, options: u32) -> CFTypeRef;
    fn IOObjectRelease(object: IoObject) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(allocator: CFTypeRef, text: *const c_char, encoding: u32) -> CFTypeRef;
    fn CFGetTypeID(value: CFTypeRef) -> usize;
    fn CFNumberGetTypeID() -> usize;
    fn CFNumberGetValue(number: CFTypeRef, kind: isize, value: *mut c_void) -> u8;
    fn CFRelease(value: CFTypeRef);
}

/// `kIOMainPortDefault`.
const MAIN_PORT: u32 = 0;
/// `kCFStringEncodingUTF8`.
const UTF8: u32 = 0x0800_0100;
/// `kCFNumberSInt64Type`.
const SINT64: isize = 4;

/// The GPU cores resident launches are bound by where the device's
/// registry entry states none ([`gpu_cores`]): one. The count is used only
/// to size a resident launch's grid (`RESIDENT`: half of the threadgroups
/// the cores hold at once), whose threadgroups wait for one another at grid
/// barriers. Counting more cores than run at once dispatches threadgroups
/// that are not co-resident, so a barrier waits on groups that cannot run
/// until its bounded wait gives up and fails the submission; counting fewer
/// only runs fewer groups, each covering more of the work. Every GPU has a
/// core, so one is the one count that is never more than the GPU's.
pub(crate) const UNSTATED_GPU_CORES: u64 = 1;

/// The GPU cores of the device whose registry entry is `registry_id`
/// (`MTLDevice.registryID`): its AGX accelerator's `gpu-core-count`; where
/// no entry states one (an Intel or AMD GPU, which is no AGX accelerator,
/// or a macOS VM's paravirtual GPU), [`UNSTATED_GPU_CORES`].
pub(crate) fn gpu_cores(registry_id: u64) -> u64 {
    stated_gpu_cores(registry_id).unwrap_or(UNSTATED_GPU_CORES)
}

/// The positive `gpu-core-count` the registry entry `registry_id` states.
fn stated_gpu_cores(registry_id: u64) -> Option<u64> {
    // SAFETY: each IOKit and CoreFoundation call gets the arguments its
    // documentation requires; the matching dictionary is consumed by
    // `IOServiceGetMatchingService`, and every object created here is
    // released here.
    unsafe {
        let matching = IORegistryEntryIDMatching(registry_id);
        if matching.is_null() {
            return None;
        }
        let service = IOServiceGetMatchingService(MAIN_PORT, matching);
        if service == 0 {
            return None;
        }
        let key = CFStringCreateWithCString(std::ptr::null(), c"gpu-core-count".as_ptr(), UTF8);
        let value = IORegistryEntryCreateCFProperty(service, key, std::ptr::null(), 0);
        CFRelease(key);
        IOObjectRelease(service);
        if value.is_null() {
            return None;
        }
        let mut cores = 0i64;
        let read = CFGetTypeID(value) == CFNumberGetTypeID()
            && CFNumberGetValue(value, SINT64, (&raw mut cores).cast()) != 0;
        CFRelease(value);
        (read && cores > 0).then_some(cores as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device no registry entry states cores of (here, an id no entry
    /// has) bounds its resident launches by the documented count, one.
    #[test]
    fn a_device_without_a_stated_core_count_is_bound_to_one_core() {
        assert_eq!(gpu_cores(u64::MAX), UNSTATED_GPU_CORES);
        assert_eq!(UNSTATED_GPU_CORES, 1);
    }
}
