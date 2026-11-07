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

/// The GPU cores of the device whose registry entry is `registry_id`
/// (`MTLDevice.registryID`); `None` when the entry states none.
pub(crate) fn gpu_cores(registry_id: u64) -> Option<u64> {
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
