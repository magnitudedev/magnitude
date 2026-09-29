//! The Metal device every backend test runs on.
use crate::{DeviceHandle, MetalDevice};
use std::sync::Once;

static VALIDATION: Once = Once::new();

/// Opens the system default device with the Metal API validation layer
/// enabled. Metal reads `MTL_DEBUG_LAYER` when the process makes its first
/// Metal call, so the variable is set once before any test touches Metal;
/// an encoding the API forbids then aborts the test process.
pub(crate) fn metal_device() -> MetalDevice {
    VALIDATION.call_once(|| std::env::set_var("MTL_DEBUG_LAYER", "1"));
    MetalDevice::open(DeviceHandle::system_default().unwrap()).unwrap()
}
