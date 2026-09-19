//! Direct PTX emission and CUDA driver execution. No nvcc or NVRTC runtime dependency.
mod driver;
pub mod ptx;
pub mod execution;
pub mod model;
pub mod tuning;
mod runtime;
pub use runtime::{Buffer, Device, DeviceInfo, Kernel, NativeImage, NativeResources, Sequence};
