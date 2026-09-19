//! CUDA backend: the `mapping` side of joint selection, direct PTX emission and CUDA
//! driver execution. No nvcc or NVRTC runtime dependency; the driver library is loaded
//! dynamically, so the crate builds on hosts without CUDA.
mod driver;
pub mod execution;
pub mod mapping;
pub mod ptx;
mod runtime;
pub use runtime::{Buffer, Device, DeviceInfo, Kernel, NativeImage, NativeResources, Sequence};
