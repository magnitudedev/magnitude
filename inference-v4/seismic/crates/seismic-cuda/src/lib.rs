//! Direct PTX emission and CUDA driver execution. No nvcc or NVRTC runtime dependency.
mod driver;
pub mod ptx;
pub mod execution;
pub mod model;
mod runtime;
pub use runtime::{Buffer, Device, DeviceInfo, Kernel, NativeArtifact, NativeImage, NativeResources, Sequence};
