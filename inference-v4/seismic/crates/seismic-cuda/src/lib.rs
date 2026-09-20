//! CUDA backend: physical strategy construction over the one common family
//! builder, direct PTX emission, and CUDA driver execution. No nvcc or
//! NVRTC runtime dependency; the driver library is loaded dynamically, so
//! the crate builds on hosts without CUDA.
mod driver;
mod emitter;
pub mod mapping;
pub mod native;
pub mod physical;
mod runtime;
pub mod target;
pub use mapping::CudaCompiler;
pub use native::{
    ControlSource, CudaLaunch, CudaParam, Emitted, EncodedLaunch, StorageMirror,
    MAX_KERNEL_PARAMETER_BYTES,
};
pub use physical::{
    capability_fingerprint, cuda_target_profile, elaborate, AtomicMode, CudaDialect,
    CudaLayoutTemplate, CudaOp, CudaResolvedLayout, CudaScalarDest, CudaSsa, CudaStorageRef,
    MathMode, StrategyFlags,
};
pub use runtime::{
    Buffer, Device, DeviceInfo, NativeFinalizationError, NativeImage, NativeResources,
    PhysicalKernel, PhysicalSequence,
};
pub use target::{
    ComputeCapability, DriverApiVersion, FactSource, PtxTarget, PtxVersion, TargetError,
    TargetObservation, TargetProfile, TargetRequirement, TargetTier,
};
