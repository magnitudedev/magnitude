//! CUDA backend: the mapping-rule catalog, the typed intrinsic catalog,
//! the exhaustive mechanical PTX encoder, the native assembler, and the
//! executor over prepared invocations. No nvcc or NVRTC runtime
//! dependency; the driver library is loaded dynamically, so the crate
//! builds on hosts without CUDA.
//!
//! There is no structural walker: universal semantic lowering is
//! core-owned, and this crate never imports the superseded realization
//! monolith or its formation internals.
mod catalog;
mod driver;
mod encode;
mod intrinsics;
pub mod mapping;
pub mod native;
pub mod runtime;
pub mod target;

pub use catalog::CudaCatalog;
pub use encode::{CudaLaunch, CudaParam, MAX_KERNEL_PARAMETER_BYTES};
pub use intrinsics::{CudaIntrinsic, CudaLayoutTemplate, CudaResolvedLayout, Dialect};
pub use mapping::{ConfigError, Cuda, CudaCompiler, EstimateModel, Limits, TARGET, WARP};
pub use native::NativeArtifact;
pub use runtime::{
    AssemblyError, Buffer, Device, DeviceInfo, Executor, NativeResources, OpenError, Outcome,
    Prepared,
};
