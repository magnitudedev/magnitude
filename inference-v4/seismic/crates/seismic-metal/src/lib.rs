//! Metal backend: the sealed `MetalIntrinsic` dialect, the declarative
//! mapping catalog (streaming, blocked, subgroup, matrix — nothing else),
//! the exhaustive mechanical MSL encoder, the native assembler, and the
//! macOS executor over prepared invocations.
//!
//! The backend owns exactly: the effective target profile and its filled
//! `TargetLimits` (no cooperative grid on Metal), the intrinsic catalog
//! total over effective signatures, the optional mapping rules expressed
//! declaratively over occurrence facts, the probe-calibrated cost model,
//! the total MSL emission over `KernelOp<MetalIntrinsic>`, native
//! assembly with reflection folded into direct handles, and runtime
//! execution over `PreparedInvocation` with only `ExecutionFailure`
//! outcomes. Boundaries name caller storage; storage, geometry, and
//! guards come only from the sealed launch; there is no fallback, retry,
//! or alternate path.

pub mod catalog;
pub mod encode;
pub mod estimate;
pub mod intrinsics;
#[cfg(target_os = "macos")]
pub mod native;
pub mod target;
#[cfg(target_os = "macos")]
pub mod runtime;
