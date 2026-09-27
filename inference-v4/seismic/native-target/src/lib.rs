//! Immutable target descriptions and native artifact formation contracts.
//!
//! This crate deliberately has no compiler, estimator, solver, executor, or
//! concrete-backend dependency.  A target family describes types.  A
//! [`DeviceDescription`] contains immutable facts.  A [`NativeCompiler`]
//! consumes closed IR using an explicit live context and produces a candidate
//! which can only become a [`NativeKernel`] after reflection is reconciled.

mod description;
mod error;
mod native;

pub use description::{
    CompatibilityIdentity, DeviceDescription, DeviceDescriptionIdentity, DeviceDescriptionParts,
    NativeNumericalModeIdentity, NumericalEnvironmentIdentity, TargetFamily,
};
pub use error::{NativeCompilationError, TargetDescriptionError, ToolchainUnavailable};
pub use native::{
    form_native_kernel, reconcile_native_kernel, ClusterPortability, NativeArtifactMetrics,
    NativeClusterDomain, NativeCompiler, NativeFormation, NativeKernel, NativeKernelCandidate,
    NativeKernelDescription, NativeKernelIdentity, NativeKernelReflection, NativeLaunchDomain,
    NativeResourceUsage, NativeResources,
};
