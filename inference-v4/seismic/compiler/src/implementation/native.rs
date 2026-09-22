//! Native compilation and reflection of a structurally closed kernel set.
//! Selection and prediction are deliberately outside this operation.

use super::*;

pub(crate) fn realize<T, C>(
    compiler: &C,
    context: &C::Context,
    kernels: &KernelArena<T>,
    target: &DeviceDescription<T>,
) -> Result<
    (
        Vec<seismic_target::NativeKernel<T, C::Handle>>,
        seismic_target::NativeArtifactMetrics,
    ),
    crate::errors::PreparationError,
>
where
    T: seismic_target::TargetFamily,
    C: seismic_target::NativeCompiler<T>,
{
    let mut native_kernels = Vec::with_capacity(kernels.kernels().count());
    let mut metrics = seismic_target::NativeArtifactMetrics {
        compilation_ns: 0,
        code_bytes: 0,
        metadata_bytes: 0,
    };
    for (_, kernel) in kernels.kernels() {
        let formation = seismic_target::form_native_kernel(compiler, context, target, kernel)
            .and_then(|candidate| seismic_target::reconcile_native_kernel(compiler, candidate))
            .map_err(crate::errors::PreparationError::NativeCompilation)?;
        let (native, artifact) = formation.into_parts();
        metrics.compilation_ns = metrics
            .compilation_ns
            .checked_add(artifact.compilation_ns)
            .ok_or_else(|| {
                crate::errors::PreparationError::NativeCompilation(
                    seismic_target::NativeCompilationError::ToolchainResourceExhausted(
                        "aggregate native compilation duration exceeds u64 nanoseconds".into(),
                    ),
                )
            })?;
        metrics.code_bytes = metrics
            .code_bytes
            .checked_add(artifact.code_bytes)
            .ok_or_else(|| {
                crate::errors::PreparationError::NativeCompilation(
                    seismic_target::NativeCompilationError::ToolchainResourceExhausted(
                        "aggregate native code size exceeds u64 bytes".into(),
                    ),
                )
            })?;
        metrics.metadata_bytes = metrics
            .metadata_bytes
            .checked_add(artifact.metadata_bytes)
            .ok_or_else(|| {
                crate::errors::PreparationError::NativeCompilation(
                    seismic_target::NativeCompilationError::ToolchainResourceExhausted(
                        "aggregate native metadata size exceeds u64 bytes".into(),
                    ),
                )
            })?;
        native_kernels.push(native);
    }

    Ok((native_kernels, metrics))
}
