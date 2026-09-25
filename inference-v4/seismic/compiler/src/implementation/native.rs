//! Native compilation and reflection of a structurally closed kernel set.
//! Selection and prediction are deliberately outside this operation.

use super::*;

/// Forms and reconciles exactly one kernel selected by native
/// specialization. Infrastructure failures remain typed preparation errors;
/// callers decide candidate eligibility only after authoritative reflection.
pub(crate) fn realize_kernel<T, C>(
    compiler: &C,
    context: &C::Context,
    kernel: &seismic_ir::kernel::Kernel<T>,
    target: &DeviceDescription<T>,
) -> Result<
    (
        seismic_native_target::NativeKernel<T, C::Handle>,
        seismic_native_target::NativeArtifactMetrics,
    ),
    crate::errors::PreparationError,
>
where
    T: seismic_native_target::TargetFamily,
    C: seismic_native_target::NativeCompiler<T>,
{
    let formation = seismic_native_target::form_native_kernel(compiler, context, target, kernel)
        .and_then(|candidate| seismic_native_target::reconcile_native_kernel(compiler, candidate))
        .map_err(crate::errors::PreparationError::NativeCompilation)?;
    Ok(formation.into_parts())
}
