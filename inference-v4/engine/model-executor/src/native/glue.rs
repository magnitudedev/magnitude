use magnitude_model_kernels::{conditioning_overlay, copy_rows, sample_rows, shape_rows};
use seismic::NativeKernel;

/// Immutable state, shaping, sampling, and conditioning handles.
#[derive(Debug)]
pub struct GlueKernels {
    pub(super) shape_rows: Option<NativeKernel<shape_rows::Entry>>,
    pub(super) sample_rows: Option<NativeKernel<sample_rows::Entry>>,
    pub(super) conditioning_overlay: Option<NativeKernel<conditioning_overlay::Entry>>,
    pub(super) copy_rows_f32: Option<NativeKernel<copy_rows::Entry>>,
    pub(super) copy_rows_f16: Option<NativeKernel<copy_rows::Entry>>,
    pub(super) copy_rows_bf16: Option<NativeKernel<copy_rows::Entry>>,
    pub(super) copy_rows_u32: Option<NativeKernel<copy_rows::Entry>>,
}
