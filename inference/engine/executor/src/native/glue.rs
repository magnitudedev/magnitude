use magnitude_kernels::{
    conditioning_overlay, copy_rows, feature_copy, sample_rows, selection_notice, shape_rows,
};
use seismic::NativeKernel;

/// Immutable state, shaping, sampling, and conditioning handles.
#[derive(Debug)]
pub struct GlueKernels {
    pub(super) feature_copy: Option<NativeKernel<feature_copy::Entry>>,
    pub(super) shape_rows: Option<NativeKernel<shape_rows::Entry>>,
    pub(super) sample_rows: Option<NativeKernel<sample_rows::Entry>>,
    /// The copy of a read's selected rows for the host, of a load whose
    /// host fills a table for such reads (`ProgramPlan::notices_selection`).
    pub(super) selection_notice: Option<NativeKernel<selection_notice::Entry>>,
    pub(super) conditioning_overlay: Option<NativeKernel<conditioning_overlay::Entry>>,
    pub(super) copy_rows_f32: Option<NativeKernel<copy_rows::Entry>>,
    pub(super) copy_rows_f16: Option<NativeKernel<copy_rows::Entry>>,
    pub(super) copy_rows_bf16: Option<NativeKernel<copy_rows::Entry>>,
    pub(super) copy_rows_u32: Option<NativeKernel<copy_rows::Entry>>,
    pub(super) copy_rows_i32: Option<NativeKernel<copy_rows::Entry>>,
}
