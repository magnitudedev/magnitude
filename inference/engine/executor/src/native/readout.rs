use super::ProgressiveReadoutKernels;
use magnitude_kernels::{readout_head_rows, readout_project_rows};
use seismic::NativeKernel;

/// The chosen implementation of a scope-owned vocabulary projection.
#[derive(Clone, Debug)]
pub enum ProjectionKernels {
    RawPacked(NativeKernel<readout_head_rows::Entry>),
    PreparedPacked(NativeKernel<readout_project_rows::Entry>),
    /// Both recipes are retained from the same completed cost comparison.
    Qualified {
        packed: Box<ProjectionKernels>,
        progressive: ProgressiveReadoutKernels,
    },
}
