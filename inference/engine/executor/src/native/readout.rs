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

impl ProjectionKernels {
    pub(crate) fn invocation_workspace_bytes(&self) -> u128 {
        match self {
            Self::RawPacked(kernel) => u128::from(kernel.invocation_workspace_bytes()),
            Self::PreparedPacked(kernel) => u128::from(kernel.invocation_workspace_bytes()),
            Self::Qualified {
                packed,
                progressive,
                ..
            } => {
                packed.invocation_workspace_bytes()
                    + u128::from(progressive.top.invocation_workspace_bytes())
                    + u128::from(progressive.refine.invocation_workspace_bytes())
                    + u128::from(progressive.exact.invocation_workspace_bytes())
                    + u128::from(progressive.planes.invocation_workspace_bytes())
            }
        }
    }
}
