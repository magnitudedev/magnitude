//! Single-owner validation of direct native launches across Metal, CUDA, and
//! Vulkan.
//!
//! Every backend that issues direct launches constructs a
//! [`DirectLaunchContract`] from its device and pipeline/function limits and
//! validates each [`DirectLaunchRequest`] through [`DirectLaunchContract::validate`]
//! before encoding or submitting work. Indirect dispatches carry
//! [`LaunchGrid::Indirect`] so host-side grid dimensions can never be checked
//! for emptiness or device bounds when the grid lives in a GPU buffer.

use crate::errors::ExecutionError;

/// How a direct launch's 3D grid dimensions are supplied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchGrid {
    /// Host-supplied 3D workgroup / block counts.
    Direct([u64; 3]),
    /// GPU-supplied indirect dispatch buffer; grid counts are not known on the
    /// host and are neither tested for zero nor checked against host grid
    /// bounds.
    Indirect,
}

/// Hardware and ABI limits for validating a direct native launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectLaunchContract {
    /// Maximum buffer argument-table slots (e.g. Metal's 31-slot table).
    /// `None` for backends without a fixed buffer slot table (CUDA, Vulkan).
    pub max_buffer_slots: Option<usize>,
    /// Maximum grid dimensions (workgroups / blocks) per axis for direct
    /// dispatches.
    pub max_grid: [u64; 3],
    /// Maximum workgroup / block dimensions per axis.
    /// `None` when workgroup size is fixed in the compiled pipeline (Vulkan).
    pub max_group_size: Option<[u64; 3]>,
    /// Maximum total threads per workgroup / block (typically reflected from
    /// the compiled pipeline/function).
    /// `None` when workgroup size is fixed in the compiled pipeline (Vulkan).
    pub max_group_threads: Option<u64>,
    /// Maximum total shared / threadgroup memory bytes (static + dynamic).
    /// `None` when shared memory is fixed in the compiled pipeline (Vulkan).
    pub max_shared_bytes: Option<u64>,
    /// Maximum kernel parameter / argument block bytes (Metal `setBytes`,
    /// CUDA kernel parameter space, Vulkan argument chunk).
    pub max_param_bytes: Option<u64>,
}

/// Per-launch geometry and resource requirements validated against a
/// [`DirectLaunchContract`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectLaunchRequest {
    /// Number of buffer argument-table slots consumed by this launch.
    pub buffer_slots: usize,
    /// Direct 3D grid counts, or [`LaunchGrid::Indirect`] for GPU-driven
    /// indirect dispatch.
    pub grid: LaunchGrid,
    /// Workgroup / block dimensions per axis, or `None` when fixed in the
    /// compiled pipeline (Vulkan).
    pub group: Option<[u64; 3]>,
    /// Static shared / threadgroup memory bytes declared by the
    /// pipeline/function.
    pub static_shared_bytes: u64,
    /// Dynamic shared / threadgroup memory bytes requested by this launch
    /// (already rounded to any backend alignment requirement).
    pub dynamic_shared_bytes: u64,
    /// Kernel parameter / argument block bytes passed to the launch.
    pub param_bytes: u64,
}

/// Outcome of validating a [`DirectLaunchRequest`] against a
/// [`DirectLaunchContract`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidatedLaunch {
    /// The launch does no device work (a direct grid axis or a workgroup axis
    /// is zero) and should be skipped without encoding or submitting.
    Empty,
    /// The launch is non-empty and satisfies every limit of the contract.
    Active {
        /// Validated 3D `u32` grid for [`LaunchGrid::Direct`], or `None` for
        /// [`LaunchGrid::Indirect`].
        grid: Option<[u32; 3]>,
        /// Validated 3D `u32` workgroup / block size when supplied on the
        /// host, or `None` when fixed in the compiled pipeline.
        group: Option<[u32; 3]>,
    },
}

impl DirectLaunchContract {
    /// Validate `request` against all configured limits of this contract.
    pub fn validate(&self, request: &DirectLaunchRequest) -> Result<ValidatedLaunch, ExecutionError> {
        let empty_grid = matches!(request.grid, LaunchGrid::Direct(grid) if grid.contains(&0));
        let empty_group = request.group.is_some_and(|group| group.contains(&0));
        if empty_grid || empty_group {
            return Ok(ValidatedLaunch::Empty);
        }

        if let Some(limit) = self.max_buffer_slots {
            if request.buffer_slots > limit {
                return Err(ExecutionError::SubmissionFailed(format!(
                    "native launch uses {} buffer slots, but the backend allows {limit}",
                    request.buffer_slots
                )));
            }
        }

        if let Some(limit) = self.max_param_bytes {
            if request.param_bytes > limit {
                return Err(ExecutionError::SubmissionFailed(format!(
                    "native launch passes {} parameter bytes, but the backend allows {limit}",
                    request.param_bytes
                )));
            }
        }

        let grid = match request.grid {
            LaunchGrid::Indirect => None,
            LaunchGrid::Direct(axes) => {
                let mut validated = [0u32; 3];
                for axis in 0..3 {
                    if axes[axis] > self.max_grid[axis] {
                        return Err(ExecutionError::SubmissionFailed(format!(
                            "native launch requests {} groups on axis {axis}, but the device allows {}",
                            axes[axis], self.max_grid[axis]
                        )));
                    }
                    validated[axis] = u32::try_from(axes[axis]).map_err(|_| {
                        ExecutionError::SubmissionFailed(format!(
                            "native launch grid axis {axis} ({}) exceeds 32-bit driver ABI",
                            axes[axis]
                        ))
                    })?;
                }
                Some(validated)
            }
        };

        let group = match request.group {
            None => None,
            Some(axes) => {
                if let Some(max_size) = self.max_group_size {
                    for axis in 0..3 {
                        if axes[axis] > max_size[axis] {
                            return Err(ExecutionError::SubmissionFailed(format!(
                                "native launch requests {} threads on workgroup axis {axis}, but the device allows {}",
                                axes[axis], max_size[axis]
                            )));
                        }
                    }
                }
                let mut validated = [0u32; 3];
                for axis in 0..3 {
                    validated[axis] = u32::try_from(axes[axis]).map_err(|_| {
                        ExecutionError::SubmissionFailed(format!(
                            "native launch workgroup axis {axis} ({}) exceeds 32-bit driver ABI",
                            axes[axis]
                        ))
                    })?;
                }
                let threads = axes
                    .iter()
                    .try_fold(1u64, |product, value| product.checked_mul(*value))
                    .ok_or_else(|| {
                        ExecutionError::SubmissionFailed(
                            "native workgroup thread count overflowed".into(),
                        )
                    })?;
                if let Some(max_threads) = self.max_group_threads {
                    if threads > max_threads {
                        return Err(ExecutionError::SubmissionFailed(format!(
                            "native launch requests {threads} threads per workgroup, but the pipeline allows {max_threads}"
                        )));
                    }
                }
                Some(validated)
            }
        };

        if let Some(max_shared) = self.max_shared_bytes {
            let total_shared = request
                .static_shared_bytes
                .checked_add(request.dynamic_shared_bytes)
                .ok_or_else(|| {
                    ExecutionError::SubmissionFailed("native shared memory size overflowed".into())
                })?;
            if total_shared > max_shared {
                return Err(ExecutionError::SubmissionFailed(format!(
                    "native launch requests {total_shared} shared memory bytes ({} static + {} dynamic), but the device allows {max_shared}",
                    request.static_shared_bytes, request.dynamic_shared_bytes
                )));
            }
        }

        Ok(ValidatedLaunch::Active { grid, group })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_contract() -> DirectLaunchContract {
        DirectLaunchContract {
            max_buffer_slots: Some(31),
            max_grid: [65_535, 65_535, 65_535],
            max_group_size: Some([1024, 1024, 64]),
            max_group_threads: Some(1024),
            max_shared_bytes: Some(32_768),
            max_param_bytes: Some(4096),
        }
    }

    fn sample_request() -> DirectLaunchRequest {
        DirectLaunchRequest {
            buffer_slots: 10,
            grid: LaunchGrid::Direct([4, 2, 1]),
            group: Some([128, 2, 1]),
            static_shared_bytes: 1024,
            dynamic_shared_bytes: 2048,
            param_bytes: 64,
        }
    }

    #[test]
    fn accepts_valid_direct_launch_at_exact_limits() {
        let contract = sample_contract();
        let request = DirectLaunchRequest {
            buffer_slots: 31,
            grid: LaunchGrid::Direct([65_535, 65_535, 65_535]),
            group: Some([16, 1, 64]),
            static_shared_bytes: 16_384,
            dynamic_shared_bytes: 16_384,
            param_bytes: 4096,
        };
        assert_eq!(
            contract.validate(&request).unwrap(),
            ValidatedLaunch::Active {
                grid: Some([65_535, 65_535, 65_535]),
                group: Some([16, 1, 64]),
            }
        );
    }

    #[test]
    fn skips_empty_direct_grid_or_workgroup() {
        let contract = sample_contract();
        for grid in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
            let mut request = sample_request();
            request.grid = LaunchGrid::Direct(grid);
            assert_eq!(contract.validate(&request).unwrap(), ValidatedLaunch::Empty);
        }
        for group in [[0, 1, 1], [1, 0, 1], [1, 1, 0]] {
            let mut request = sample_request();
            request.group = Some(group);
            assert_eq!(contract.validate(&request).unwrap(), ValidatedLaunch::Empty);
        }
        // Indirect launch with a zero workgroup axis is still empty.
        let mut request = sample_request();
        request.grid = LaunchGrid::Indirect;
        request.group = Some([0, 1, 1]);
        assert_eq!(contract.validate(&request).unwrap(), ValidatedLaunch::Empty);
    }

    #[test]
    fn allows_valid_indirect_launches_with_or_without_host_group() {
        // Metal-style indirect launch (host workgroup size, GPU grid).
        let metal = sample_contract();
        let mut request = sample_request();
        request.grid = LaunchGrid::Indirect;
        assert_eq!(
            metal.validate(&request).unwrap(),
            ValidatedLaunch::Active {
                grid: None,
                group: Some([128, 2, 1]),
            }
        );

        // Vulkan-style indirect launch (pipeline workgroup size, GPU grid).
        let vulkan = DirectLaunchContract {
            max_buffer_slots: None,
            max_grid: [65_535, 65_535, 65_535],
            max_group_size: None,
            max_group_threads: None,
            max_shared_bytes: None,
            max_param_bytes: Some(256 << 10),
        };
        let vk_request = DirectLaunchRequest {
            buffer_slots: 0,
            grid: LaunchGrid::Indirect,
            group: None,
            static_shared_bytes: 0,
            dynamic_shared_bytes: 0,
            param_bytes: 128,
        };
        assert_eq!(
            vulkan.validate(&vk_request).unwrap(),
            ValidatedLaunch::Active {
                grid: None,
                group: None,
            }
        );
    }

    #[test]
    fn rejects_over_limit_buffer_slots() {
        let contract = sample_contract();
        let mut request = sample_request();
        request.buffer_slots = 32;
        let error = contract.validate(&request).unwrap_err();
        assert!(
            matches!(error, ExecutionError::SubmissionFailed(ref msg) if msg.contains("buffer slots")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn rejects_over_limit_per_axis_grid() {
        let contract = sample_contract();
        for axis in 0..3 {
            let mut request = sample_request();
            let mut grid = [1, 1, 1];
            grid[axis] = 65_536;
            request.grid = LaunchGrid::Direct(grid);
            let error = contract.validate(&request).unwrap_err();
            assert!(
                matches!(error, ExecutionError::SubmissionFailed(ref msg) if msg.contains("groups on axis")),
                "axis {axis}: unexpected error {error:?}"
            );
        }
    }

    #[test]
    fn rejects_over_limit_per_axis_workgroup_and_total_threads() {
        let contract = sample_contract();
        // Per-axis Z limit (64) exceeded even though total threads (65) <= 1024.
        let mut request = sample_request();
        request.group = Some([1, 1, 65]);
        let error = contract.validate(&request).unwrap_err();
        assert!(
            matches!(error, ExecutionError::SubmissionFailed(ref msg) if msg.contains("workgroup axis 2")),
            "unexpected error: {error:?}"
        );

        // Total threads (32 * 64 = 2048) exceeds max_group_threads (1024)
        // even though each axis is within per-axis limits.
        let mut request = sample_request();
        request.group = Some([32, 64, 1]);
        let error = contract.validate(&request).unwrap_err();
        assert!(
            matches!(error, ExecutionError::SubmissionFailed(ref msg) if msg.contains("threads per workgroup")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn rejects_over_limit_and_overflowing_shared_memory() {
        let contract = sample_contract();
        let mut request = sample_request();
        request.static_shared_bytes = 20_000;
        request.dynamic_shared_bytes = 12_769;
        let error = contract.validate(&request).unwrap_err();
        assert!(
            matches!(error, ExecutionError::SubmissionFailed(ref msg) if msg.contains("shared memory")),
            "unexpected error: {error:?}"
        );

        let mut overflow = sample_request();
        overflow.static_shared_bytes = u64::MAX;
        overflow.dynamic_shared_bytes = 1;
        let error = contract.validate(&overflow).unwrap_err();
        assert!(
            matches!(error, ExecutionError::SubmissionFailed(ref msg) if msg.contains("overflowed")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn rejects_over_limit_param_bytes() {
        let contract = sample_contract();
        let mut request = sample_request();
        request.param_bytes = 4097;
        let error = contract.validate(&request).unwrap_err();
        assert!(
            matches!(error, ExecutionError::SubmissionFailed(ref msg) if msg.contains("parameter bytes")),
            "unexpected error: {error:?}"
        );
    }
}
