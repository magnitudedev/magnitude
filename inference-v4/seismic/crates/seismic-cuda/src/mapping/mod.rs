//! CUDA target configuration for constructive physical compilation.

mod estimate;
pub use estimate::{EstimateModel, Totals, IDENTITY};

use seismic_lang::sir::IntrinsicUse;

pub const TARGET: &str = "cuda";
pub const WARP: u32 = 32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_threads_per_block: u32,
    pub max_grid_x: u32,
    pub warp_size: u32,
    pub max_scratch_bytes: u64,
}

impl Limits {
    pub fn gb10() -> Self {
        Self {
            max_threads_per_block: 1024,
            max_grid_x: i32::MAX as u32,
            warp_size: WARP,
            max_scratch_bytes: 130_663_231_488 / 4,
        }
    }

    pub fn from_device(device: &crate::DeviceInfo) -> Self {
        Self {
            max_threads_per_block: device.max_threads_per_block,
            max_grid_x: device.max_grid_x,
            warp_size: device.warp_size,
            max_scratch_bytes: device.global_memory_bytes / 4,
        }
    }

    pub(crate) fn target(&self) -> crate::target::TargetLimits {
        crate::target::TargetLimits {
            max_threads_per_block: self.max_threads_per_block,
            max_grid_x: self.max_grid_x,
            warp_size: self.warp_size,
            max_scratch_bytes: self.max_scratch_bytes,
        }
    }
}

pub struct Cuda {
    pub(crate) limits: Limits,
    pub(crate) target_profile: crate::target::TargetProfile,
    pub(crate) physical_target_profile:
        seismic_realization::executable::ExecutableTargetProfile<crate::physical::CudaCapability>,
}

pub struct CudaCompiler<'a> {
    planner: Cuda,
    device: &'a crate::Device,
}

impl<'a> CudaCompiler<'a> {
    pub fn new(device: &'a crate::Device) -> Result<Self, String> {
        Ok(Self {
            planner: Cuda::from_device(&device.info)?,
            device,
        })
    }

    pub fn planner(&self) -> &Cuda {
        &self.planner
    }

    pub fn device(&self) -> &'a crate::Device {
        self.device
    }
}

impl Cuda {
    pub fn new(limits: Limits, estimate: EstimateModel) -> Result<Self, String> {
        if limits.max_threads_per_block == 0
            || limits.max_grid_x == 0
            || limits.warp_size == 0
            || i64::try_from(limits.max_scratch_bytes).is_err()
        {
            return Err(format!(
                "CUDA needs positive block, grid and warp capacities; the device offers {} threads per block, {} blocks, {}-lane warps",
                limits.max_threads_per_block, limits.max_grid_x, limits.warp_size
            ));
        }
        estimate.validate()?;
        let target_profile = crate::target::TargetProfile::synthetic_baseline(limits.target());
        let physical_target_profile = crate::physical::target_profile(&limits, &target_profile);
        Ok(Self {
            limits,
            target_profile,
            physical_target_profile,
        })
    }

    pub fn from_device(device: &crate::DeviceInfo) -> Result<Self, String> {
        let mut backend = Self::new(Limits::from_device(device), EstimateModel::default())?;
        backend.target_profile = crate::target::TargetProfile::from_observation(
            crate::target::TargetObservation::driver(
                device.compute_capability,
                device.driver_version,
            )
            .map_err(|error| error.to_string())?,
            backend.limits.target(),
        )
        .map_err(|error| error.to_string())?;
        backend.physical_target_profile =
            crate::physical::target_profile(&backend.limits, &backend.target_profile);
        Ok(backend)
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    pub fn target_profile(&self) -> &crate::target::TargetProfile {
        &self.target_profile
    }
}

impl seismic_compiler::pipeline::Backend for Cuda {
    type Dialect = crate::physical::CudaDialect;
    type EncodedLaunch = crate::native::CudaLaunch;
    type NativeArtifact = crate::native::Emitted;

    fn target(&self) -> &'static str {
        TARGET
    }
    fn capability_fingerprint(&self) -> String {
        self.physical_target_profile.capability_fingerprint.clone()
    }
    fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String> {
        self.target_profile.supports_intrinsic(intrinsic)
    }
    fn target_profile(
        &self,
    ) -> &seismic_realization::executable::ExecutableTargetProfile<crate::physical::CudaCapability>
    {
        &self.physical_target_profile
    }
    fn elaborate(
        &self,
        logical: &seismic_lang::logical::LogicalProgram,
    ) -> Result<seismic_realization::executable::PlanFamily<Self::Dialect>, String> {
        crate::physical::elaborate(logical, &self.limits)
    }
    fn encode_launch(
        &self,
        launch: &seismic_realization::executable::ResolvedLaunch<Self::Dialect>,
    ) -> Result<Self::EncodedLaunch, String> {
        crate::native::encode_launch(launch, &self.target_profile)
    }
    fn assemble(
        &self,
        encoded: seismic_compiler::pipeline::EncodedPlan<Self::Dialect, Self::EncodedLaunch>,
    ) -> Result<Self::NativeArtifact, String> {
        crate::native::assemble(encoded)
    }
}

impl seismic_compiler::pipeline::Backend for CudaCompiler<'_> {
    type Dialect = crate::physical::CudaDialect;
    type EncodedLaunch = crate::native::CudaLaunch;
    type NativeArtifact = crate::PhysicalSequence;

    fn target(&self) -> &'static str {
        TARGET
    }
    fn capability_fingerprint(&self) -> String {
        self.planner
            .physical_target_profile
            .capability_fingerprint
            .clone()
    }
    fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String> {
        self.planner.target_profile.supports_intrinsic(intrinsic)
    }
    fn target_profile(
        &self,
    ) -> &seismic_realization::executable::ExecutableTargetProfile<crate::physical::CudaCapability>
    {
        &self.planner.physical_target_profile
    }
    fn elaborate(
        &self,
        logical: &seismic_lang::logical::LogicalProgram,
    ) -> Result<seismic_realization::executable::PlanFamily<Self::Dialect>, String> {
        crate::physical::elaborate(logical, &self.planner.limits)
    }
    fn encode_launch(
        &self,
        launch: &seismic_realization::executable::ResolvedLaunch<Self::Dialect>,
    ) -> Result<Self::EncodedLaunch, String> {
        crate::native::encode_launch(launch, &self.planner.target_profile)
    }
    fn assemble(
        &self,
        encoded: seismic_compiler::pipeline::EncodedPlan<Self::Dialect, Self::EncodedLaunch>,
    ) -> Result<Self::NativeArtifact, String> {
        self.device
            .compile_emitted(crate::native::assemble(encoded)?)
            .map_err(|error| error.to_string())
    }
}
