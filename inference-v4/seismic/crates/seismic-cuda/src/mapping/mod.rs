//! The CUDA backend: target configuration, the mapping catalog, and the
//! pipeline `Backend` implementation.
//!
//! `Cuda` is the device-independent planner (profile, catalog, cost model)
//! used for selection; `CudaCompiler` binds one open device and implements
//! `Backend`, whose `assemble` compiles every encoded launch through that
//! device's driver context.

mod estimate;
pub use estimate::{EstimateError, EstimateModel, IDENTITY as COST_MODEL_IDENTITY};

use crate::catalog::CudaCatalog;
use crate::intrinsics::Dialect;
use seismic_compiler::pipeline::{AssemblyFailure, Backend, EncodedPlan};
use seismic_lang::sir::IntrinsicUse;
use seismic_realization::physical::SealedLaunch;
use seismic_realization::target::{CooperativeGrid, EffectiveTargetProfile, TargetLimits};

pub const TARGET: &str = "cuda";
pub const WARP: u32 = 32;

/// The device facts the CUDA target is configured from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
}

/// The CUDA backend planner: the effective target profile, the mapping
/// catalog, and the cost model. Device-bound assembly lives on
/// `CudaCompiler`.
pub struct Cuda {
    limits: Limits,
    cooperative: Option<CooperativeGrid>,
    target_profile: crate::target::TargetProfile,
    effective_profile: EffectiveTargetProfile,
    catalog: CudaCatalog,
}

/// The target configuration is invalid for this backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    /// The device facts do not support a CUDA target.
    Limits {
        max_threads_per_block: u32,
        max_grid_x: u32,
        warp_size: u32,
    },
    /// The cost configuration is invalid.
    Estimate(EstimateError),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Limits {
                max_threads_per_block,
                max_grid_x,
                warp_size,
            } => write!(
                f,
                "CUDA needs positive block, grid and warp capacities; the device offers \
                 {max_threads_per_block} threads per block, {max_grid_x} blocks, \
                 {warp_size}-lane warps"
            ),
            Self::Estimate(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl Cuda {
    /// Configure the backend from device facts without the cooperative
    /// facility (selection tests and synthetic targets).
    pub fn new(limits: Limits, estimate: EstimateModel) -> Result<Self, ConfigError> {
        Self::with_cooperative(limits, estimate, None)
    }

    /// Configure the backend from device facts. `cooperative` is the
    /// device's cooperative-grid facility (`None` when unsupported); a
    /// grid-cooperative proposal is then never made, never a fallback.
    pub fn with_cooperative(
        limits: Limits,
        estimate: EstimateModel,
        cooperative: Option<CooperativeGrid>,
    ) -> Result<Self, ConfigError> {
        if limits.max_threads_per_block == 0
            || limits.max_grid_x == 0
            || limits.warp_size == 0
            || i64::try_from(limits.max_scratch_bytes).is_err()
        {
            return Err(ConfigError::Limits {
                max_threads_per_block: limits.max_threads_per_block,
                max_grid_x: limits.max_grid_x,
                warp_size: limits.warp_size,
            });
        }
        estimate.validate().map_err(ConfigError::Estimate)?;
        let target_profile =
            crate::target::TargetProfile::synthetic_baseline(limits.clone(), cooperative.clone());
        let effective_profile = target_profile.effective_profile();
        let catalog = CudaCatalog::new(&limits, cooperative.clone(), &estimate);
        Ok(Self {
            limits,
            cooperative,
            target_profile,
            effective_profile,
            catalog,
        })
    }

    /// Configure the backend from one observed device.
    pub fn from_device(device: &crate::DeviceInfo) -> Result<Self, ConfigError> {
        Self::with_cooperative(
            Limits::from_device(device),
            EstimateModel::default(),
            device.cooperative_grid(),
        )
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    pub fn cooperative(&self) -> Option<&CooperativeGrid> {
        self.cooperative.as_ref()
    }

    pub fn target_profile(&self) -> &crate::target::TargetProfile {
        &self.target_profile
    }

    /// The fully populated effective target profile.
    pub fn effective_profile(&self) -> &EffectiveTargetProfile {
        &self.effective_profile
    }

    /// The effective hard limits.
    pub fn effective_limits(&self) -> &TargetLimits {
        &self.effective_profile.limits
    }

    pub fn capability_fingerprint(&self) -> String {
        self.effective_profile.capability_fingerprint.clone()
    }
}

/// The device-bound CUDA backend: compilation and native assembly through
/// one open device's driver context.
pub struct CudaCompiler<'a> {
    planner: Cuda,
    device: &'a crate::Device,
}

impl<'a> CudaCompiler<'a> {
    pub fn new(device: &'a crate::Device) -> Result<Self, ConfigError> {
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

impl Backend for CudaCompiler<'_> {
    type Dialect = Dialect;
    type Catalog = CudaCatalog;
    type EncodedLaunch = crate::encode::CudaLaunch;
    type NativeArtifact = crate::native::NativeArtifact;

    fn profile(&self) -> &EffectiveTargetProfile {
        self.planner.effective_profile()
    }

    fn catalog(&self) -> &CudaCatalog {
        &self.planner.catalog
    }

    /// Applicability of one exact typed capability use on this target.
    fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String> {
        self.planner.target_profile.supports_intrinsic(intrinsic)
    }

    /// Exhaustive mechanical encoding of one sealed launch. Total.
    fn encode(&self, launch: &SealedLaunch<Dialect>) -> crate::encode::CudaLaunch {
        crate::encode::encode(launch)
    }

    /// Compile every encoded launch through the device, reflect native
    /// facts against the selected contract, and seal one native tree
    /// mirroring the physical schedule exactly once.
    fn assemble(
        &self,
        plan: EncodedPlan<Dialect, crate::encode::CudaLaunch>,
    ) -> Result<crate::native::NativeArtifact, AssemblyFailure> {
        crate::native::assemble(self.device, plan)
    }
}
