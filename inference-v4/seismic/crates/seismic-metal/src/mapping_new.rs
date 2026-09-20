//! Metal target configuration and the pipeline `Backend` implementation.

#[path = "mapping/estimate.rs"]
pub mod estimate;

pub use estimate::{EstimateModel, Group, Totals, IDENTITY};

use crate::physical::{self, MetalDialect, StrategyLimits};
use seismic_lang::{logical::LogicalProgram, sir::IntrinsicUse};
use seismic_realization::executable::{
    EffectiveTargetProfile, PlanFamily, ResolvedLaunch, TargetLimits,
};

pub const TARGET: &str = "metal";
pub const MAX_GROUPS: u64 = 65_535;
pub const SUBGROUP: u32 = 32;
pub const CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES: u64 = 128 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_bytes: u64,
    pub max_private_bytes: u64,
}

impl Limits {
    #[cfg(target_os = "macos")]
    pub fn from_device(device: &crate::runtime::DeviceInfo) -> Self {
        Self {
            max_threads_per_threadgroup: device.max_threads_per_threadgroup,
            max_threadgroup_bytes: device.max_threadgroup_bytes,
            max_private_bytes: device.profile.private_storage_budget_bytes.value,
        }
    }

    pub fn synthetic() -> Self {
        Self {
            max_threads_per_threadgroup: 1024,
            max_threadgroup_bytes: 32 * 1024,
            max_private_bytes: CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES,
        }
    }
}

/// The Metal backend: the effective target profile (exact signatures and
/// limits).
pub struct Metal {
    target_profile: crate::target::TargetProfile,
    executable_target_profile: EffectiveTargetProfile,
}

impl Metal {
    pub fn new(limits: Limits) -> Result<Self, String> {
        if limits.max_threads_per_threadgroup < u64::from(SUBGROUP)
            || i64::try_from(limits.max_threads_per_threadgroup).is_err()
            || i64::try_from(limits.max_threadgroup_bytes).is_err()
        {
            return Err(format!(
                "Metal needs at least {SUBGROUP} threads per threadgroup; the target offers {}",
                limits.max_threads_per_threadgroup
            ));
        }
        let target_profile = crate::target::TargetProfile::synthetic(
            limits.max_threads_per_threadgroup,
            limits.max_threadgroup_bytes,
            u64::MAX,
            limits.max_private_bytes,
        );
        Ok(Self::from_profile(limits, target_profile))
    }

    fn from_profile(limits: Limits, target_profile: crate::target::TargetProfile) -> Self {
        let executable_target_profile = EffectiveTargetProfile {
            backend: TARGET.into(),
            capability_fingerprint: target_profile.fingerprint().into(),
            toolchain_fingerprint: format!(
                "{};{}",
                crate::target::BACKEND_IMPLEMENTATION_REVISION,
                crate::target::COMPILER_PROBE_REVISION
            ),
            effective_signatures: target_profile.effective_signatures(),
            limits: TargetLimits {
                max_participants: limits.max_threads_per_threadgroup.min(i64::MAX as u64) as i64,
                max_workgroups_axis: [MAX_GROUPS.min(i64::MAX as u64) as i64; 3],
                max_workgroup_bytes: limits.max_threadgroup_bytes.min(i64::MAX as u64) as i64,
                max_explicit_private_bytes: limits.max_private_bytes.min(i64::MAX as u64) as i64,
                max_direct_bindings: (crate::msl::MAX_KERNEL_BUFFERS - 3) as i64,
                max_argument_table_bytes: 1 << 16,
                max_device_bytes: i64::MAX,
            },
        };
        Self {
            target_profile,
            executable_target_profile,
        }
    }

    #[cfg(target_os = "macos")]
    pub fn from_device(device: &crate::runtime::DeviceInfo) -> Result<Self, String> {
        let limits = Limits::from_device(device);
        let target_profile = crate::target::TargetProfile::from_evidence(
            &device.capability_fingerprint(),
            &device.profile.scalar_dtypes.value,
            &device.profile.matrix_dtypes.value,
            &device.profile.matrix_combinations.value,
            device.max_threads_per_threadgroup,
            device.max_threadgroup_bytes,
            device.max_buffer_bytes,
            device.profile.private_storage_budget_bytes.value,
        );
        Ok(Self::from_profile(limits, target_profile))
    }

    pub fn target_profile(&self) -> &crate::target::TargetProfile {
        &self.target_profile
    }

    /// The effective target profile of the realization layer (exact
    /// signatures and hard limits).
    pub fn executable_profile(&self) -> &EffectiveTargetProfile {
        &self.executable_target_profile
    }
}

#[cfg(target_os = "macos")]
pub struct MetalCompiler<'a> {
    planner: Metal,
    device: &'a crate::runtime::Device,
}

#[cfg(target_os = "macos")]
impl<'a> MetalCompiler<'a> {
    pub fn from_device(device: &'a crate::runtime::Device) -> Result<Self, String> {
        Ok(Self {
            planner: Metal::from_device(&device.info())?,
            device,
        })
    }

    pub fn planner(&self) -> &Metal {
        &self.planner
    }
}

impl seismic_compiler::pipeline::Backend for Metal {
    type Dialect = MetalDialect;
    type EncodedLaunch = crate::msl::EncodedLaunch;
    type NativeArtifact = crate::msl::Emitted;

    fn target(&self) -> &'static str {
        TARGET
    }
    fn capability_fingerprint(&self) -> String {
        self.executable_target_profile
            .capability_fingerprint
            .clone()
    }
    fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String> {
        self.target_profile.supports_intrinsic(intrinsic)
    }
    fn target_profile(&self) -> &EffectiveTargetProfile {
        &self.executable_target_profile
    }
    fn elaborate(&self, logical: &LogicalProgram) -> Result<PlanFamily<Self::Dialect>, String> {
        physical::elaborate(
            logical,
            &self.executable_target_profile,
            StrategyLimits {
                max_participants: self.executable_target_profile.limits.max_participants,
            },
        )
        .map_err(|error| error.to_string())
    }
    fn encode_launch(
        &self,
        launch: &ResolvedLaunch<Self::Dialect>,
    ) -> Result<Self::EncodedLaunch, String> {
        crate::msl::encode_launch(launch)
    }
    fn assemble(
        &self,
        encoded: seismic_compiler::pipeline::EncodedPlan<Self::Dialect, Self::EncodedLaunch>,
    ) -> Result<Self::NativeArtifact, String> {
        crate::msl::assemble(encoded)
    }
}

#[cfg(target_os = "macos")]
impl seismic_compiler::pipeline::Backend for MetalCompiler<'_> {
    type Dialect = MetalDialect;
    type EncodedLaunch = crate::msl::EncodedLaunch;
    type NativeArtifact = crate::runtime::Pipeline;

    fn target(&self) -> &'static str {
        TARGET
    }
    fn capability_fingerprint(&self) -> String {
        self.planner
            .executable_target_profile
            .capability_fingerprint
            .clone()
    }
    fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String> {
        self.planner.target_profile.supports_intrinsic(intrinsic)
    }
    fn target_profile(&self) -> &EffectiveTargetProfile {
        &self.planner.executable_target_profile
    }
    fn elaborate(&self, logical: &LogicalProgram) -> Result<PlanFamily<Self::Dialect>, String> {
        seismic_compiler::pipeline::Backend::elaborate(&self.planner, logical)
    }
    fn encode_launch(
        &self,
        launch: &ResolvedLaunch<Self::Dialect>,
    ) -> Result<Self::EncodedLaunch, String> {
        crate::msl::encode_launch(launch)
    }
    fn assemble(
        &self,
        encoded: seismic_compiler::pipeline::EncodedPlan<Self::Dialect, Self::EncodedLaunch>,
    ) -> Result<Self::NativeArtifact, String> {
        let emitted = seismic_compiler::pipeline::Backend::assemble(&self.planner, encoded)?;
        self.device.compile(emitted)
    }
}
