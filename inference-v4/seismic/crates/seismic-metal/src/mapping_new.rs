//! Metal target configuration for executable planning.

#[path = "mapping/estimate.rs"]
mod estimate;
pub use estimate::{EstimateModel, Group, Totals, IDENTITY};

use seismic_lang::sir::IntrinsicUse;

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
}

pub struct Metal {
    pub(crate) limits: Limits,
    pub(crate) target_profile: crate::target::TargetProfile,
    pub(crate) executable_target_profile:
        seismic_realization::executable::ExecutableTargetProfile<crate::physical::MetalCapability>,
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

fn physical_capabilities(
    target: &crate::target::TargetProfile,
) -> std::collections::BTreeSet<crate::physical::MetalCapability> {
    let mut capabilities =
        std::collections::BTreeSet::from([crate::physical::MetalCapability::Scalar]);
    if target.supports_family("metal.subgroup") {
        capabilities.insert(crate::physical::MetalCapability::Simdgroup);
    }
    if target.supports_family("metal.matrix") {
        capabilities.insert(crate::physical::MetalCapability::SimdgroupMatrix);
    }
    if target.supports_dtype(seismic_lang::types::DType::BF16) {
        capabilities.insert(crate::physical::MetalCapability::BFloat);
    }
    capabilities
}

impl Metal {
    pub fn new(limits: Limits, estimate: EstimateModel) -> Result<Self, String> {
        if limits.max_threads_per_threadgroup < SUBGROUP as u64
            || i64::try_from(limits.max_threads_per_threadgroup).is_err()
            || i64::try_from(limits.max_threadgroup_bytes).is_err()
        {
            return Err(format!(
                "Metal needs at least {} threads per threadgroup; the target offers {}",
                SUBGROUP, limits.max_threads_per_threadgroup
            ));
        }
        estimate.validate()?;
        let target_profile = crate::target::TargetProfile::synthetic(
            limits.max_threads_per_threadgroup,
            limits.max_threadgroup_bytes,
            u64::MAX,
            limits.max_private_bytes,
        );
        let executable_target_profile = seismic_realization::executable::ExecutableTargetProfile {
            target: TARGET.into(),
            capability_fingerprint: target_profile.fingerprint().into(),
            toolchain_fingerprint: format!(
                "{};{}",
                crate::target::BACKEND_IMPLEMENTATION_REVISION,
                crate::target::COMPILER_PROBE_REVISION
            ),
            limits: seismic_realization::executable::ExecutableTargetLimits {
                max_allocation_bytes: i64::MAX as u64,
                max_device_bytes: i64::MAX as u64,
                max_workgroup_bytes: limits.max_threadgroup_bytes,
                max_private_bytes_per_participant: limits.max_private_bytes,
                max_bindings_per_launch: crate::msl::MAX_KERNEL_BUFFERS as u64,
                max_registers_per_kernel: i64::MAX as u64,
                max_workgroups: [MAX_GROUPS; 3],
                max_participants_per_workgroup: limits.max_threads_per_threadgroup,
            },
            capabilities: physical_capabilities(&target_profile),
        };
        Ok(Self {
            limits,
            target_profile,
            executable_target_profile,
        })
    }

    #[cfg(target_os = "macos")]
    pub fn from_device(device: &crate::runtime::DeviceInfo) -> Result<Self, String> {
        let mut backend = Self::new(
            Limits::from_device(device),
            EstimateModel::from_device(device),
        )?;
        backend.target_profile = crate::target::TargetProfile::from_evidence(
            &device.capability_fingerprint(),
            &device.profile.scalar_dtypes.value,
            &device.profile.matrix_dtypes.value,
            &device.profile.matrix_combinations.value,
            device.max_threads_per_threadgroup,
            device.max_threadgroup_bytes,
            device.max_buffer_bytes,
            device.profile.private_storage_budget_bytes.value,
        );
        backend.executable_target_profile.capability_fingerprint =
            backend.target_profile.fingerprint().into();
        backend.executable_target_profile.capabilities =
            physical_capabilities(&backend.target_profile);
        backend
            .executable_target_profile
            .limits
            .max_allocation_bytes = device.max_buffer_bytes;
        Ok(backend)
    }

    pub fn target_profile(&self) -> &crate::target::TargetProfile {
        &self.target_profile
    }
}

impl seismic_compiler::pipeline::Backend for Metal {
    type Dialect = crate::physical::MetalDialect;
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
    fn target_profile(
        &self,
    ) -> &seismic_realization::executable::ExecutableTargetProfile<
        <Self::Dialect as seismic_realization::executable::ExecutableDialect>::Capability,
    > {
        &self.executable_target_profile
    }
    fn elaborate(
        &self,
        logical: &seismic_lang::logical::LogicalProgram,
    ) -> Result<seismic_realization::executable::PlanFamily<Self::Dialect>, String> {
        crate::physical::elaborate(logical, self.limits.max_threads_per_threadgroup)
    }
    fn encode_launch(
        &self,
        launch: &seismic_realization::executable::ResolvedLaunch<Self::Dialect>,
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
    type Dialect = crate::physical::MetalDialect;
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
    fn target_profile(
        &self,
    ) -> &seismic_realization::executable::ExecutableTargetProfile<
        <Self::Dialect as seismic_realization::executable::ExecutableDialect>::Capability,
    > {
        &self.planner.executable_target_profile
    }
    fn elaborate(
        &self,
        logical: &seismic_lang::logical::LogicalProgram,
    ) -> Result<seismic_realization::executable::PlanFamily<Self::Dialect>, String> {
        crate::physical::elaborate(logical, self.planner.limits.max_threads_per_threadgroup)
    }
    fn encode_launch(
        &self,
        launch: &seismic_realization::executable::ResolvedLaunch<Self::Dialect>,
    ) -> Result<Self::EncodedLaunch, String> {
        crate::msl::encode_launch(launch)
    }
    fn assemble(
        &self,
        encoded: seismic_compiler::pipeline::EncodedPlan<Self::Dialect, Self::EncodedLaunch>,
    ) -> Result<Self::NativeArtifact, String> {
        self.device.compile(crate::msl::assemble(encoded)?)
    }
}
