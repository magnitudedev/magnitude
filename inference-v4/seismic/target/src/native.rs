use crate::{CompatibilityIdentity, DeviceDescription, NativeCompilationError, TargetFamily};
use seismic_ir::kernel::Kernel;
use seismic_ir::target::{KernelAbiLayout, KernelEmissionLayout};
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClusterPortability {
    Portable,
    NonPortableAllowed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeClusterDomain {
    NotApplicable,
    Required {
        dimensions: [u32; 3],
        portability: ClusterPortability,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeLaunchDomain<M> {
    pub modes: Vec<M>,
    pub subgroup_width: Option<u32>,
    pub cluster: NativeClusterDomain,
    pub max_grid: [u64; 3],
    pub max_workgroup_size: [u64; 3],
    pub max_workgroup_threads: u64,
    pub max_dynamic_local_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeResourceUsage {
    NotApplicable,
    Exact(u64),
    EnforcedUpperBound(u64),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeResources {
    pub registers_per_participant: NativeResourceUsage,
    pub spill_bytes_per_participant: NativeResourceUsage,
    pub static_local_bytes: NativeResourceUsage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NativeArtifactMetrics {
    pub compilation_ns: u64,
    pub code_bytes: u64,
    pub metadata_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeKernelIdentity {
    pub compatibility: CompatibilityIdentity,
    pub artifact_digest: [u8; 32],
}

/// Immutable facts reflected from one concrete compiled kernel.
///
/// This value is cloneable and comparable without access to the executable
/// handle. It supports reconciliation, native legality, execution metadata,
/// diagnostics, and qualification-side observation. Production prediction is
/// pre-native and does not consume this type.
#[derive(Debug)]
pub struct NativeKernelDescription<T: TargetFamily> {
    pub identity: NativeKernelIdentity,
    pub abi: KernelAbiLayout,
    pub launch: NativeLaunchDomain<T::LaunchDescriptor>,
    pub resources: NativeResources,
    pub numerics: T::NativeNumericalMode,
    pub numerical_identity: crate::NativeNumericalModeIdentity,
    pub properties: T::NativeProperties,
}

impl<T: TargetFamily> Clone for NativeKernelDescription<T> {
    fn clone(&self) -> Self {
        Self {
            identity: self.identity.clone(),
            abi: self.abi.clone(),
            launch: self.launch.clone(),
            resources: self.resources.clone(),
            numerics: self.numerics.clone(),
            numerical_identity: self.numerical_identity.clone(),
            properties: self.properties.clone(),
        }
    }
}

impl<T: TargetFamily> PartialEq for NativeKernelDescription<T> {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
            && self.abi == other.abi
            && self.launch == other.launch
            && self.resources == other.resources
            && self.numerics == other.numerics
            && self.numerical_identity == other.numerical_identity
            && self.properties == other.properties
    }
}

impl<T: TargetFamily> Eq for NativeKernelDescription<T> {}

impl<T: TargetFamily> Hash for NativeKernelDescription<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
        self.abi.hash(state);
        self.launch.hash(state);
        self.resources.hash(state);
        self.numerics.hash(state);
        self.numerical_identity.hash(state);
        self.properties.hash(state);
    }
}

/// A backend reflection result.  Its handle cannot be observed before core
/// reconciliation consumes this value.
pub struct NativeKernelReflection<T: TargetFamily, H> {
    handle: H,
    description: NativeKernelDescription<T>,
    metrics: NativeArtifactMetrics,
}

impl<T: TargetFamily, H> NativeKernelReflection<T, H> {
    pub fn new(
        handle: H,
        description: NativeKernelDescription<T>,
        metrics: NativeArtifactMetrics,
    ) -> Self {
        Self {
            handle,
            description,
            metrics,
        }
    }

    pub fn description(&self) -> &NativeKernelDescription<T> {
        &self.description
    }

    pub fn metrics(&self) -> NativeArtifactMetrics {
        self.metrics
    }
}

impl<T: TargetFamily, H> fmt::Debug for NativeKernelReflection<T, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeKernelReflection")
            .field("description", &self.description)
            .field("metrics", &self.metrics)
            .finish_non_exhaustive()
    }
}

/// Native artifact formation and authoritative reflection.
///
/// The implementation is a service object, not the target marker.  It needs
/// neither an estimator nor an executor implementation.
pub trait NativeCompiler<T: TargetFamily> {
    type Context;
    type Candidate: Send + 'static;
    type Handle: Send + Sync + 'static;

    fn form(
        &self,
        context: &Self::Context,
        target: &DeviceDescription<T>,
        kernel: &Kernel<T>,
        layout: &KernelEmissionLayout,
    ) -> Result<Self::Candidate, NativeCompilationError>;

    fn reflect(
        &self,
        target: &DeviceDescription<T>,
        kernel: &Kernel<T>,
        layout: &KernelEmissionLayout,
        candidate: Self::Candidate,
    ) -> Result<NativeKernelReflection<T, Self::Handle>, NativeCompilationError>;
}

/// Unusable native output plus the exact contract against which it was formed.
pub struct NativeKernelCandidate<'a, T: TargetFamily, C> {
    raw: C,
    target: &'a DeviceDescription<T>,
    kernel: &'a Kernel<T>,
    layout: KernelEmissionLayout,
    abi: KernelAbiLayout,
    compatibility: CompatibilityIdentity,
    _target: std::marker::PhantomData<fn() -> T>,
}

/// Forms native code without granting access to an executable handle.
pub fn form_native_kernel<'a, T, C>(
    compiler: &C,
    context: &C::Context,
    target: &'a DeviceDescription<T>,
    kernel: &'a Kernel<T>,
) -> Result<NativeKernelCandidate<'a, T, C::Candidate>, NativeCompilationError>
where
    T: TargetFamily,
    C: NativeCompiler<T>,
{
    let layout = target.kernel_emission_layout(kernel);
    let abi = target.kernel_abi_layout(kernel);
    let raw = compiler.form(context, target, kernel, &layout)?;
    Ok(NativeKernelCandidate {
        raw,
        target,
        kernel,
        layout,
        abi,
        compatibility: target.compatibility_identity().clone(),
        _target: std::marker::PhantomData,
    })
}

/// The executable native artifact.  Descriptive facts are available without
/// exposing or cloning the backend-private handle.
pub struct NativeKernel<T: TargetFamily, H> {
    handle: H,
    description: NativeKernelDescription<T>,
}

impl<T: TargetFamily, H> NativeKernel<T, H> {
    pub fn description(&self) -> &NativeKernelDescription<T> {
        &self.description
    }

    /// Execution adapters alone need this backend-private typed handle.
    pub fn handle(&self) -> &H {
        &self.handle
    }

    /// Splits storage ownership for composition roots that keep executable
    /// handles in a registry separate from immutable candidate-domain facts.
    pub fn into_parts(self) -> (H, NativeKernelDescription<T>) {
        (self.handle, self.description)
    }
}

impl<T: TargetFamily, H> fmt::Debug for NativeKernel<T, H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeKernel")
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

/// Reconciled executable plus preparation cost.  Compilation metrics are not
/// part of the immutable kernel description or its identity.
pub struct NativeFormation<T: TargetFamily, H> {
    kernel: NativeKernel<T, H>,
    metrics: NativeArtifactMetrics,
}

impl<T: TargetFamily, H> NativeFormation<T, H> {
    pub fn kernel(&self) -> &NativeKernel<T, H> {
        &self.kernel
    }

    pub fn metrics(&self) -> NativeArtifactMetrics {
        self.metrics
    }

    pub fn into_parts(self) -> (NativeKernel<T, H>, NativeArtifactMetrics) {
        (self.kernel, self.metrics)
    }
}

/// Reflects and reconciles one candidate.  The candidate is consumed and no
/// executable value exists if reflection contradicts immutable target truth.
pub fn reconcile_native_kernel<T, C>(
    compiler: &C,
    candidate: NativeKernelCandidate<'_, T, C::Candidate>,
) -> Result<NativeFormation<T, C::Handle>, NativeCompilationError>
where
    T: TargetFamily,
    C: NativeCompiler<T>,
{
    let NativeKernelCandidate {
        raw,
        target,
        kernel,
        layout,
        abi,
        compatibility,
        _target: _,
    } = candidate;
    let reflection = compiler.reflect(target, kernel, &layout, raw)?;
    let NativeKernelReflection {
        handle,
        description,
        metrics,
    } = reflection;
    validate_reflection(target, kernel, &abi, &compatibility, &description)?;
    if metrics.code_bytes == 0 {
        return Err(NativeCompilationError::MalformedToolchainOutput(
            "reflection reports an empty native code artifact".into(),
        ));
    }
    Ok(NativeFormation {
        kernel: NativeKernel {
            handle,
            description,
        },
        metrics,
    })
}

fn validate_reflection<T: TargetFamily>(
    target: &DeviceDescription<T>,
    kernel: &Kernel<T>,
    abi: &KernelAbiLayout,
    compatibility: &CompatibilityIdentity,
    description: &NativeKernelDescription<T>,
) -> Result<(), NativeCompilationError> {
    fn malformed(message: impl Into<String>) -> NativeCompilationError {
        NativeCompilationError::MalformedToolchainOutput(message.into())
    }
    if &description.identity.compatibility != compatibility {
        return Err(malformed(
            "reflection changed the native compatibility identity",
        ));
    }
    if &description.abi != abi {
        return Err(malformed(
            "reflection disagrees with the canonical kernel ABI",
        ));
    }
    if description.launch.modes.is_empty() {
        return Err(malformed("reflection admits no launch mode"));
    }
    if kernel.interface().uses_subgroup && description.launch.subgroup_width.is_none() {
        return Err(malformed(
            "subgroup-using kernel reflection omitted its subgroup width",
        ));
    }
    let launch = &description.launch;
    if launch.max_workgroup_threads == 0
        || launch.max_workgroup_size.contains(&0)
        || launch.max_grid.contains(&0)
        || launch.subgroup_width == Some(0)
    {
        return Err(malformed("reflection contains an empty launch domain"));
    }
    if let NativeClusterDomain::Required { dimensions, .. } = launch.cluster {
        if dimensions.contains(&0) {
            return Err(malformed("reflection contains an empty cluster dimension"));
        }
    }
    let limits = target.limits();
    if launch.max_workgroup_threads > limits.max_workgroup_threads
        || launch
            .max_workgroup_size
            .iter()
            .zip(limits.max_workgroup_size)
            .any(|(native, device)| *native > device)
        || launch
            .max_grid
            .iter()
            .zip(limits.max_grid)
            .any(|(native, device)| *native > device)
        || launch.max_dynamic_local_bytes > limits.max_workgroup_bytes
    {
        return Err(malformed("reflection exceeds a device-wide hard limit"));
    }
    let static_local = match description.resources.static_local_bytes {
        NativeResourceUsage::NotApplicable => 0,
        NativeResourceUsage::Exact(bytes) | NativeResourceUsage::EnforcedUpperBound(bytes) => bytes,
    };
    if static_local
        .checked_add(launch.max_dynamic_local_bytes)
        .is_none_or(|total| total > limits.max_workgroup_bytes)
    {
        return Err(malformed(
            "reflected static and dynamic local memory exceed the device limit",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DeviceDescriptionIdentity, DeviceDescriptionParts, NativeNumericalModeIdentity,
        NumericalEnvironmentIdentity, TargetDescriptionError,
    };
    use seismic_ir::kernel::ops::AddressableResourceHandle;
    use seismic_ir::target::{
        IntrinsicIdentityBuilder, IntrinsicNumericalSemantics, KernelAbiFootprint, KernelAbiModel,
        LocalRealization, LocalRealizationPolicy, NumericalEnvironment, TargetLimits,
    };
    use seismic_lang::registry::{BackendName, IntrinsicSignature};
    use std::collections::BTreeSet;

    #[derive(Debug)]
    struct FakeTarget;

    #[derive(Clone, Debug, PartialEq)]
    struct FakeFacts;

    impl seismic_ir::target::PhysicalDialect for FakeTarget {
        type LaunchDescriptor = ();
        fn ordinary_launch() -> Self::LaunchDescriptor {
            ()
        }

        const NAME: BackendName = BackendName::Cpu;
        type Intrinsic = ();
        type Facts = FakeFacts;

        fn write_intrinsic_identity(_: &(), _: &mut IntrinsicIdentityBuilder) {}

        fn intrinsic_numerics(
            _: &Self::Facts,
            signature: &IntrinsicSignature,
            _: &(),
        ) -> IntrinsicNumericalSemantics {
            IntrinsicNumericalSemantics {
                arithmetic: signature.numerical.clone(),
                flush_to_zero: false,
            }
        }

        fn intrinsic_addressable_resources(_: &()) -> Vec<AddressableResourceHandle> {
            Vec::new()
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct FakeAbi;

    impl KernelAbiModel<FakeTarget> for FakeAbi {
        fn layout(&self, _: &Kernel<FakeTarget>) -> KernelAbiLayout {
            KernelAbiLayout {
                footprint: KernelAbiFootprint {
                    bytes: 0,
                    alignment: 1,
                },
                allocations: Vec::new(),
            }
        }
    }

    impl TargetFamily for FakeTarget {
        type KernelAbi = FakeAbi;
        type NativeNumericalMode = ();
        type NativeProperties = ();
    }

    struct CompilerOnly;

    impl NativeCompiler<FakeTarget> for CompilerOnly {
        type Context = ();
        type Candidate = Vec<u8>;
        type Handle = Box<[u8]>;

        fn form(
            &self,
            _: &(),
            _: &DeviceDescription<FakeTarget>,
            _: &Kernel<FakeTarget>,
            _: &KernelEmissionLayout,
        ) -> Result<Self::Candidate, NativeCompilationError> {
            Ok(Vec::new())
        }

        fn reflect(
            &self,
            _: &DeviceDescription<FakeTarget>,
            _: &Kernel<FakeTarget>,
            _: &KernelEmissionLayout,
            _: Self::Candidate,
        ) -> Result<NativeKernelReflection<FakeTarget, Self::Handle>, NativeCompilationError>
        {
            unreachable!("compile-time contract test does not invoke native reflection")
        }
    }

    fn description_parts() -> DeviceDescriptionParts<FakeTarget> {
        DeviceDescriptionParts {
            identity: DeviceDescriptionIdentity {
                backend: BackendName::Cpu,
                fingerprint: [1; 32],
            },
            compatibility: CompatibilityIdentity {
                backend: BackendName::Cpu,
                backend_revision: "fake-v1",
                hardware: "fake".into(),
                driver: "fake".into(),
                toolchain: "fake".into(),
                fingerprint: [2; 32],
            },
            numerical_environment: NumericalEnvironmentIdentity {
                backend: BackendName::Cpu,
                fingerprint: [3; 32],
            },
            limits: TargetLimits {
                max_workgroup_size: [1, 1, 1],
                max_workgroup_threads: 1,
                max_grid: [1, 1, 1],
                max_workgroup_bytes: 1,
                participant_local_bytes: None,
                max_bindings: 1,
                max_argument_bytes: 1,
                max_allocation_bytes: 1,
                max_allocation_alignment: 1,
                max_index_bits: 1,
                subgroup_width: None,
            },
            dtypes: seismic_ir::target::DataTypeSupport {
                scalars: BTreeSet::new(),
                atomics: BTreeSet::new(),
                representations: BTreeSet::new(),
            },
            vectors: seismic_ir::target::VectorSupport::default(),
            numerics: NumericalEnvironment {
                contraction_available: false,
                flush_to_zero_available: false,
                approximate_transcendentals: BTreeSet::new(),
                denormals_preserved: true,
            },
            capabilities: BTreeSet::new(),
            intrinsics: BTreeSet::new(),
            facts: FakeFacts,
            kernel_abi: FakeAbi,
            local_realization: LocalRealizationPolicy {
                workgroup: LocalRealization::NativeDynamic,
                participant: LocalRealization::NativeDynamic,
                register: LocalRealization::NativeDynamic,
            },
            addressable_resources: Vec::new(),
        }
    }

    #[test]
    fn native_compiler_requires_no_estimator_or_executor_contract() {
        fn accepts<C: NativeCompiler<FakeTarget>>() {}
        accepts::<CompilerOnly>();
    }

    #[test]
    fn immutable_description_rejects_another_backend_identity() {
        let mut parts = description_parts();
        parts.compatibility.backend = BackendName::Metal;
        assert_eq!(
            DeviceDescription::new(parts).unwrap_err(),
            TargetDescriptionError::BackendIdentityMismatch
        );
    }

    #[test]
    fn executable_handle_is_not_part_of_the_reflected_description() {
        struct SecretHandle;

        let description = NativeKernelDescription::<FakeTarget> {
            identity: NativeKernelIdentity {
                compatibility: description_parts().compatibility,
                artifact_digest: [4; 32],
            },
            abi: KernelAbiLayout {
                footprint: KernelAbiFootprint {
                    bytes: 0,
                    alignment: 1,
                },
                allocations: Vec::new(),
            },
            launch: NativeLaunchDomain {
                modes: vec![()],
                subgroup_width: None,
                cluster: NativeClusterDomain::NotApplicable,
                max_grid: [1, 1, 1],
                max_workgroup_size: [1, 1, 1],
                max_workgroup_threads: 1,
                max_dynamic_local_bytes: 0,
            },
            resources: NativeResources {
                registers_per_participant: NativeResourceUsage::NotApplicable,
                spill_bytes_per_participant: NativeResourceUsage::NotApplicable,
                static_local_bytes: NativeResourceUsage::NotApplicable,
            },
            numerics: (),
            numerical_identity: NativeNumericalModeIdentity {
                fingerprint: [5; 32],
            },
            properties: (),
        };
        let metrics = NativeArtifactMetrics {
            compilation_ns: 1,
            code_bytes: 2,
            metadata_bytes: 3,
        };
        let reflection = NativeKernelReflection::new(SecretHandle, description.clone(), metrics);
        assert_eq!(reflection.description(), &description);
        assert_eq!(reflection.metrics(), metrics);
        assert!(format!("{reflection:?}").contains("NativeKernelReflection"));
    }

    #[test]
    fn device_and_artifact_numerical_identities_are_distinct_types() {
        fn accepts_environment(_: &NumericalEnvironmentIdentity) {}
        fn accepts_artifact_mode(_: &NativeNumericalModeIdentity) {}

        let target = DeviceDescription::new(description_parts()).unwrap();
        accepts_environment(target.numerical_environment_identity());
        let artifact = NativeNumericalModeIdentity {
            fingerprint: [8; 32],
        };
        accepts_artifact_mode(&artifact);
    }
}
