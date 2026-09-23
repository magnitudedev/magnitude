//! Backend device truth, compiler policy, analytical profiles, and execution association.
//!
//! Machine truth has three non-interchangeable owners: `DeviceDescription<T>`
//! for device-wide legality and numerics, reflection-reconciled target-owned
//! native-kernel descriptions for concrete function facts, and
//! `ExecutionProfile<T>` for measured performance. Only closed native-kernel
//! descriptions may enter planning; opaque handles remain in the realization
//! registry.
//!
//! Registry assembly is target-family generic. Each backend crate supplies its
//! compiler-policy registrations separately from native formation, analytical
//! estimation, and execution services.

use seismic_ir::target::*;
use seismic_target::{
    CompatibilityIdentity, DeviceDescription, DeviceDescriptionIdentity,
    NumericalEnvironmentIdentity,
};

use crate::errors::TargetError;
use seismic_estimator::*;
use seismic_lang::expr::{ExprArena, SymbolId, SymbolSort, SymbolValue, TargetConstantId};
use seismic_lang::ids::{CapabilityId, IntrinsicId};
use seismic_lang::types::DType;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

/// Per-open profile identity. Timing observations and probe methodology live
/// here, never in the stable native-code compatibility identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExecutionProfileIdentity {
    pub device: DeviceDescriptionIdentity,
    pub probe_suite_revision: &'static str,
    pub fingerprint: [u8; 32],
}

/// Device-wide facts gathered from the exact opened service. Compiler closure
/// validates these facts and derives the capability sets in the immutable target
/// description.
#[derive(Debug)]
pub struct DiscoveredTarget<T: seismic_target::TargetFamily> {
    pub identity: CompatibilityIdentity,
    pub limits: TargetLimits,
    pub dtypes: DataTypeSupport,
    pub vectors: VectorSupport,
    pub numerics: NumericalEnvironment,
    pub facts: T::Facts,
    pub kernel_abi: T::KernelAbi,
    pub local_realization: LocalRealizationPolicy,
}

/// Per-open measured execution behavior. This type owns no legality,
/// capability, numerical, ABI, or native-kernel facts.
#[derive(Debug)]
pub struct ExecutionProfile<T: seismic_target::TargetFamily> {
    identity: ExecutionProfileIdentity,
    services: Vec<ServiceDefinition>,
    acquisition: ProfileAcquisitionMetrics,
    composition_qualification: CompositionQualification,
    _target: std::marker::PhantomData<fn() -> T>,
}

#[derive(Debug)]
pub struct ExecutionProfileParts<T: seismic_target::TargetFamily> {
    device: Arc<DeviceDescription<T>>,
    probe_suite_revision: &'static str,
    services: Vec<ServiceDefinition>,
    acquisition: ProfileAcquisitionMetrics,
    composition_qualification: CompositionQualificationParts,
}

impl<T: seismic_target::TargetFamily> ExecutionProfileParts<T> {
    pub fn new(
        device: Arc<DeviceDescription<T>>,
        probe_suite_revision: &'static str,
        services: Vec<ServiceDefinition>,
        acquisition: ProfileAcquisitionMetrics,
        composition_qualification: CompositionQualificationParts,
    ) -> Self {
        Self {
            device,
            probe_suite_revision,
            services,
            acquisition,
            composition_qualification,
        }
    }

    pub(crate) fn device(&self) -> &Arc<DeviceDescription<T>> {
        &self.device
    }
}

/// Closes discovered target facts against the independently owned compiler
/// registry and returns immutable target truth. The registry is consumed only
/// while deriving supported capabilities and is not retained by the result.
pub fn assemble_device_description<T: seismic_target::TargetFamily>(
    discovered: DiscoveredTarget<T>,
    registry: &CompilerRegistry<T>,
) -> Result<DeviceDescription<T>, TargetError> {
    internals::assemble(discovered, registry)
}

/// Binds every target constant the compiler may reference into one arena.
pub fn bind_target_constants<T: seismic_target::TargetFamily>(
    device: &DeviceDescription<T>,
    arena: &mut ExprArena,
) -> TargetConstants {
    internals::bind_constants(device, arena)
}

impl<T: seismic_target::TargetFamily> ExecutionProfile<T> {
    pub(crate) fn assemble(
        required_services: BTreeSet<ServiceClassId>,
        parts: ExecutionProfileParts<T>,
    ) -> Result<Self, TargetError> {
        internals::assemble_execution(required_services, parts)
    }

    pub fn identity(&self) -> &ExecutionProfileIdentity {
        &self.identity
    }

    pub fn device_identity(&self) -> &DeviceDescriptionIdentity {
        &self.identity.device
    }

    pub(crate) fn service_by_class(&self, class: ServiceClassId) -> &ServiceDefinition {
        self.services
            .iter()
            .find(|definition| definition.class == class)
            .expect("certified service reference is absent from its bound profile")
    }

    pub fn services(&self) -> &[ServiceDefinition] {
        &self.services
    }

    pub fn acquisition_metrics(&self) -> ProfileAcquisitionMetrics {
        self.acquisition
    }

    pub fn composition_qualification(&self) -> &CompositionQualification {
        &self.composition_qualification
    }
}

/// Target constants bound into an arena. Every hard limit that enters a
/// constraint is one of these symbols.
#[derive(Clone, Debug)]
pub struct TargetConstants {
    pub max_workgroup_size: [TargetConstantId; 3],
    pub max_workgroup_threads: TargetConstantId,
    pub max_workgroup_bytes: TargetConstantId,
    pub participant_local_bytes: Option<TargetConstantId>,
    pub max_grid: [TargetConstantId; 3],
    pub max_bindings: TargetConstantId,
    pub max_argument_bytes: TargetConstantId,
    pub max_allocation_bytes: TargetConstantId,
    pub max_allocation_alignment: TargetConstantId,
    pub max_index_bits: TargetConstantId,
    pub subgroup_width: Option<TargetConstantId>,
    addressable_resource_capacity: Vec<TargetConstantId>,
    bindings: Vec<(SymbolId, SymbolValue)>,
}

impl TargetConstants {
    /// Concrete target values used to partially evaluate every physical
    /// expression before a `FrozenPlan` is formed.  Native artifacts never
    /// rediscover or rebind device facts.
    pub fn bindings(&self) -> &[(SymbolId, SymbolValue)] {
        &self.bindings
    }
    pub(crate) fn addressable_resource_capacity(&self, id: ResourceClassId) -> TargetConstantId {
        self.addressable_resource_capacity[id.ordinal() as usize]
    }
}

/// One complete intrinsic implementation row. The type requires both the
/// launch-requirement callback and lowering callback for every registered id.
pub struct IntrinsicImplementation<T: seismic_target::TargetFamily> {
    pub id: IntrinsicId,
    pub launch_requirements: fn(
        &DeviceDescription<T>,
        &mut ExprArena,
        &seismic_lang::registry::IntrinsicSignature,
        seismic_lang::expr::NatExpr,
    )
        -> seismic_ir::kernel::ops::SemanticIntrinsicLaunchRequirements,
    pub lower: fn(
        &DeviceDescription<T>,
        &seismic_ir::kernel::ops::SegmentLaunchDomain,
        seismic_ir::kernel::ops::SemanticIntrinsicCall<'_>,
        &mut seismic_ir::kernel::ops::SemanticIntrinsicSink<'_, '_, T>,
    ),
}

impl<T: seismic_target::TargetFamily> fmt::Debug for IntrinsicImplementation<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntrinsicImplementation")
            .field("id", &self.id)
            .finish()
    }
}

/// One capability registration: its complete intrinsic implementation rows
/// and the target predicate that selects a subset for one device.
pub struct CapabilityRegistration<T: seismic_target::TargetFamily> {
    pub capability: CapabilityId,
    /// Every intrinsic implementation owned by this capability.
    pub implementations: Vec<IntrinsicImplementation<T>>,
    /// Which implementation ids this target profile supports, given the facts.
    /// May only narrow `implementations`; an empty result means the capability
    /// is not advertised on this target.
    pub supported: fn(&T::Facts, &TargetLimits, &DataTypeSupport) -> BTreeSet<IntrinsicId>,
}

impl<T: seismic_target::TargetFamily> fmt::Debug for CapabilityRegistration<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CapabilityRegistration")
            .field("capability", &self.capability)
            .finish()
    }
}

/// The sealed static registry of one backend, assembled once at compiler
/// initialization. Duplicate ids, a signature without an emitter, or an
/// emitter without a signature are startup panics (§4.2, §13.3.1).
pub struct CompilerRegistryParts<T: seismic_target::TargetFamily> {
    pub capabilities: Vec<CapabilityRegistration<T>>,

    pub native_launch_constraints: fn(
        &DeviceDescription<T>,
        &mut ExprArena,
        &seismic_ir::schedule::Launch<T>,
        &seismic_ir::storage::LaunchLocalLayout,
        &seismic_ir::kernel::Kernel<T>,
        &seismic_target::NativeKernelDescription<T>,
    ) -> Vec<seismic_lang::expr::BoolExpr>,
    pub addressable_resources: fn(&T::Facts) -> Vec<AddressableResourceClass>,
    pub emitted_intrinsics: BTreeSet<IntrinsicId>,
}

pub struct CompilerRegistry<T: seismic_target::TargetFamily> {
    inner: internals::Registry<T>,
}

impl<T: seismic_target::TargetFamily> CompilerRegistry<T> {
    /// Assembles and seals. Panics on an inconsistent registration set.
    pub fn assemble(parts: CompilerRegistryParts<T>) -> Self {
        Self {
            inner: internals::Registry::assemble(parts),
        }
    }

    /// A capability from the backend's sealed intrinsic vocabulary.
    pub fn capability(&self, id: CapabilityId) -> Option<&CapabilityRegistration<T>> {
        self.inner.capability(id)
    }

    pub fn intrinsic(&self, id: IntrinsicId) -> Option<&IntrinsicImplementation<T>> {
        self.inner.intrinsic(id)
    }

    pub fn native_launch_constraints(
        &self,
        target: &DeviceDescription<T>,
        arena: &mut ExprArena,
        launch: &seismic_ir::schedule::Launch<T>,
        locals: &seismic_ir::storage::LaunchLocalLayout,
        kernel: &seismic_ir::kernel::Kernel<T>,
        native: &seismic_target::NativeKernelDescription<T>,
    ) -> Vec<seismic_lang::expr::BoolExpr> {
        (self.inner.native_launch_constraints)(target, arena, launch, locals, kernel, native)
    }
}

impl<T: seismic_target::TargetFamily> fmt::Debug for CompilerRegistry<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompilerRegistry").finish()
    }
}

mod internals {
    //! Registry and profile assembly (W5-shared, performed by the CPU lane
    //! per §24.5 R12).
    //!
    //! Every check here is over values assembled by compiler developers
    //! (static registrations) or gathered by a backend's own discovery. A
    //! contradiction in a registration set is a startup panic (§13.3.1). A
    //! device below the backend's floor is the typed `TargetError`.

    use super::*;
    use seismic_lang::registry;

    pub(super) struct Registry<T: seismic_target::TargetFamily> {
        capabilities: Vec<CapabilityRegistration<T>>,

        pub(super) native_launch_constraints: fn(
            &DeviceDescription<T>,
            &mut ExprArena,
            &seismic_ir::schedule::Launch<T>,
            &seismic_ir::storage::LaunchLocalLayout,
            &seismic_ir::kernel::Kernel<T>,
            &seismic_target::NativeKernelDescription<T>,
        ) -> Vec<seismic_lang::expr::BoolExpr>,
        addressable_resources: fn(&T::Facts) -> Vec<AddressableResourceClass>,
    }

    impl<T: seismic_target::TargetFamily> Registry<T> {
        /// Seals one backend's registrations. Panics (§13.3.1) on:
        /// a capability registered twice; a capability of another backend;
        /// a capability advertised with no implemented signature; an
        /// implemented signature that is not a registered signature of that
        /// capability (an emitter without a signature); a signature
        /// implemented twice.
        pub(super) fn assemble(parts: CompilerRegistryParts<T>) -> Self {
            let CompilerRegistryParts {
                capabilities,

                native_launch_constraints,
                addressable_resources,
                emitted_intrinsics,
            } = parts;
            let mut registered = BTreeSet::new();
            for (position, registration) in capabilities.iter().enumerate() {
                let capability = registration.capability;
                if capabilities[..position]
                    .iter()
                    .any(|earlier| earlier.capability == capability)
                {
                    panic!(
                        "CompilerRegistry<{:?}>: capability {capability:?} is registered twice",
                        T::NAME
                    );
                }
                let info = registry::capability_info(capability);
                if info.backend != T::NAME {
                    panic!(
                        "CompilerRegistry<{:?}>: capability {capability:?} (`{}`) belongs to backend {:?}",
                        T::NAME,
                        info.name,
                        info.backend
                    );
                }
                if registration.implementations.is_empty() {
                    panic!(
                        "CompilerRegistry<{:?}>: capability `{}` is advertised without an implemented signature",
                        T::NAME,
                        info.name
                    );
                }
                let signatures = registry::intrinsics(capability);
                for implementation in &registration.implementations {
                    let intrinsic = implementation.id;
                    if !registered.insert(intrinsic) {
                        panic!(
                            "CompilerRegistry<{:?}>: signature {intrinsic:?} of `{}` is implemented twice",
                            T::NAME,
                            info.name
                        );
                    }
                    if !signatures.iter().any(|signature| signature.id == intrinsic) {
                        panic!(
                            "CompilerRegistry<{:?}>: `{}` implements {intrinsic:?}, which is not a registered signature of that capability (an emitter without a signature)",
                            T::NAME,
                            info.name
                        );
                    }
                }
            }
            assert_eq!(
                registered,
                emitted_intrinsics,
                "CompilerRegistry<{:?}>: registered typed intrinsic signatures and native emitter signatures differ",
                T::NAME
            );
            Self {
                capabilities,

                native_launch_constraints,
                addressable_resources,
            }
        }

        pub(super) fn capability(&self, id: CapabilityId) -> Option<&CapabilityRegistration<T>> {
            self.capabilities
                .iter()
                .find(|registration| registration.capability == id)
        }

        pub(super) fn intrinsic(&self, id: IntrinsicId) -> Option<&IntrinsicImplementation<T>> {
            self.capabilities
                .iter()
                .flat_map(|registration| &registration.implementations)
                .find(|implementation| implementation.id == id)
        }
    }

    /// The floor every backend profile must clear. Below it the device is
    /// `TargetError::UnsupportedDevice`; nothing in the compiler models a
    /// target without these facts.
    fn check_floor(
        parts: &DiscoveredTarget<impl seismic_target::TargetFamily>,
    ) -> Result<(), TargetError> {
        let unsupported = |reason: String| TargetError::UnsupportedDevice(reason);
        let limits = &parts.limits;
        if limits.max_workgroup_threads == 0 {
            return Err(unsupported("no workgroup thread can be launched".into()));
        }
        for (axis, size) in limits.max_workgroup_size.iter().enumerate() {
            if *size == 0 {
                return Err(unsupported(format!(
                    "workgroup axis {axis} admits no thread"
                )));
            }
        }
        for (axis, size) in limits.max_grid.iter().enumerate() {
            if *size == 0 {
                return Err(unsupported(format!("grid axis {axis} admits no workgroup")));
            }
        }
        if limits.max_bindings == 0 {
            return Err(unsupported("no buffer can be bound to a launch".into()));
        }
        if limits.max_index_bits < 32 {
            return Err(unsupported(format!(
                "indices are {} bits wide; the kernel IR addresses at least 32",
                limits.max_index_bits
            )));
        }
        if !limits.max_allocation_alignment.is_power_of_two() {
            return Err(unsupported(format!(
                "allocation alignment {} is not a power of two",
                limits.max_allocation_alignment
            )));
        }
        if limits.max_allocation_bytes < limits.max_allocation_alignment {
            return Err(unsupported("no allocation of one aligned unit fits".into()));
        }
        if let Some(width) = limits.subgroup_width {
            if width == 0 {
                return Err(unsupported("a subgroup of zero lanes".into()));
            }
        }
        for dtype in [DType::F32, DType::I32, DType::U32, DType::Bool] {
            if !parts.dtypes.scalars.contains(&dtype) {
                return Err(unsupported(format!(
                    "scalar `{}` is not supported",
                    dtype.name()
                )));
            }
        }
        for dtype in &parts.dtypes.scalars {
            let dense = registry::dense(*dtype);
            if !parts.dtypes.representations.contains(&dense) {
                return Err(unsupported(format!(
                    "scalar `{}` is supported but its dense representation is not",
                    dtype.name()
                )));
            }
        }
        for dtype in &parts.dtypes.atomics {
            if !parts.dtypes.scalars.contains(dtype) {
                return Err(unsupported(format!(
                    "atomic `{}` is supported without the scalar",
                    dtype.name()
                )));
            }
        }
        for entry in &parts.vectors.entries {
            if entry.lanes == 0 || !entry.lanes.is_power_of_two() {
                return Err(unsupported(format!(
                    "vector width {} for `{}` is not a nonzero power of two",
                    entry.lanes,
                    entry.dtype.name()
                )));
            }
            if !parts.dtypes.scalars.contains(&entry.dtype) {
                return Err(unsupported(format!(
                    "vector `{}`x{} is supported without its scalar type",
                    entry.dtype.name(),
                    entry.lanes
                )));
            }
            if entry.operations.is_empty() {
                return Err(unsupported(format!(
                    "vector `{}`x{} has no supported operations",
                    entry.dtype.name(),
                    entry.lanes
                )));
            }
        }
        Ok(())
    }

    /// Profile assembly: the parts clear the floor, and the advertised
    /// capability and intrinsic sets are derived from the sealed registry's
    /// predicates over the facts. A predicate that names a signature its
    /// registration does not implement is a registry bug (§13.3.1).
    pub(super) fn assemble<T: seismic_target::TargetFamily>(
        parts: DiscoveredTarget<T>,
        registry: &CompilerRegistry<T>,
    ) -> Result<DeviceDescription<T>, TargetError> {
        if parts.identity.backend != T::NAME {
            panic!(
                "DeviceDescription<{:?}>: the backend gathered an identity for {:?}",
                T::NAME,
                parts.identity.backend
            );
        }
        check_floor(&parts)?;
        let mut capabilities = BTreeSet::new();
        let mut intrinsics = BTreeSet::new();
        for registration in &registry.inner.capabilities {
            let supported = (registration.supported)(&parts.facts, &parts.limits, &parts.dtypes);
            for intrinsic in &supported {
                if !registration
                    .implementations
                    .iter()
                    .any(|implementation| implementation.id == *intrinsic)
                {
                    panic!(
                        "CompilerRegistry<{:?}>: the predicate of capability {:?} advertises {intrinsic:?}, which the registration does not implement",
                        T::NAME,
                        registration.capability
                    );
                }
            }
            if !supported.is_empty() {
                capabilities.insert(registration.capability);
                intrinsics.extend(supported);
            }
        }
        let addressable_resources = (registry.inner.addressable_resources)(&parts.facts);
        let mut names = BTreeSet::new();
        for class in &addressable_resources {
            assert!(
                !class.stable_name.is_empty(),
                "resource class stable name is empty"
            );
            assert!(
                !class.unit_name.is_empty(),
                "resource class unit name is empty"
            );
            assert!(
                names.insert(class.stable_name),
                "resource class stable name is duplicated"
            );
            assert!(
                class.capacity_units > 0,
                "resource class capacity must be nonzero"
            );
            assert!(
                class.alignment_units.is_power_of_two(),
                "resource class alignment must be a nonzero power of two"
            );
        }
        let mut compatibility = parts.identity;
        let mut fingerprint = Sha256::new();
        fingerprint.update(b"seismic-target-compatibility-v2");
        fingerprint.update(compatibility.fingerprint);
        for class in &addressable_resources {
            fingerprint.update((class.stable_name.len() as u64).to_le_bytes());
            fingerprint.update(class.stable_name.as_bytes());
            fingerprint.update((class.unit_name.len() as u64).to_le_bytes());
            fingerprint.update(class.unit_name.as_bytes());
            fingerprint.update([match class.ownership {
                ResourceOwnershipScope::Participant => 0,
                ResourceOwnershipScope::Subgroup => 1,
                ResourceOwnershipScope::Workgroup => 2,
                ResourceOwnershipScope::CooperativeGrid => 3,
            }]);
            fingerprint.update(class.capacity_units.to_le_bytes());
            fingerprint.update(class.alignment_units.to_le_bytes());
            fingerprint.update([match class.realization {
                AddressableResourceRealization::Native => 0,
            }]);
        }
        compatibility.fingerprint = fingerprint.finalize().into();
        let identity = DeviceDescriptionIdentity {
            backend: T::NAME,
            fingerprint: compatibility.fingerprint,
        };
        let mut numerical = Sha256::new();
        numerical.update(b"seismic-numerical-environment-v1");
        numerical.update(T::NAME.as_str().as_bytes());
        numerical.update(compatibility.backend_revision.as_bytes());
        numerical.update(compatibility.toolchain.as_bytes());
        numerical.update([
            parts.numerics.contraction_available as u8,
            parts.numerics.flush_to_zero_available as u8,
            parts.numerics.denormals_preserved as u8,
        ]);
        for operation in &parts.numerics.approximate_transcendentals {
            numerical.update((operation.len() as u64).to_le_bytes());
            numerical.update(operation.as_bytes());
        }
        let numerical_environment = NumericalEnvironmentIdentity {
            backend: T::NAME,
            fingerprint: numerical.finalize().into(),
        };
        DeviceDescription::new(seismic_target::DeviceDescriptionParts {
            identity,
            compatibility,
            numerical_environment,
            limits: parts.limits,
            dtypes: parts.dtypes,
            vectors: parts.vectors,
            numerics: parts.numerics,
            capabilities,
            intrinsics,
            facts: parts.facts,
            kernel_abi: parts.kernel_abi,
            local_realization: parts.local_realization,
            addressable_resources,
        })
        .map_err(|error| TargetError::UnsupportedDevice(error.to_string()))
    }

    pub(super) fn assemble_execution<T: seismic_target::TargetFamily>(
        required: BTreeSet<ServiceClassId>,
        mut parts: ExecutionProfileParts<T>,
    ) -> Result<ExecutionProfile<T>, TargetError> {
        let device = parts.device.clone();
        macro_rules! require_profile {
            ($condition:expr, $($message:tt)*) => {
                if !$condition {
                    return Err(TargetError::InvalidExecutionProfile(format!($($message)*)));
                }
            };
        }
        let core_required: BTreeSet<_> = <CoreService as AnalyticalService>::ALL
            .iter()
            .copied()
            .map(|service| ServiceClassId::new(service.stable_name()))
            .collect();
        require_profile!(
            core_required.is_subset(&required),
            "required-service set omits a core command service"
        );
        require_profile!(
            !parts.probe_suite_revision.is_empty()
                && !parts.composition_qualification.suite_revision.is_empty(),
            "profile probe or composition suite revision is empty"
        );
        parts.services.sort_by_key(|definition| definition.class);
        for definition in &mut parts.services {
            if let FactProvenance::Measured { batches } = &mut definition.provenance {
                batches.sort_by_key(seismic_estimator::MeasurementBatch::identity);
            }
        }
        parts
            .composition_qualification
            .cases
            .sort_by_key(|case| case.stable_name);
        for case in &mut parts.composition_qualification.cases {
            case.demands.sort_by_key(|demand| {
                (
                    demand.class,
                    match demand.mode {
                        DemandMode::DependencyLatency => 0u8,
                        DemandMode::SaturatedCapacity => 1u8,
                    },
                    demand.units,
                )
            });
        }

        let mut provided = BTreeSet::new();
        for definition in &parts.services {
            if definition.class.stable_name().is_empty() {
                return Err(TargetError::InvalidExecutionProfile(format!(
                    "ExecutionProfile<{:?}> contains an empty service identity",
                    T::NAME
                )));
            }
            if !provided.insert(definition.class) {
                return Err(TargetError::InvalidExecutionProfile(format!(
                    "ExecutionProfile<{:?}> supplies service `{}` twice",
                    T::NAME,
                    definition.class.stable_name()
                )));
            }
            require_profile!(
                !definition.correlation.stable_name().is_empty(),
                "ExecutionProfile<{:?}> service `{}` has an empty correlation identity",
                T::NAME,
                definition.class.stable_name()
            );
            require_profile!(
                definition.qualification.minimum_units != 0
                    && definition.qualification.minimum_units
                        <= definition.qualification.maximum_units
                    && definition.qualification.maximum_concurrent_uses != 0,
                "ExecutionProfile<{:?}> service `{}` has an empty qualification domain",
                T::NAME,
                definition.class.stable_name()
            );
            if definition.topology.resources == 0 || definition.topology.max_concurrency == 0 {
                return Err(TargetError::InvalidExecutionProfile(format!(
                    "ExecutionProfile<{:?}> service `{}` has an empty resource topology",
                    T::NAME,
                    definition.class.stable_name()
                )));
            }
            if definition.saturated_capacity.regimes.is_empty() {
                return Err(TargetError::InvalidExecutionProfile(format!(
                    "ExecutionProfile<{:?}> service `{}` has no capacity regime",
                    T::NAME,
                    definition.class.stable_name()
                )));
            }
            let mut previous = None;
            for (index, regime) in definition.saturated_capacity.regimes.iter().enumerate() {
                match (previous, regime.max_units) {
                    (_, None) if index + 1 == definition.saturated_capacity.regimes.len() => {}
                    (_, None) => {
                        return Err(TargetError::InvalidExecutionProfile(format!(
                            "ExecutionProfile<{:?}> service `{}` has a non-final unbounded regime",
                            T::NAME,
                            definition.class.stable_name()
                        )))
                    }
                    (Some(previous), Some(current)) if current <= previous => {
                        return Err(TargetError::InvalidExecutionProfile(format!(
                            "ExecutionProfile<{:?}> service `{}` has unordered regimes",
                            T::NAME,
                            definition.class.stable_name()
                        )))
                    }
                    (_, Some(current)) => previous = Some(current),
                }
            }
            if definition
                .saturated_capacity
                .regimes
                .last()
                .is_some_and(|regime| regime.max_units.is_some())
            {
                return Err(TargetError::InvalidExecutionProfile(format!(
                    "ExecutionProfile<{:?}> service `{}` lacks a final unbounded regime",
                    T::NAME,
                    definition.class.stable_name()
                )));
            }
            let intervals = std::iter::once(definition.dependency_latency)
                .chain(std::iter::once(definition.saturated_capacity.setup))
                .chain(
                    definition
                        .saturated_capacity
                        .regimes
                        .iter()
                        .map(|regime| regime.per_unit),
                );
            for interval in intervals {
                if interval.denominator == 0 || interval.lower_numerator > interval.upper_numerator
                {
                    return Err(TargetError::InvalidExecutionProfile(format!(
                        "ExecutionProfile<{:?}> service `{}` has an invalid duration interval",
                        T::NAME,
                        definition.class.stable_name()
                    )));
                }
                require_profile!(
                    interval.denominator <= u64::MAX / 10_000
                        && interval.lower_numerator <= u64::MAX / 10_000
                        && interval.upper_numerator <= u64::MAX / 10_500,
                    "ExecutionProfile<{:?}> service `{}` interval exceeds total evaluation arithmetic bounds",
                    T::NAME,
                    definition.class.stable_name()
                );
                if interval.upper_numerator != 0 {
                    let allowed_percent = match definition.accuracy {
                        ServiceAccuracyClass::Compute => 1u128,
                        ServiceAccuracyClass::MemoryOrTransfer => 2u128,
                    };
                    let width = u128::from(interval.upper_numerator - interval.lower_numerator);
                    let midpoint_twice =
                        u128::from(interval.upper_numerator) + u128::from(interval.lower_numerator);
                    require_profile!(
                        width * 100 <= midpoint_twice * allowed_percent,
                        "ExecutionProfile<{:?}> service `{}` interval {}/{}..{}/{} ns exceeds its {}% acquisition interval budget",
                        T::NAME,
                        definition.class.stable_name(),
                        interval.lower_numerator,
                        interval.denominator,
                        interval.upper_numerator,
                        interval.denominator,
                        allowed_percent
                    );
                }
            }
            if let FactProvenance::Measured { batches } = &definition.provenance {
                require_profile!(
                    !batches.is_empty(),
                    "ExecutionProfile<{:?}> service `{}` has no measurement batches",
                    T::NAME,
                    definition.class.stable_name()
                );
                for batch in batches {
                    require_profile!(
                        !batch.probe.is_empty()
                            && !batch.method.is_empty()
                            && batch.series.is_valid(),
                        "ExecutionProfile<{:?}> service `{}` has an incomplete measurement batch",
                        T::NAME,
                        definition.class.stable_name()
                    );
                    require_profile!(
                        batch.timer_resolution_ns.denominator != 0
                            && batch.timer_resolution_ns.lower_numerator
                                <= batch.timer_resolution_ns.upper_numerator,
                        "ExecutionProfile<{:?}> service `{}` has invalid timer resolution evidence",
                        T::NAME,
                        definition.class.stable_name()
                    );
                }
            }
        }
        require_profile!(
            required == provided,
            "ExecutionProfile<{:?}> required/provided execution service sets differ",
            T::NAME,
        );

        #[derive(Clone, Copy)]
        struct WideInterval {
            lower: u128,
            upper: u128,
            denominator: u128,
        }

        fn gcd(mut left: u128, mut right: u128) -> u128 {
            while right != 0 {
                let remainder = left % right;
                left = right;
                right = remainder;
            }
            left
        }

        fn scale_interval(interval: DurationInterval, units: u64) -> Option<WideInterval> {
            Some(WideInterval {
                lower: u128::from(interval.lower_numerator).checked_mul(u128::from(units))?,
                upper: u128::from(interval.upper_numerator).checked_mul(u128::from(units))?,
                denominator: u128::from(interval.denominator),
            })
        }

        fn add_intervals(left: WideInterval, right: WideInterval) -> Option<WideInterval> {
            let common = gcd(left.denominator, right.denominator);
            let left_scale = right.denominator / common;
            let right_scale = left.denominator / common;
            Some(WideInterval {
                lower: left.lower.checked_mul(left_scale).and_then(|value| {
                    right
                        .lower
                        .checked_mul(right_scale)
                        .and_then(|right| value.checked_add(right))
                })?,
                upper: left.upper.checked_mul(left_scale).and_then(|value| {
                    right
                        .upper
                        .checked_mul(right_scale)
                        .and_then(|right| value.checked_add(right))
                })?,
                denominator: left.denominator.checked_mul(left_scale)?,
            })
        }

        require_profile!(
            !parts.composition_qualification.cases.is_empty(),
            "ExecutionProfile<{:?}> has no held-out composition qualification cases",
            T::NAME
        );
        let mut composition_names = BTreeSet::new();
        let mut maximum_relative_error_basis_points = 0u16;
        for case in &parts.composition_qualification.cases {
            require_profile!(
                !case.stable_name.is_empty() && composition_names.insert(case.stable_name),
                "ExecutionProfile<{:?}> has an empty or duplicated composition case `{}`",
                T::NAME,
                case.stable_name
            );
            require_profile!(
                !case.demands.is_empty(),
                "ExecutionProfile<{:?}> composition case `{}` has no demands",
                T::NAME,
                case.stable_name
            );
            require_profile!(
                case.observed_ns.denominator != 0
                    && case.observed_ns.lower_numerator <= case.observed_ns.upper_numerator,
                "ExecutionProfile<{:?}> composition case `{}` has an invalid observation interval",
                T::NAME,
                case.stable_name
            );

            let mut predicted = WideInterval {
                lower: 0,
                upper: 0,
                denominator: 1,
            };
            for demand in &case.demands {
                require_profile!(
                    demand.units != 0,
                    "ExecutionProfile<{:?}> composition case `{}` has zero service demand",
                    T::NAME,
                    case.stable_name
                );
                let service = parts
                    .services
                    .iter()
                    .find(|service| service.class == demand.class)
                    .ok_or_else(|| {
                        TargetError::InvalidExecutionProfile(format!(
                        "ExecutionProfile<{:?}> composition case `{}` names absent service `{}`",
                        T::NAME, case.stable_name, demand.class.stable_name()
                    ))
                    })?;
                let contribution = match demand.mode {
                    DemandMode::DependencyLatency => {
                        scale_interval(service.dependency_latency, demand.units)
                    }
                    DemandMode::SaturatedCapacity => {
                        let capacity = u64::from(service.topology.resources)
                            .checked_mul(u64::from(service.topology.max_concurrency))
                            .ok_or_else(|| {
                                TargetError::InvalidExecutionProfile(
                                    "composition service topology overflows u64".into(),
                                )
                            })?;
                        let waves = demand.units.div_ceil(capacity);
                        let regime = service
                            .saturated_capacity
                            .regimes
                            .iter()
                            .find(|regime| {
                                regime
                                    .max_units
                                    .is_none_or(|maximum| demand.units <= maximum)
                            })
                            .ok_or_else(|| {
                                TargetError::InvalidExecutionProfile(
                                    "validated service curve has no applicable final regime".into(),
                                )
                            })?;
                        add_intervals(
                            scale_interval(service.saturated_capacity.setup, 1).ok_or_else(
                                || {
                                    TargetError::InvalidExecutionProfile(
                                        "composition prediction arithmetic overflow".into(),
                                    )
                                },
                            )?,
                            scale_interval(regime.per_unit, waves).ok_or_else(|| {
                                TargetError::InvalidExecutionProfile(
                                    "composition prediction arithmetic overflow".into(),
                                )
                            })?,
                        )
                    }
                }
                .ok_or_else(|| {
                    TargetError::InvalidExecutionProfile(
                        "composition prediction arithmetic overflow".into(),
                    )
                })?;
                require_profile!(
                    demand.units >= service.qualification.minimum_units
                        && demand.units <= service.qualification.maximum_units,
                    "ExecutionProfile<{:?}> composition case `{}` is outside service `{}` qualification",
                    T::NAME,
                    case.stable_name,
                    demand.class.stable_name()
                );
                predicted = add_intervals(predicted, contribution).ok_or_else(|| {
                    TargetError::InvalidExecutionProfile(
                        "composition prediction arithmetic overflow".into(),
                    )
                })?;
            }

            let predicted_midpoint =
                predicted
                    .lower
                    .checked_add(predicted.upper)
                    .ok_or_else(|| {
                        TargetError::InvalidExecutionProfile(
                            "composition prediction midpoint overflows u128".into(),
                        )
                    })?;
            let observed_midpoint = u128::from(case.observed_ns.lower_numerator)
                .checked_add(u128::from(case.observed_ns.upper_numerator))
                .ok_or_else(|| {
                    TargetError::InvalidExecutionProfile(
                        "composition observation midpoint overflows u128".into(),
                    )
                })?;
            require_profile!(
                observed_midpoint != 0,
                "composition observation midpoint is zero"
            );
            let predicted_scaled = predicted_midpoint
                .checked_mul(u128::from(case.observed_ns.denominator))
                .ok_or_else(|| {
                    TargetError::InvalidExecutionProfile(
                        "composition prediction comparison overflows u128".into(),
                    )
                })?;
            let observed_scaled = observed_midpoint
                .checked_mul(predicted.denominator)
                .ok_or_else(|| {
                    TargetError::InvalidExecutionProfile(
                        "composition observation comparison overflows u128".into(),
                    )
                })?;
            let error = predicted_scaled
                .abs_diff(observed_scaled)
                .checked_mul(10_000)
                .ok_or_else(|| {
                    TargetError::InvalidExecutionProfile(
                        "composition relative error overflows u128".into(),
                    )
                })?;
            let relative_error_basis_points = error.div_ceil(observed_scaled);
            require_profile!(
                relative_error_basis_points <= 500,
                "ExecutionProfile<{:?}> held-out composition `{}` predicted {}/{}..{}/{} ns versus observed {}/{}..{}/{} ns ({} bp) exceeds the 5% qualification budget",
                T::NAME,
                case.stable_name,
                predicted.lower,
                predicted.denominator,
                predicted.upper,
                predicted.denominator,
                case.observed_ns.lower_numerator,
                case.observed_ns.denominator,
                case.observed_ns.upper_numerator,
                case.observed_ns.denominator,
                relative_error_basis_points,
            );
            maximum_relative_error_basis_points =
                maximum_relative_error_basis_points.max(relative_error_basis_points as u16);
        }
        fn tagged(execution: &mut Sha256, tag: &'static [u8]) {
            execution.update((tag.len() as u64).to_le_bytes());
            execution.update(tag);
        }
        fn text(execution: &mut Sha256, tag: &'static [u8], value: &str) {
            tagged(execution, tag);
            execution.update((value.len() as u64).to_le_bytes());
            execution.update(value.as_bytes());
        }

        let mut execution = Sha256::new();
        tagged(&mut execution, b"seismic-execution-profile-v3");
        execution.update(device.identity().fingerprint);
        text(
            &mut execution,
            b"probe-suite-revision",
            parts.probe_suite_revision,
        );
        text(
            &mut execution,
            b"composition-suite-revision",
            parts.composition_qualification.suite_revision,
        );
        execution.update(maximum_relative_error_basis_points.to_le_bytes());
        execution.update(parts.composition_qualification.observations_digest);
        execution.update((parts.composition_qualification.cases.len() as u64).to_le_bytes());
        for case in &parts.composition_qualification.cases {
            text(&mut execution, b"composition-case", case.stable_name);
            execution.update((case.demands.len() as u64).to_le_bytes());
            for demand in &case.demands {
                text(
                    &mut execution,
                    b"composition-demand-service",
                    demand.class.stable_name(),
                );
                execution.update(demand.units.to_le_bytes());
                execution.update([match demand.mode {
                    DemandMode::DependencyLatency => 0,
                    DemandMode::SaturatedCapacity => 1,
                }]);
            }
            execution.update(case.observed_ns.lower_numerator.to_le_bytes());
            execution.update(case.observed_ns.upper_numerator.to_le_bytes());
            execution.update(case.observed_ns.denominator.to_le_bytes());
        }
        for service in &parts.services {
            text(
                &mut execution,
                b"service-class",
                service.class.stable_name(),
            );
            text(
                &mut execution,
                b"service-correlation",
                service.correlation.stable_name(),
            );
            execution.update(service.qualification.minimum_units.to_le_bytes());
            execution.update(service.qualification.maximum_units.to_le_bytes());
            execution.update(service.qualification.maximum_concurrent_uses.to_le_bytes());
            execution.update([match service.accuracy {
                ServiceAccuracyClass::Compute => 0,
                ServiceAccuracyClass::MemoryOrTransfer => 1,
            }]);
            execution.update(service.topology.resources.to_le_bytes());
            execution.update(service.topology.max_concurrency.to_le_bytes());
            let intervals = std::iter::once(service.dependency_latency)
                .chain(std::iter::once(service.saturated_capacity.setup))
                .chain(
                    service
                        .saturated_capacity
                        .regimes
                        .iter()
                        .map(|regime| regime.per_unit),
                );
            for interval in intervals {
                execution.update(interval.lower_numerator.to_le_bytes());
                execution.update(interval.upper_numerator.to_le_bytes());
                execution.update(interval.denominator.to_le_bytes());
            }
            for regime in &service.saturated_capacity.regimes {
                match regime.max_units {
                    Some(maximum) => {
                        execution.update([1]);
                        execution.update(maximum.to_le_bytes());
                    }
                    None => execution.update([0]),
                }
            }
            service.provenance.update_identity(&mut execution);
        }
        let identity = ExecutionProfileIdentity {
            device: device.identity().clone(),
            probe_suite_revision: parts.probe_suite_revision,
            fingerprint: execution.finalize().into(),
        };
        let composition_qualification = CompositionQualification {
            suite_revision: parts.composition_qualification.suite_revision,
            cases: parts.composition_qualification.cases,
            maximum_relative_error_basis_points,
            observations_digest: parts.composition_qualification.observations_digest,
        };
        Ok(ExecutionProfile {
            identity,
            services: parts.services,
            acquisition: parts.acquisition,
            composition_qualification,
            _target: std::marker::PhantomData,
        })
    }

    pub(super) fn bind_constants<T: seismic_target::TargetFamily>(
        profile: &DeviceDescription<T>,
        arena: &mut ExprArena,
    ) -> TargetConstants {
        fn nat(
            arena: &mut ExprArena,
            bindings: &mut Vec<(SymbolId, SymbolValue)>,
            value: u64,
        ) -> TargetConstantId {
            let (id, symbol) = arena.target_constant(SymbolSort::Nat);
            bindings.push((symbol, SymbolValue::Nat((value).into())));
            id
        }
        fn optional(
            arena: &mut ExprArena,
            bindings: &mut Vec<(SymbolId, SymbolValue)>,
            value: Option<u64>,
        ) -> Option<TargetConstantId> {
            value.map(|value| nat(arena, bindings, value))
        }

        let limits = profile.limits();
        let mut bindings = Vec::new();
        let max_workgroup_size = limits
            .max_workgroup_size
            .map(|value| nat(arena, &mut bindings, value));
        let max_workgroup_threads = nat(arena, &mut bindings, limits.max_workgroup_threads);
        let max_workgroup_bytes = nat(arena, &mut bindings, limits.max_workgroup_bytes);
        let participant_local_bytes =
            optional(arena, &mut bindings, limits.participant_local_bytes);
        let max_grid = limits
            .max_grid
            .map(|value| nat(arena, &mut bindings, value));
        let max_bindings = nat(arena, &mut bindings, u64::from(limits.max_bindings));
        let max_argument_bytes = nat(arena, &mut bindings, limits.max_argument_bytes);
        let max_allocation_bytes = nat(arena, &mut bindings, limits.max_allocation_bytes);
        let max_allocation_alignment = nat(arena, &mut bindings, limits.max_allocation_alignment);
        let max_index_bits = nat(arena, &mut bindings, u64::from(limits.max_index_bits));
        let subgroup_width = optional(arena, &mut bindings, limits.subgroup_width.map(u64::from));
        let addressable_resource_capacity = profile
            .addressable_resources()
            .iter()
            .map(|class| nat(arena, &mut bindings, class.capacity_units))
            .collect();
        TargetConstants {
            max_workgroup_size,
            max_workgroup_threads,
            max_workgroup_bytes,
            participant_local_bytes,
            max_grid,
            max_bindings,
            max_argument_bytes,
            max_allocation_bytes,
            max_allocation_alignment,
            max_index_bits,
            subgroup_width,
            addressable_resource_capacity,
            bindings,
        }
    }
}
