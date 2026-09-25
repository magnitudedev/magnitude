use crate::TargetDescriptionError;
use seismic_ir::kernel::Kernel;
use seismic_ir::physical_target::{
    AddressableResourceClass, AddressableResourceEmissionLayout, BindingEmissionLayout,
    DataTypeSupport, KernelAbiLayout, KernelAbiModel, PhysicalDialect, KernelEmissionLayout,
    KernelWordLayout, LocalEmissionLayout, LocalRealizationPolicy, NumericalEnvironment,
    RepresentationGeometry, ResourceClassId, TargetLimits, VectorSupport,
};
use seismic_lang::ids::{CapabilityId, IntrinsicId};
use seismic_lang::registry::BackendName;
use std::collections::BTreeSet;
use std::fmt;

/// The type family shared by executable IR and immutable native descriptions.
///
/// This is deliberately smaller than a backend.  It has no native context,
/// compiler, executor, cost model, factory catalog, or capability lowering.
pub trait TargetFamily: PhysicalDialect {
    type KernelAbi: KernelAbiModel<Self>;
    type NativeNumericalMode: Clone
        + fmt::Debug
        + PartialEq
        + Eq
        + std::hash::Hash
        + Send
        + Sync
        + 'static;
    /// Backend-specific immutable facts reflected from one native artifact.
    /// These facts may drive legality or diagnostics without exposing the
    /// executable handle.
    type NativeProperties: Clone
        + fmt::Debug
        + PartialEq
        + Eq
        + std::hash::Hash
        + Send
        + Sync
        + 'static;
}

/// Stable compatibility identity for native code reuse.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompatibilityIdentity {
    pub backend: BackendName,
    pub backend_revision: &'static str,
    pub hardware: String,
    pub driver: String,
    pub toolchain: String,
    pub fingerprint: [u8; 32],
}

/// Stable identity of the device-wide numerical environment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NumericalEnvironmentIdentity {
    pub backend: BackendName,
    pub fingerprint: [u8; 32],
}

/// Stable identity of one concrete artifact's native numerical mode.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeNumericalModeIdentity {
    pub fingerprint: [u8; 32],
}

/// Identity of one immutable device description.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceDescriptionIdentity {
    pub backend: BackendName,
    pub fingerprint: [u8; 32],
}

/// Parts supplied after target discovery and capability-registry validation.
///
/// Registry validation remains compiler integration policy.  This crate owns
/// the closed immutable result and verifies its structural invariants.
pub struct DeviceDescriptionParts<T: TargetFamily> {
    pub identity: DeviceDescriptionIdentity,
    pub compatibility: CompatibilityIdentity,
    pub numerical_environment: NumericalEnvironmentIdentity,
    pub limits: TargetLimits,
    pub dtypes: DataTypeSupport,
    pub vectors: VectorSupport,
    pub numerics: NumericalEnvironment,
    pub capabilities: BTreeSet<CapabilityId>,
    pub intrinsics: BTreeSet<IntrinsicId>,
    pub facts: T::Facts,
    pub kernel_abi: T::KernelAbi,
    pub local_realization: LocalRealizationPolicy,
    pub addressable_resources: Vec<AddressableResourceClass>,
}

/// Complete immutable target truth available before native compilation.
pub struct DeviceDescription<T: TargetFamily> {
    identity: DeviceDescriptionIdentity,
    compatibility: CompatibilityIdentity,
    numerical_environment: NumericalEnvironmentIdentity,
    limits: TargetLimits,
    dtypes: DataTypeSupport,
    vectors: VectorSupport,
    numerics: NumericalEnvironment,
    capabilities: BTreeSet<CapabilityId>,
    intrinsics: BTreeSet<IntrinsicId>,
    facts: T::Facts,
    kernel_abi: T::KernelAbi,
    local_realization: LocalRealizationPolicy,
    addressable_resources: Vec<AddressableResourceClass>,
}

impl<T: TargetFamily> fmt::Debug for DeviceDescription<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceDescription")
            .field("identity", &self.identity)
            .field("compatibility", &self.compatibility)
            .field("numerical_environment", &self.numerical_environment)
            .field("limits", &self.limits)
            .field("dtypes", &self.dtypes)
            .field("vectors", &self.vectors)
            .field("numerics", &self.numerics)
            .field("capabilities", &self.capabilities)
            .field("intrinsics", &self.intrinsics)
            .field("facts", &self.facts)
            .field("kernel_abi", &self.kernel_abi)
            .field("local_realization", &self.local_realization)
            .field("addressable_resources", &self.addressable_resources)
            .finish()
    }
}

impl<T: TargetFamily> DeviceDescription<T> {
    pub fn new(parts: DeviceDescriptionParts<T>) -> Result<Self, TargetDescriptionError> {
        if parts.identity.backend != T::NAME
            || parts.compatibility.backend != T::NAME
            || parts.numerical_environment.backend != T::NAME
        {
            return Err(TargetDescriptionError::BackendIdentityMismatch);
        }
        validate_limits(&parts.limits)?;
        let mut resource_names = BTreeSet::new();
        for resource in &parts.addressable_resources {
            if resource.alignment_units == 0 || !resource.alignment_units.is_power_of_two() {
                return Err(TargetDescriptionError::InvalidAlignment(
                    "addressable resource",
                ));
            }
            if !resource_names.insert(resource.stable_name) {
                return Err(TargetDescriptionError::DuplicateResourceName(
                    resource.stable_name,
                ));
            }
        }
        Ok(Self {
            identity: parts.identity,
            compatibility: parts.compatibility,
            numerical_environment: parts.numerical_environment,
            limits: parts.limits,
            dtypes: parts.dtypes,
            vectors: parts.vectors,
            numerics: parts.numerics,
            capabilities: parts.capabilities,
            intrinsics: parts.intrinsics,
            facts: parts.facts,
            kernel_abi: parts.kernel_abi,
            local_realization: parts.local_realization,
            addressable_resources: parts.addressable_resources,
        })
    }

    pub fn identity(&self) -> &DeviceDescriptionIdentity {
        &self.identity
    }
    pub fn compatibility_identity(&self) -> &CompatibilityIdentity {
        &self.compatibility
    }
    pub fn numerical_environment_identity(&self) -> &NumericalEnvironmentIdentity {
        &self.numerical_environment
    }
    pub fn limits(&self) -> &TargetLimits {
        &self.limits
    }
    pub fn dtypes(&self) -> &DataTypeSupport {
        &self.dtypes
    }
    pub fn vectors(&self) -> &VectorSupport {
        &self.vectors
    }
    pub fn numerics(&self) -> &NumericalEnvironment {
        &self.numerics
    }
    pub fn facts(&self) -> &T::Facts {
        &self.facts
    }
    pub fn kernel_abi(&self) -> &T::KernelAbi {
        &self.kernel_abi
    }
    pub fn local_realization(&self) -> LocalRealizationPolicy {
        self.local_realization
    }
    pub fn supports_capability(&self, id: CapabilityId) -> bool {
        self.capabilities.contains(&id)
    }
    pub fn supports_intrinsic(&self, id: IntrinsicId) -> bool {
        self.intrinsics.contains(&id)
    }
    pub fn capabilities(&self) -> &BTreeSet<CapabilityId> {
        &self.capabilities
    }
    pub fn intrinsics(&self) -> &BTreeSet<IntrinsicId> {
        &self.intrinsics
    }
    pub fn addressable_resources(&self) -> &[AddressableResourceClass] {
        &self.addressable_resources
    }
    pub fn addressable_resource_class(&self, stable_name: &str) -> Option<ResourceClassId> {
        ResourceClassId::lookup(&self.addressable_resources, stable_name)
    }
    pub fn addressable_resource(&self, id: ResourceClassId) -> &AddressableResourceClass {
        self.addressable_resources
            .get(id.ordinal() as usize)
            .expect("resource class id belongs to another device description")
    }

    pub fn kernel_abi_layout(&self, kernel: &Kernel<T>) -> KernelAbiLayout {
        let layout = self.kernel_abi.layout(kernel);
        let mut roles = BTreeSet::new();
        for allocation in &layout.allocations {
            assert!(
                roles.insert(allocation.role),
                "kernel ABI repeats an allocation role"
            );
            assert!(
                allocation.alignment.is_power_of_two(),
                "kernel ABI allocation alignment must be a nonzero power of two"
            );
            assert!(
                allocation.alignment <= self.limits.max_allocation_alignment,
                "kernel ABI allocation alignment exceeds the target limit"
            );
            assert!(
                allocation.bytes <= self.limits.max_allocation_bytes,
                "kernel ABI allocation size exceeds the target limit"
            );
        }
        assert!(
            layout.footprint.alignment.is_power_of_two(),
            "kernel ABI footprint alignment must be a nonzero power of two"
        );
        assert!(
            layout.footprint.alignment <= self.limits.max_allocation_alignment,
            "kernel ABI footprint alignment exceeds the target limit"
        );
        assert!(
            layout.footprint.bytes <= self.limits.max_argument_bytes,
            "kernel ABI footprint exceeds the target argument-table limit"
        );
        layout
    }

    /// Derives the only native-emission view from the authoritative kernel.
    pub fn kernel_emission_layout(&self, kernel: &Kernel<T>) -> KernelEmissionLayout {
        let words = KernelWordLayout::for_kernel(kernel);
        let bindings = kernel
            .interface()
            .bindings
            .iter()
            .zip(&words.bindings)
            .map(|(binding, word_layout)| BindingEmissionLayout {
                slot: binding.slot,
                access: binding.access,
                words: *word_layout,
                geometry: RepresentationGeometry::of(binding.view.representation()),
            })
            .collect();
        let locals = kernel
            .locals()
            .iter()
            .zip(&words.locals)
            .map(|(local, word_layout)| LocalEmissionLayout {
                kind: local.kind,
                alignment: local.alignment,
                words: *word_layout,
                geometry: RepresentationGeometry::of(local.representation),
                realization: self.local_realization.for_kind(local.kind),
            })
            .collect();
        let addressable_resources = kernel
            .addressable_resources()
            .iter()
            .zip(&words.addressable_resources)
            .map(|(lease, word_layout)| AddressableResourceEmissionLayout {
                handle: lease.handle,
                class_id: lease.class_id,
                class: lease.class.clone(),
                words: *word_layout,
                alignment_units: lease.alignment_units,
                lifetime: lease.lifetime,
            })
            .collect();
        KernelEmissionLayout {
            words,
            bindings,
            locals,
            addressable_resources,
            scalar_args: kernel
                .interface()
                .scalar_args
                .iter()
                .map(|(_, dtype)| *dtype)
                .collect(),
            result_types: kernel
                .interface()
                .result_slots
                .iter()
                .map(|slot| slot.kind())
                .collect(),
        }
    }
}

fn validate_limits(limits: &TargetLimits) -> Result<(), TargetDescriptionError> {
    for (name, value) in [
        ("max_workgroup_threads", limits.max_workgroup_threads),
        ("max_workgroup_bytes", limits.max_workgroup_bytes),
        ("max_bindings", u64::from(limits.max_bindings)),
        ("max_argument_bytes", limits.max_argument_bytes),
        ("max_allocation_bytes", limits.max_allocation_bytes),
        ("max_index_bits", u64::from(limits.max_index_bits)),
    ] {
        if value == 0 {
            return Err(TargetDescriptionError::EmptyLimit(name));
        }
    }
    if limits.max_workgroup_size.contains(&0) {
        return Err(TargetDescriptionError::EmptyLimit("max_workgroup_size"));
    }
    if limits.max_grid.contains(&0) {
        return Err(TargetDescriptionError::EmptyLimit("max_grid"));
    }
    if limits.max_allocation_alignment == 0 || !limits.max_allocation_alignment.is_power_of_two() {
        return Err(TargetDescriptionError::InvalidAlignment(
            "max_allocation_alignment",
        ));
    }
    if limits.subgroup_width == Some(0) {
        return Err(TargetDescriptionError::EmptyLimit("subgroup_width"));
    }
    Ok(())
}
