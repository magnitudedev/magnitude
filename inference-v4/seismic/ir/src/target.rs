//! Pure target vocabulary shared by IR, emission and prediction.

mod dialect;
pub use crate::identity::IntrinsicIdentityBuilder;
pub use dialect::PhysicalDialect;
use seismic_lang::ids::RepresentationId;
use seismic_lang::types::DType;
use std::{collections::BTreeSet, fmt};

/// Profile-local identity of one native addressable intrinsic resource.
/// The constructor is core-owned; backends obtain ids by stable-name lookup
/// on the assembled profile and cannot forge another target's class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceClassId(u32);

impl ResourceClassId {
    pub fn lookup(classes: &[AddressableResourceClass], name: &str) -> Option<Self> {
        classes
            .iter()
            .position(|class| class.stable_name == name)
            .map(|index| Self(u32::try_from(index).expect("resource class count exceeds u32")))
    }
    pub fn ordinal(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceOwnershipScope {
    Participant,
    Subgroup,
    Workgroup,
    CooperativeGrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AddressableResourceRealization {
    Native,
}

/// Immutable target fact describing one native resource class. Capacity and
/// alignment are expressed in the named native unit, never reinterpreted as
/// bytes by core or an executor.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AddressableResourceClass {
    pub stable_name: &'static str,
    pub unit_name: &'static str,
    pub ownership: ResourceOwnershipScope,
    pub capacity_units: u64,
    pub alignment_units: u64,
    pub realization: AddressableResourceRealization,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceLifetime {
    Operation,
    Segment,
    Launch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntrinsicNumericalSemantics {
    pub arithmetic: seismic_lang::registry::IntrinsicNumerics,
    pub flush_to_zero: bool,
}

/// Common hard limits every backend states (§4.1, §8.3). Values are exact
/// device facts, never conservative guesses, unless a backend documents the
/// limit as statically modelled (`participant_local_bytes`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TargetLimits {
    pub max_workgroup_size: [u64; 3],
    pub max_workgroup_threads: u64,
    pub max_grid: [u64; 3],
    pub max_workgroup_bytes: u64,
    /// Statically modelled participant-local (thread stack/private) bytes,
    /// or `None` when the backend has no such limit.
    pub participant_local_bytes: Option<u64>,
    pub max_bindings: u32,
    /// Maximum encoded metadata bytes in one kernel's argument table.
    pub max_argument_bytes: u64,
    pub max_allocation_bytes: u64,
    /// Greatest allocation alignment the device service accepts.
    pub max_allocation_alignment: u64,
    /// Widest integer index the backend can address in one dimension.
    pub max_index_bits: u32,
    pub subgroup_width: Option<u32>,
}

/// Exact encoded footprint of one kernel argument table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelAbiFootprint {
    pub bytes: u64,
    pub alignment: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KernelAbiAllocationRole {
    BufferTable,
    WordTable,
    ScalarResults,
    LaunchFrame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelAbiAllocation {
    pub role: KernelAbiAllocationRole,
    pub bytes: u64,
    pub alignment: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelAbiLayout {
    pub footprint: KernelAbiFootprint,
    pub allocations: Vec<KernelAbiAllocation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindingWordLayout {
    pub first: u32,
    pub rank: u32,
}

/// Canonical word-table slice for one launch-local tensor. The first word is
/// its byte base, followed by rank extents and rank strides.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalWordLayout {
    pub first: u32,
    pub rank: u32,
}

/// Canonical dynamic-word slots for one native addressable-resource lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AddressableResourceWordLayout {
    pub offset_units: u32,
    pub units: u32,
}

/// The one word-table schema consumed by every native emitter/executor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelWordLayout {
    pub nat_first: u32,
    pub scalar_first: u32,
    pub bindings: Vec<BindingWordLayout>,
    pub locals: Vec<LocalWordLayout>,
    pub addressable_resources: Vec<AddressableResourceWordLayout>,
    pub grid_first: u32,
    pub workgroup_first: u32,
    pub local_total_first: u32,
    pub total: u32,
}

impl KernelWordLayout {
    /// Constructs the one canonical dynamic-word schema for a closed kernel.
    /// ABI models, emitters, and executors all consume this exact layout.
    pub fn for_kernel<B: PhysicalDialect>(kernel: &crate::kernel::Kernel<B>) -> Self {
        let mut next = 0u32;
        let nat_first = next;
        next = next
            .checked_add(
                u32::try_from(kernel.interface().nat_args.len())
                    .expect("kernel nat argument count exceeds u32"),
            )
            .expect("kernel word-table ordinal space exhausted");
        let scalar_first = next;
        next = next
            .checked_add(
                u32::try_from(kernel.interface().scalar_args.len())
                    .expect("kernel scalar argument count exceeds u32"),
            )
            .expect("kernel word-table ordinal space exhausted");
        let mut bindings = Vec::with_capacity(kernel.interface().bindings.len());
        for binding in &kernel.interface().bindings {
            bindings.push(BindingWordLayout {
                first: next,
                rank: binding.rank,
            });
            next = next
                .checked_add(
                    binding
                        .rank
                        .checked_mul(2)
                        .expect("binding rank word count overflow"),
                )
                .expect("kernel word-table ordinal space exhausted");
        }
        let mut locals = Vec::with_capacity(kernel.locals().len());
        for local in kernel.locals() {
            let rank = u32::try_from(local.extents.len()).expect("kernel local rank exceeds u32");
            locals.push(LocalWordLayout { first: next, rank });
            next = next
                .checked_add(
                    rank.checked_mul(2)
                        .and_then(|words| words.checked_add(1))
                        .expect("local rank word count overflow"),
                )
                .expect("kernel word-table ordinal space exhausted");
        }
        let mut addressable_resources = Vec::with_capacity(kernel.addressable_resources().len());
        for _ in kernel.addressable_resources() {
            addressable_resources.push(AddressableResourceWordLayout {
                offset_units: next,
                units: next
                    .checked_add(1)
                    .expect("kernel word-table ordinal space exhausted"),
            });
            next = next
                .checked_add(2)
                .expect("kernel word-table ordinal space exhausted");
        }
        let grid_first = next;
        next = next
            .checked_add(3)
            .expect("kernel word-table ordinal space exhausted");
        let workgroup_first = next;
        next = next
            .checked_add(3)
            .expect("kernel word-table ordinal space exhausted");
        let local_total_first = next;
        next = next
            .checked_add(3)
            .expect("kernel word-table ordinal space exhausted");
        Self {
            nat_first,
            scalar_first,
            bindings,
            locals,
            addressable_resources,
            grid_first,
            workgroup_first,
            local_total_first,
            total: next,
        }
    }
}

/// Canonical registered physical geometry of a representation. The packed
/// descriptor and decode recipe come directly from the registry; backends do
/// not rebuild packet/plane layouts.
#[derive(Clone, Debug)]
pub struct RepresentationGeometry {
    pub info: &'static seismic_lang::registry::RepresentationInfo,
    pub decode: Option<seismic_lang::registry::DecodeRecipe>,
}

#[derive(Clone, Debug)]
pub struct DenseRepresentationGeometry {
    pub info: &'static seismic_lang::registry::RepresentationInfo,
    pub dtype: DType,
}

#[derive(Clone, Debug)]
pub struct PackedRepresentationGeometry {
    pub info: &'static seismic_lang::registry::RepresentationInfo,
    pub layout: seismic_lang::registry::PackedPacketLayout,
    pub decode: seismic_lang::registry::DecodeRecipe,
}

#[derive(Clone, Debug)]
pub struct ExternalRepresentationGeometry {
    pub info: &'static seismic_lang::registry::RepresentationInfo,
    pub layout: seismic_lang::registry::ExternalPacketLayout,
}

#[derive(Clone, Debug)]
pub enum ReadableRepresentationGeometry {
    Dense(DenseRepresentationGeometry),
    Packed(PackedRepresentationGeometry),
}

impl RepresentationGeometry {
    pub fn of(representation: RepresentationId) -> Self {
        let info = seismic_lang::registry::representation_info(representation);
        let decode = match info.kind {
            seismic_lang::registry::RepresentationKind::Dense(_) => None,
            seismic_lang::registry::RepresentationKind::Packed(_) => {
                seismic_lang::registry::decode_recipe(representation, info.decoded)
            }
            seismic_lang::registry::RepresentationKind::External(_) => None,
        };
        Self { info, decode }
    }

    pub fn readable(&self) -> ReadableRepresentationGeometry {
        match &self.info.kind {
            seismic_lang::registry::RepresentationKind::Dense(dtype) => {
                ReadableRepresentationGeometry::Dense(DenseRepresentationGeometry {
                    info: self.info,
                    dtype: *dtype,
                })
            }
            seismic_lang::registry::RepresentationKind::Packed(layout) => {
                ReadableRepresentationGeometry::Packed(PackedRepresentationGeometry {
                    info: self.info,
                    layout: layout.clone(),
                    decode: self
                        .decode
                        .clone()
                        .expect("registered packed representation has no decode recipe"),
                })
            }
            seismic_lang::registry::RepresentationKind::External(_) => {
                panic!("external representation is not element-readable")
            }
        }
    }

    pub fn dense(&self) -> DenseRepresentationGeometry {
        let seismic_lang::registry::RepresentationKind::Dense(dtype) = &self.info.kind else {
            panic!("non-dense representation reached a dense kernel operation")
        };
        DenseRepresentationGeometry {
            info: self.info,
            dtype: *dtype,
        }
    }

    pub fn packed(&self) -> PackedRepresentationGeometry {
        let seismic_lang::registry::RepresentationKind::Packed(layout) = &self.info.kind else {
            panic!("non-packed representation reached a packed kernel operation")
        };
        PackedRepresentationGeometry {
            info: self.info,
            layout: layout.clone(),
            decode: self
                .decode
                .clone()
                .expect("registered packed representation has no decode recipe"),
        }
    }

    pub fn external(&self) -> ExternalRepresentationGeometry {
        let seismic_lang::registry::RepresentationKind::External(layout) = &self.info.kind else {
            panic!("non-external representation reached an external conversion operation")
        };
        ExternalRepresentationGeometry {
            info: self.info,
            layout: layout.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BindingEmissionLayout {
    pub slot: crate::kernel::BindingSlot,
    pub access: crate::kernel::ops::BindingAccess,
    pub words: BindingWordLayout,
    pub geometry: RepresentationGeometry,
}

#[derive(Clone, Debug)]
pub struct LocalEmissionLayout {
    pub kind: crate::storage::LaunchLocalKind,
    pub alignment: u64,
    pub words: LocalWordLayout,
    pub geometry: RepresentationGeometry,
    pub realization: LocalRealization,
}

#[derive(Clone, Debug)]
pub struct AddressableResourceEmissionLayout {
    pub handle: crate::kernel::ops::AddressableResourceHandle,
    pub class_id: ResourceClassId,
    pub class: AddressableResourceClass,
    pub words: AddressableResourceWordLayout,
    pub alignment_units: u64,
    pub lifetime: ResourceLifetime,
}

/// Compiler-owned static layout passed to native compilation together with
/// the typed kernel. It replaces backend KernelShape/WordLayout mirrors.
#[derive(Clone, Debug)]
pub struct KernelEmissionLayout {
    pub words: KernelWordLayout,
    pub bindings: Vec<BindingEmissionLayout>,
    pub locals: Vec<LocalEmissionLayout>,
    pub addressable_resources: Vec<AddressableResourceEmissionLayout>,
    pub scalar_args: Vec<crate::repr::ScalarKind>,
    pub result_types: Vec<crate::repr::ScalarKind>,
}

/// Physical realization of one launch-local address space. This is a
/// profile fact, fixed before planning; executors never choose it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LocalRealization {
    /// The native launch owns storage for this class. Its exact canonical
    /// total is passed to the native launch mechanism.
    NativeDynamic,
    /// The native compiler realizes a statically-sized local. Factories for
    /// such a profile must construct only closed constant extents.
    NativeStatic,
    /// Core allocates one invocation buffer containing one canonical class
    /// total per workgroup.
    InvocationScratchPerWorkgroup,
    /// Core allocates one invocation buffer containing `bytes_per_participant
    /// * grid * workgroup` and binds it to the launch.
    InvocationScratchPerParticipant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LocalRealizationPolicy {
    pub workgroup: LocalRealization,
    pub participant: LocalRealization,
    pub register: LocalRealization,
}

impl LocalRealizationPolicy {
    pub fn for_kind(self, kind: crate::storage::LaunchLocalKind) -> LocalRealization {
        match kind {
            crate::storage::LaunchLocalKind::Workgroup => self.workgroup,
            crate::storage::LaunchLocalKind::Participant => self.participant,
            crate::storage::LaunchLocalKind::Register => self.register,
        }
    }
}

/// Backend-owned, profile-fixed kernel ABI layout. Planning and native
/// emission consume this same object; neither may rederive a shadow layout.
pub trait KernelAbiModel<B: PhysicalDialect>:
    Clone + fmt::Debug + PartialEq + Send + Sync + 'static
{
    fn layout(&self, kernel: &crate::kernel::Kernel<B>) -> KernelAbiLayout;
}

/// Data-type support matrix.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DataTypeSupport {
    pub scalars: BTreeSet<DType>,
    pub atomics: BTreeSet<DType>,
    pub representations: BTreeSet<RepresentationId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VectorOperationClass {
    Splat,
    Binary(crate::kernel::ops::BinaryOp),
    Unary(crate::kernel::ops::UnaryOp),
    Bit(crate::kernel::ops::BitOp),
    Fma,
    Cast { to: DType },
    Lane,
    ReduceAdd,
    Read { representation: RepresentationId },
    Write { representation: RepresentationId },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VectorSupportEntry {
    pub dtype: DType,
    pub lanes: u16,
    pub operations: Vec<VectorOperationClass>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct VectorSupport {
    pub entries: Vec<VectorSupportEntry>,
}

impl VectorSupport {
    pub fn supports(&self, dtype: DType, lanes: u16, operation: VectorOperationClass) -> bool {
        self.entries.iter().any(|entry| {
            entry.dtype == dtype && entry.lanes == lanes && entry.operations.contains(&operation)
        })
    }
}

/// Numerical environment (§4.1, §9.4): the exact operation semantics a
/// backend guarantees when fast math is off, and which relaxations exist as
/// explicit implementation choices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NumericalEnvironment {
    pub contraction_available: bool,
    pub flush_to_zero_available: bool,
    pub approximate_transcendentals: BTreeSet<&'static str>,
    pub denormals_preserved: bool,
}
