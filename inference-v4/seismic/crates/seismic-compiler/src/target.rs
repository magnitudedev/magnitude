//! Backend device/native/execution contracts and capability registry.
//!
//! Machine truth has three non-interchangeable owners: `DeviceContract<B>`
//! for device-wide legality and numerics, reflection-reconciled
//! `NativeKernel<B>` for concrete function facts, and `ExecutionProfile<B>`
//! for measured performance. Only closed native kernels may enter planning.
//!
//! W5 (shared) owns the registry assembly internals and each backend crate
//! owns its `Backend` impl and profile discovery. The types and traits here
//! are frozen.

use crate::errors::{NativeCompilationError, TargetError};
use crate::executable::NativeExecutor;
pub use crate::identity::IntrinsicIdentityBuilder;
use crate::implementation::ImplementationFactory;
use crate::kernel::ops::ClosedOpView;
use seismic_lang::expr::{ExprArena, SymbolId, SymbolSort, SymbolValue, TargetConstantId};
use seismic_lang::ids::{CapabilityId, IntrinsicId, RepresentationId};
use seismic_lang::registry::BackendName;
use seismic_lang::types::DType;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fmt;

/// A backend: a closed family of intrinsics, target facts, native kernels
/// and commands, and device services. Implemented once per backend crate.
pub trait Backend: Sized + 'static + fmt::Debug {
    const NAME: BackendName;

    /// Backend kernel intrinsic operations admitted in typed kernel IR.
    /// Parameterized by backend so a Metal intrinsic cannot appear in a CUDA
    /// kernel (§7.3).
    type Intrinsic: Clone + fmt::Debug + Send + Sync + 'static;

    /// Backend-specific target facts beyond the common limits (SIMD width,
    /// compute capability, language version, ...). Complete before planning.
    type Facts: Clone + fmt::Debug + PartialEq + Send + Sync + 'static;

    /// Exact argument-table layout shared by planning and native emission.
    type KernelAbi: KernelAbiModel<Self>;
    /// Backend-native launch mode carried by an executable. Backends without
    /// cooperative launch use a unit-like type, so their executor cannot
    /// receive an unsupported compiler mode.
    type NativeLaunchMode: Clone
        + fmt::Debug
        + PartialEq
        + Eq
        + std::hash::Hash
        + Send
        + Sync
        + 'static;
    fn independent_launch_mode() -> Self::NativeLaunchMode;
    fn cooperative_launch_mode(facts: &Self::Facts) -> Option<Self::NativeLaunchMode>;

    /// Backend-specific legality derived from one reconciled native function.
    /// This is evaluated during native closure, before `PlanSpace`; it is the
    /// place for rules such as CUDA cooperative-grid residency whose inputs
    /// combine device-wide limits, reflected function resources, and symbolic
    /// launch geometry.
    fn native_launch_constraints(
        _target: &DeviceContract<Self>,
        _arena: &mut ExprArena,
        _launch: &crate::schedule::Launch,
        _locals: &crate::storage::LaunchLocalLayout,
        _kernel: &crate::kernel::Kernel<Self>,
        _native: &NativeKernel<Self>,
    ) -> Vec<seismic_lang::expr::BoolExpr> {
        Vec::new()
    }

    /// Native addressable resource classes available to intrinsic
    /// implementations on this exact target. Ordinary byte-addressable
    /// storage belongs in launch locals; this is for hardware state such as
    /// CUDA tensor memory whose native unit and ownership scope are distinct.
    fn addressable_resource_classes(_facts: &Self::Facts) -> Vec<AddressableResourceClass> {
        Vec::new()
    }

    /// Canonically identifies one backend intrinsic without Debug rendering
    /// or opaque backend-supplied bytes.
    fn write_intrinsic_identity(
        intrinsic: &Self::Intrinsic,
        identity: &mut IntrinsicIdentityBuilder,
    );

    /// Registered intrinsic signatures for which the native compiler has an
    /// exhaustive emitter arm. Registry assembly proves exact equality with
    /// capability registrations.
    fn emitted_intrinsics() -> BTreeSet<IntrinsicId>;

    /// The exact finite service-class vocabulary exercised by this backend's
    /// core commands, primitive operations, vector operations, and
    /// capability intrinsics. Profile assembly requires exactly one
    /// definition for every class and rejects unused definitions.
    fn required_service_classes(
        facts: &Self::Facts,
        supported_intrinsics: &BTreeSet<IntrinsicId>,
    ) -> BTreeSet<ServiceClassId>;

    /// Execution demand of one already-closed operation. This is the same
    /// typed operation the native emitter consumes; it may not classify by
    /// source spelling or reconstruct a shadow operation graph.
    fn execution_demand(
        target: &DeviceContract<Self>,
        arena: &mut ExprArena,
        kernel: &crate::kernel::Kernel<Self>,
        emission: &KernelEmissionLayout,
        op: ClosedOpView<'_, Self>,
    ) -> Vec<ExecutionDemand>;

    /// Refines structural service demand with facts owned by the concrete
    /// reflected native function. The default is exact identity for backends
    /// whose reflected resources do not change service demand.
    fn refine_execution_demand(
        _target: &DeviceContract<Self>,
        _arena: &mut ExprArena,
        _launch: &crate::schedule::Launch,
        _locals: &crate::storage::LaunchLocalLayout,
        _native: &NativeKernel<Self>,
        demand: ExecutionDemand,
    ) -> Vec<ExecutionDemand> {
        vec![demand]
    }

    /// Exact numerical semantics of the concrete intrinsic instruction this
    /// target emits. The registry supplies target-independent source
    /// semantics; this hook closes target facts such as accumulator rounding
    /// and subnormal handling before the solver derives admissibility.
    fn intrinsic_numerics(
        facts: &Self::Facts,
        signature: &seismic_lang::registry::IntrinsicSignature,
        intrinsic: &Self::Intrinsic,
    ) -> IntrinsicNumericalSemantics;

    /// Addressable-resource handles referenced by one concrete intrinsic.
    /// Core uses this projection to prove that every lease has an owner and
    /// that operation-lifetime state belongs to exactly one emitted op.
    fn intrinsic_addressable_resources(
        intrinsic: &Self::Intrinsic,
    ) -> Vec<crate::kernel::ops::AddressableResourceHandle>;

    /// Pure launch requirements of one registry-validated intrinsic. Core
    /// collects every intrinsic in a segment, intersects the requirements,
    /// and constructs the final domain before any kernel emission begins.
    fn semantic_intrinsic_requirements(
        target: &DeviceContract<Self>,
        arena: &mut seismic_lang::expr::ExprArena,
        signature: &seismic_lang::registry::IntrinsicSignature,
        parallel_extent: seismic_lang::expr::NatExpr,
    ) -> crate::kernel::ops::SemanticIntrinsicLaunchRequirements;

    /// Lowers one registry-validated intrinsic in an authored backend
    /// lowering/helper body. The single core semantic walker has already
    /// checked operand categories and allocated an owned result destination;
    /// the backend must emit the declared result exactly once through the
    /// sink. The universal reference walker never calls this hook.
    fn lower_semantic_intrinsic(
        target: &DeviceContract<Self>,
        domain: &crate::kernel::ops::SegmentLaunchDomain,
        call: crate::kernel::ops::SemanticIntrinsicCall<'_>,
        sink: &mut crate::kernel::ops::SemanticIntrinsicSink<'_, '_, Self>,
    );

    /// Backend-private native output. It is deliberately unusable by
    /// planning and execution until core consumes it together with
    /// authoritative reflection.
    type RawNativeCandidate: Send + 'static;
    /// Backend-native executable handle stored only inside a reconciled
    /// [`NativeKernel`].
    type NativeKernelHandle: Send + Sync + 'static;
    /// Exact native numerical mode reflected or enforced for one artifact.
    type NativeNumericalMode: Clone
        + fmt::Debug
        + PartialEq
        + Eq
        + std::hash::Hash
        + Send
        + Sync
        + 'static;

    /// Forms native code for one fully fixed kernel template. The raw result
    /// has no public planning or launch surface.
    fn form_native_kernel_candidate(
        target: &DeviceContract<Self>,
        kernel: &crate::kernel::Kernel<Self>,
        layout: &KernelEmissionLayout,
    ) -> Result<Self::RawNativeCandidate, NativeCompilationError>;

    /// Consumes a raw native artifact and returns authoritative reflection.
    /// Core alone reconciles this with the template and constructs the
    /// executable [`NativeKernel`].
    fn reflect_native_kernel(
        target: &DeviceContract<Self>,
        kernel: &crate::kernel::Kernel<Self>,
        layout: &KernelEmissionLayout,
        candidate: Self::RawNativeCandidate,
    ) -> Result<NativeKernelReflection<Self>, NativeCompilationError>;

    /// The executor service type the runtime drives.
    type Executor: NativeExecutor<Self>;
}

/// Profile-local identity of one native addressable intrinsic resource.
/// The constructor is core-owned; backends obtain ids by stable-name lookup
/// on the assembled profile and cannot forge another target's class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceClassId(u32);

impl ResourceClassId {
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

/// Identity of one target profile: the cache-key component (§15.1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompatibilityIdentity {
    pub backend: BackendName,
    pub backend_revision: &'static str,
    pub hardware: String,
    pub driver: String,
    pub toolchain: String,
    /// Digest over every fact that changes admissibility or selection.
    pub fingerprint: [u8; 32],
}

/// Stable identity of the device-wide numerical environment. It excludes
/// all performance observations, acquisition metadata, and qualification
/// results so reopening an unchanged device does not invalidate numerical
/// evidence.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NumericalEnvironmentIdentity {
    pub backend: BackendName,
    pub fingerprint: [u8; 32],
}

/// Stable identity of one concrete native artifact's numerical mode. An
/// evidence key combines this with `NumericalEnvironmentIdentity`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeNumericalModeIdentity {
    pub fingerprint: [u8; 32],
}

/// Identity of one assembled device contract. Execution profiles bind this
/// identity without acquiring ownership of the contract's legality facts.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceContractIdentity {
    pub backend: BackendName,
    pub fingerprint: [u8; 32],
}

/// Per-open profile identity. Timing observations and probe methodology live
/// here, never in the stable native-code compatibility identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExecutionProfileIdentity {
    pub device: DeviceContractIdentity,
    pub probe_suite_revision: &'static str,
    pub fingerprint: [u8; 32],
}

/// Sealed identity of one execution service. Names are stable registry
/// identities, not display strings or late lookup conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ServiceClassId(&'static str);

impl ServiceClassId {
    pub const fn new(stable_name: &'static str) -> Self {
        Self(stable_name)
    }
    pub const fn stable_name(self) -> &'static str {
        self.0
    }
}

pub const SERVICE_SUBMISSION: ServiceClassId = ServiceClassId::new("core.submission");
pub const SERVICE_COPY: ServiceClassId = ServiceClassId::new("core.copy");
pub const SERVICE_FILL: ServiceClassId = ServiceClassId::new("core.fill");
pub const SERVICE_SCALAR_READ: ServiceClassId = ServiceClassId::new("core.scalar-read");
pub const SERVICE_SCALAR_MOVE: ServiceClassId = ServiceClassId::new("core.scalar-move");
pub const SERVICE_DATA_CHECK: ServiceClassId = ServiceClassId::new("core.data-check");

pub fn core_service_classes() -> BTreeSet<ServiceClassId> {
    [
        SERVICE_SUBMISSION,
        SERVICE_COPY,
        SERVICE_FILL,
        SERVICE_SCALAR_READ,
        SERVICE_SCALAR_MOVE,
        SERVICE_DATA_CHECK,
    ]
    .into_iter()
    .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DurationInterval {
    pub lower_numerator: u64,
    pub upper_numerator: u64,
    pub denominator: u64,
}

impl DurationInterval {
    pub fn new(lower_numerator: u64, upper_numerator: u64, denominator: u64) -> Self {
        assert!(denominator != 0, "service duration denominator is zero");
        assert!(
            lower_numerator <= upper_numerator,
            "service duration interval is inverted"
        );
        Self {
            lower_numerator,
            upper_numerator,
            denominator,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SingleMeasurementSeries {
    workload_units: u64,
    observations_ns: Box<[u64]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PairedAdjacentMeasurementSeries {
    small_workload_units: u64,
    large_workload_units: u64,
    small_observations_ns: Box<[u64]>,
    large_observations_ns: Box<[u64]>,
}

/// Raw timer observations retained by one measurement batch. Adjacent-count
/// estimators keep both unsigned timer readings here; a backend derives signed
/// differences without forcing a potentially negative observation into u64.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MeasurementSeries {
    Single(SingleMeasurementSeries),
    PairedAdjacent(PairedAdjacentMeasurementSeries),
}

impl MeasurementSeries {
    pub fn single(workload_units: u64, observations_ns: Vec<u64>) -> Self {
        assert!(workload_units != 0, "measurement workload is empty");
        assert!(
            !observations_ns.is_empty(),
            "measurement observations are empty"
        );
        Self::Single(SingleMeasurementSeries {
            workload_units,
            observations_ns: observations_ns.into_boxed_slice(),
        })
    }

    pub fn paired_adjacent(
        small_workload_units: u64,
        large_workload_units: u64,
        small_observations_ns: Vec<u64>,
        large_observations_ns: Vec<u64>,
    ) -> Self {
        assert!(
            small_workload_units != 0 && small_workload_units < large_workload_units,
            "adjacent measurement workloads are not increasing"
        );
        assert!(
            !small_observations_ns.is_empty()
                && small_observations_ns.len() == large_observations_ns.len(),
            "adjacent measurement observations are empty or unpaired"
        );
        Self::PairedAdjacent(PairedAdjacentMeasurementSeries {
            small_workload_units,
            large_workload_units,
            small_observations_ns: small_observations_ns.into_boxed_slice(),
            large_observations_ns: large_observations_ns.into_boxed_slice(),
        })
    }

    pub fn single_parts(&self) -> Option<(u64, &[u64])> {
        match self {
            Self::Single(series) => Some((series.workload_units, &series.observations_ns)),
            Self::PairedAdjacent(_) => None,
        }
    }

    pub fn paired_adjacent_parts(&self) -> Option<(u64, u64, &[u64], &[u64])> {
        match self {
            Self::Single(_) => None,
            Self::PairedAdjacent(series) => Some((
                series.small_workload_units,
                series.large_workload_units,
                &series.small_observations_ns,
                &series.large_observations_ns,
            )),
        }
    }

    fn is_valid(&self) -> bool {
        match self {
            Self::Single(series) => {
                series.workload_units != 0 && !series.observations_ns.is_empty()
            }
            Self::PairedAdjacent(series) => {
                series.small_workload_units != 0
                    && series.small_workload_units < series.large_workload_units
                    && !series.small_observations_ns.is_empty()
                    && series.small_observations_ns.len() == series.large_observations_ns.len()
            }
        }
    }

    fn update_identity(&self, digest: &mut Sha256) {
        match self {
            Self::Single(series) => {
                digest.update(b"measurement-series/single/v1");
                digest.update(series.workload_units.to_le_bytes());
                digest.update((series.observations_ns.len() as u64).to_le_bytes());
                for observation in &series.observations_ns {
                    digest.update(observation.to_le_bytes());
                }
            }
            Self::PairedAdjacent(series) => {
                digest.update(b"measurement-series/paired-adjacent/v1");
                digest.update(series.small_workload_units.to_le_bytes());
                digest.update(series.large_workload_units.to_le_bytes());
                digest.update((series.small_observations_ns.len() as u64).to_le_bytes());
                for (small, large) in series
                    .small_observations_ns
                    .iter()
                    .zip(&series.large_observations_ns)
                {
                    digest.update(small.to_le_bytes());
                    digest.update(large.to_le_bytes());
                }
            }
        }
    }
}

#[cfg(test)]
mod measurement_series_tests {
    use super::*;

    #[test]
    fn single_series_retains_workload_and_observation_order() {
        let series = MeasurementSeries::single(7, vec![11, 13, 12]);
        assert_eq!(series.single_parts(), Some((7, [11, 13, 12].as_slice())));
        assert!(series.paired_adjacent_parts().is_none());
    }

    #[test]
    fn paired_series_retains_both_unsigned_sides_and_pair_order() {
        let series =
            MeasurementSeries::paired_adjacent(10, 20, vec![120, 150, 110], vec![140, 130, 160]);
        let (small_work, large_work, small, large) = series.paired_adjacent_parts().unwrap();
        assert_eq!((small_work, large_work), (10, 20));
        assert_eq!(small, [120, 150, 110]);
        assert_eq!(large, [140, 130, 160]);
        assert!(series.single_parts().is_none());
    }

    #[test]
    fn series_identity_domain_separates_variant_workload_and_order() {
        fn identity(series: &MeasurementSeries) -> [u8; 32] {
            let mut digest = Sha256::new();
            series.update_identity(&mut digest);
            digest.finalize().into()
        }
        let single = MeasurementSeries::single(10, vec![100, 200]);
        let different_work = MeasurementSeries::single(11, vec![100, 200]);
        let different_order = MeasurementSeries::single(10, vec![200, 100]);
        let paired = MeasurementSeries::paired_adjacent(5, 10, vec![40, 50], vec![100, 200]);
        assert_ne!(identity(&single), identity(&different_work));
        assert_ne!(identity(&single), identity(&different_order));
        assert_ne!(identity(&single), identity(&paired));
    }

    #[test]
    #[should_panic(expected = "empty or unpaired")]
    fn paired_series_rejects_unpaired_observations() {
        MeasurementSeries::paired_adjacent(1, 2, vec![10], vec![20, 30]);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeasurementBatch {
    pub probe: &'static str,
    pub method: &'static str,
    pub timer_resolution_ns: DurationInterval,
    /// Raw backend timer observations and their exact workload identity.
    pub series: MeasurementSeries,
    pub observations_digest: [u8; 32],
    pub acquisition_duration_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FactProvenance {
    Queried {
        api: &'static str,
        field: &'static str,
    },
    Derived {
        rule: &'static str,
        inputs: Box<[&'static str]>,
    },
    Measured {
        /// Independent batches preserve the distinct workload and method
        /// used for dependency, setup, and capacity observations.
        batches: Box<[MeasurementBatch]>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ResourceTopology {
    pub resources: u32,
    pub max_concurrency: u32,
}

/// Stable identity of service coefficients derived from the same underlying
/// observation. Interval composition retains this identity so repeated uses
/// never manufacture independence and narrow uncertainty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ServiceCorrelationId(&'static str);

impl ServiceCorrelationId {
    pub const fn new(stable_name: &'static str) -> Self {
        Self(stable_name)
    }

    pub const fn stable_name(self) -> &'static str {
        self.0
    }
}

/// Finite workload domain for which one measured service definition makes a
/// prediction claim. Demands outside it cannot enter a qualified plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ServiceQualificationDomain {
    pub minimum_units: u64,
    pub maximum_units: u64,
    pub maximum_concurrent_uses: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ServiceDefinition {
    pub class: ServiceClassId,
    pub correlation: ServiceCorrelationId,
    pub qualification: ServiceQualificationDomain,
    pub accuracy: ServiceAccuracyClass,
    pub topology: ResourceTopology,
    pub dependency_latency: DurationInterval,
    pub saturated_capacity: ServiceCurve,
    pub provenance: FactProvenance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ServiceAccuracyClass {
    /// Compute/control dependency and issue services: at most 1% relative
    /// interval half-width.
    Compute,
    /// Memory and transfer services: at most 2% relative interval half-width.
    MemoryOrTransfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ServiceCurveRegime {
    /// Inclusive maximum demand served by this regime. `None` is the final
    /// unbounded regime and must appear exactly once at the end.
    pub max_units: Option<u64>,
    pub per_unit: DurationInterval,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ServiceCurve {
    /// Fixed setup/dispatch contribution for exercising the service once.
    pub setup: DurationInterval,
    /// Ordered, non-empty working-set/concurrency regimes.
    pub regimes: Vec<ServiceCurveRegime>,
}

/// Observed lifecycle latency of constructing this exact opened profile.
/// These values are telemetry, not execution-model coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ProfileAcquisitionMetrics {
    pub identity_and_limits_ns: u64,
    pub probe_build_ns: u64,
    pub probe_execution_ns: u64,
    pub total_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompositionQualificationCase {
    pub stable_name: &'static str,
    /// Fixed regular composition stated only in sealed service demand. Core
    /// predicts it through the production service model; backends may not
    /// provide or fit a parallel prediction.
    pub demands: Vec<ConcreteExecutionDemand>,
    pub observed_ns: DurationInterval,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConcreteExecutionDemand {
    pub class: ServiceClassId,
    pub units: u64,
    pub mode: DemandMode,
}

/// Fixed held-out composition qualification of the service model. These
/// cases are never candidate workloads and never fit model coefficients.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompositionQualification {
    pub suite_revision: &'static str,
    pub cases: Vec<CompositionQualificationCase>,
    pub maximum_relative_error_basis_points: u16,
    pub observations_digest: [u8; 32],
}

/// Backend-supplied held-out evidence. It contains no predicted value or
/// claimed error; core computes both through the production service model.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CompositionQualificationParts {
    pub suite_revision: &'static str,
    pub cases: Vec<CompositionQualificationCase>,
    pub observations_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DemandMode {
    DependencyLatency,
    SaturatedCapacity,
}

/// The execution population over which `ExecutionDemand::units` is stated.
/// Backends choose this from the semantics of the emitted operation; core
/// alone applies the launch geometry, so aggregate launch work is never
/// accidentally multiplied once per participant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DemandScope {
    /// `units` is incurred independently by every launch participant.
    PerParticipant,
    /// `units` is the aggregate demand of the complete launch.
    PerLaunch,
}

#[derive(Clone, Copy, Debug)]
pub struct ExecutionDemand {
    pub class: ServiceClassId,
    pub units: seismic_lang::expr::NatExpr,
    pub mode: DemandMode,
    pub scope: DemandScope,
}

/// One service-model term retained beside the duration expression so
/// correlation and qualification are not erased by interval arithmetic.
#[derive(Clone, Copy, Debug)]
pub struct ServiceModelTerm {
    /// The semantic launch whose physical population incurs this demand.
    /// Host-side schedule services are not launch-chunkable and use `None`.
    pub(crate) launch_ordinal: Option<u32>,
    pub class: ServiceClassId,
    pub correlation: ServiceCorrelationId,
    pub units: seismic_lang::expr::NatExpr,
    pub mode: DemandMode,
}

/// A duration is usable for selection only together with every finite-domain
/// qualification condition produced by its service terms.
#[derive(Clone, Debug)]
pub struct QualifiedDuration {
    pub duration: seismic_lang::expr::DurationExpr,
    pub qualification: Vec<seismic_lang::expr::BoolExpr>,
    pub terms: Vec<ServiceModelTerm>,
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

/// Stable identity of one concrete native artifact. The compatibility
/// identity includes the exact backend/toolchain contract under which the
/// artifact may be reused; the digest identifies its code and metadata.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeKernelIdentity {
    pub compatibility: CompatibilityIdentity,
    pub artifact_digest: [u8; 32],
}

/// Launch space accepted by one concrete reflected native function. These
/// are function/pipeline limits, never copied from device-wide maxima.
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
pub enum NativeClusterDomain {
    NotApplicable,
    Required {
        dimensions: [u32; 3],
        portability: ClusterPortability,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClusterPortability {
    Portable,
    NonPortableAllowed,
}

/// A native resource fact has one explicit establishment route. There is no
/// unknown or zero sentinel: unsupported reflection prevents reconciliation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NativeResourceUsage {
    NotApplicable,
    Exact(u64),
    EnforcedUpperBound(u64),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeKernelResources {
    pub registers_per_participant: NativeResourceUsage,
    pub spill_bytes_per_participant: NativeResourceUsage,
    pub static_local_bytes: NativeResourceUsage,
}

/// Preparation cost of forming and reflecting this exact native artifact.
/// These values charge the shared preparation budget and never enter the
/// execution-duration model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NativeArtifactMetrics {
    pub compilation_ns: u64,
    pub code_bytes: u64,
    pub metadata_bytes: u64,
}

/// Backend reflection result. Backends construct this only from the concrete
/// native artifact; core consumes it and reconciles it with the fixed kernel
/// template before any planning API can observe the result.
pub struct NativeKernelReflection<B: Backend> {
    pub handle: B::NativeKernelHandle,
    pub identity: NativeKernelIdentity,
    pub abi: KernelAbiLayout,
    pub launch: NativeLaunchDomain<B::NativeLaunchMode>,
    pub resources: NativeKernelResources,
    pub numerics: B::NativeNumericalMode,
    pub numerical_identity: NativeNumericalModeIdentity,
    pub artifact: NativeArtifactMetrics,
}

impl<B: Backend> fmt::Debug for NativeKernelReflection<B>
where
    B::NativeKernelHandle: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeKernelReflection")
            .field("handle", &self.handle)
            .field("identity", &self.identity)
            .field("abi", &self.abi)
            .field("launch", &self.launch)
            .field("resources", &self.resources)
            .field("numerics", &self.numerics)
            .field("numerical_identity", &self.numerical_identity)
            .field("artifact", &self.artifact)
            .finish()
    }
}

/// Unusable output of native formation. Its fields and constructor are
/// private; only the core reconciliation transition may consume it.
pub struct NativeKernelCandidate<B: Backend> {
    raw: B::RawNativeCandidate,
    layout: KernelEmissionLayout,
    abi: KernelAbiLayout,
    compatibility: CompatibilityIdentity,
    services: BTreeSet<ServiceClassId>,
}

/// Reconciled native contract owned by a concrete kernel.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeKernelContract<B: Backend> {
    pub identity: NativeKernelIdentity,
    pub abi: KernelAbiLayout,
    pub launch: NativeLaunchDomain<B::NativeLaunchMode>,
    pub resources: NativeKernelResources,
    pub numerics: B::NativeNumericalMode,
    pub numerical_identity: NativeNumericalModeIdentity,
    pub services: BTreeSet<ServiceClassId>,
    pub artifact: NativeArtifactMetrics,
}

/// The only native-kernel value planning and execution may own. It cannot be
/// constructed from a raw candidate without core reconciliation.
pub struct NativeKernel<B: Backend> {
    handle: B::NativeKernelHandle,
    contract: NativeKernelContract<B>,
}

impl<B: Backend> NativeKernel<B> {
    pub fn handle(&self) -> &B::NativeKernelHandle {
        &self.handle
    }

    pub fn contract(&self) -> &NativeKernelContract<B> {
        &self.contract
    }
}

impl<B: Backend> fmt::Debug for NativeKernel<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeKernel")
            .field("contract", &self.contract)
            .finish_non_exhaustive()
    }
}

/// Canonical word-table slice for one bound tensor.
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
    pub fn for_kernel<B: Backend>(kernel: &crate::kernel::Kernel<B>) -> Self {
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
    pub scalar_args: Vec<DType>,
    pub result_slots: Vec<(crate::schedule::AnyScalarSlot, DType)>,
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
pub trait KernelAbiModel<B: Backend>:
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

/// Immutable device-wide legality and numerical contract. No measured
/// performance or concrete-kernel resource fact may enter this type.
#[derive(Debug)]
pub struct DeviceContract<B: Backend> {
    identity: DeviceContractIdentity,
    compatibility_identity: CompatibilityIdentity,
    numerical_identity: NumericalEnvironmentIdentity,
    limits: TargetLimits,
    dtypes: DataTypeSupport,
    vectors: VectorSupport,
    numerics: NumericalEnvironment,
    capabilities: BTreeSet<CapabilityId>,
    intrinsics: BTreeSet<IntrinsicId>,
    facts: B::Facts,
    kernel_abi: B::KernelAbi,
    local_realization: LocalRealizationPolicy,
    addressable_resources: Vec<AddressableResourceClass>,
    registry: &'static CapabilityRegistry<B>,
}

/// Device-wide facts gathered from the exact opened service. Consumed by
/// [`DeviceContract::assemble`].
#[derive(Debug)]
pub struct DeviceContractParts<B: Backend> {
    pub identity: CompatibilityIdentity,
    pub limits: TargetLimits,
    pub dtypes: DataTypeSupport,
    pub vectors: VectorSupport,
    pub numerics: NumericalEnvironment,
    pub facts: B::Facts,
    pub kernel_abi: B::KernelAbi,
    pub local_realization: LocalRealizationPolicy,
}

/// Per-open measured execution behavior. This type owns no legality,
/// capability, numerical, ABI, or native-kernel facts.
#[derive(Debug)]
pub struct ExecutionProfile<B: Backend> {
    identity: ExecutionProfileIdentity,
    services: Vec<ServiceDefinition>,
    acquisition: ProfileAcquisitionMetrics,
    composition_qualification: CompositionQualification,
    _backend: std::marker::PhantomData<fn() -> B>,
}

#[derive(Debug)]
pub struct ExecutionProfileParts {
    pub probe_suite_revision: &'static str,
    pub services: Vec<ServiceDefinition>,
    pub acquisition: ProfileAcquisitionMetrics,
    pub composition_qualification: CompositionQualificationParts,
}

/// Sealed borrowed capability proving that a device contract and execution
/// profile were assembled for the same opened machine. It owns and copies no
/// machine facts.
pub struct PlanningMachine<'a, B: Backend> {
    device: &'a DeviceContract<B>,
    execution: &'a ExecutionProfile<B>,
}

impl<B: Backend> Copy for PlanningMachine<'_, B> {}
impl<B: Backend> Clone for PlanningMachine<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, B: Backend> PlanningMachine<'a, B> {
    pub fn device(&self) -> &'a DeviceContract<B> {
        self.device
    }

    pub fn execution(&self) -> &'a ExecutionProfile<B> {
        self.execution
    }
}

impl<B: Backend> DeviceContract<B> {
    /// The only constructor. Derives the advertised capability and intrinsic
    /// sets from the static registry's predicates over the facts, and fails
    /// with a typed error when the device is below the backend's floor.
    pub fn assemble(
        parts: DeviceContractParts<B>,
        registry: &'static CapabilityRegistry<B>,
    ) -> Result<Self, TargetError> {
        internals::assemble(parts, registry)
    }

    pub fn compatibility_identity(&self) -> &CompatibilityIdentity {
        &self.compatibility_identity
    }
    pub fn identity(&self) -> &DeviceContractIdentity {
        &self.identity
    }
    pub fn numerical_identity(&self) -> &NumericalEnvironmentIdentity {
        &self.numerical_identity
    }
    pub fn planning_with<'a>(
        &'a self,
        execution: &'a ExecutionProfile<B>,
    ) -> PlanningMachine<'a, B> {
        assert_eq!(
            self.identity, execution.identity.device,
            "execution profile belongs to another device contract"
        );
        PlanningMachine {
            device: self,
            execution,
        }
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
    pub fn facts(&self) -> &B::Facts {
        &self.facts
    }
    pub fn kernel_abi(&self) -> &B::KernelAbi {
        &self.kernel_abi
    }
    pub fn local_realization(&self) -> LocalRealizationPolicy {
        self.local_realization
    }
    pub fn kernel_abi_layout(&self, kernel: &crate::kernel::Kernel<B>) -> KernelAbiLayout {
        let layout = self.kernel_abi.layout(kernel);
        let mut roles = BTreeSet::new();
        for allocation in &layout.allocations {
            assert!(
                roles.insert(allocation.role),
                "kernel ABI model declared one allocation role twice"
            );
            assert!(
                allocation.alignment.is_power_of_two(),
                "kernel ABI allocation alignment must be a nonzero power of two"
            );
        }
        layout
    }
    pub fn kernel_emission_layout(
        &self,
        kernel: &crate::kernel::Kernel<B>,
    ) -> KernelEmissionLayout {
        let words = KernelWordLayout::for_kernel(kernel);
        let mut bindings = Vec::with_capacity(kernel.interface().bindings.len());
        for (binding, word_layout) in kernel.interface().bindings.iter().zip(&words.bindings) {
            bindings.push(BindingEmissionLayout {
                slot: binding.slot,
                access: binding.access,
                words: *word_layout,
                geometry: RepresentationGeometry::of(binding.view.representation),
            });
        }
        let mut locals = Vec::with_capacity(kernel.locals().len());
        for (local, word_layout) in kernel.locals().iter().zip(&words.locals) {
            locals.push(LocalEmissionLayout {
                kind: local.kind,
                alignment: local.alignment,
                words: *word_layout,
                geometry: RepresentationGeometry::of(local.representation),
                realization: self.local_realization.for_kind(local.kind),
            });
        }
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
            result_slots: kernel.interface().result_slots.clone(),
        }
    }
    pub fn supports_capability(&self, id: CapabilityId) -> bool {
        self.capabilities.contains(&id)
    }
    pub fn supports_intrinsic(&self, id: IntrinsicId) -> bool {
        self.intrinsics.contains(&id)
    }
    pub fn addressable_resource_class(&self, stable_name: &str) -> Option<ResourceClassId> {
        self.addressable_resources
            .iter()
            .position(|class| class.stable_name == stable_name)
            .map(|index| {
                ResourceClassId(u32::try_from(index).expect("resource class count exceeds u32"))
            })
    }
    pub fn addressable_resource(&self, id: ResourceClassId) -> &AddressableResourceClass {
        self.addressable_resources
            .get(id.0 as usize)
            .expect("resource class id belongs to another device contract")
    }
    pub fn addressable_resources(&self) -> &[AddressableResourceClass] {
        &self.addressable_resources
    }
    pub fn registry(&self) -> &'static CapabilityRegistry<B> {
        self.registry
    }

    /// Binds every target constant the planner may reference into an arena,
    /// returning the mapping. Called once per plan space.
    pub fn bind_constants(&self, arena: &mut ExprArena) -> TargetConstants {
        internals::bind_constants(self, arena)
    }

    /// Forms the deliberately unusable candidate state. This is crate-local
    /// so native formation can only occur in the implementation-closing
    /// path above `PlanSpace`.
    pub(crate) fn form_native_kernel_candidate(
        &self,
        arena: &mut ExprArena,
        kernel: &crate::kernel::Kernel<B>,
    ) -> Result<NativeKernelCandidate<B>, NativeCompilationError> {
        let layout = self.kernel_emission_layout(kernel);
        let abi = self.kernel_abi_layout(kernel);
        fn collect<B: Backend>(
            device: &DeviceContract<B>,
            arena: &mut ExprArena,
            kernel: &crate::kernel::Kernel<B>,
            layout: &KernelEmissionLayout,
            block: crate::kernel::BlockId,
            services: &mut BTreeSet<ServiceClassId>,
        ) {
            for op in &kernel.block(block).ops {
                let closed = kernel.closed_op(op, layout);
                let nested = match &closed {
                    ClosedOpView::Branch {
                        then_block,
                        else_block,
                        ..
                    } => vec![*then_block, *else_block],
                    ClosedOpView::Repeat { body, .. } => vec![*body],
                    _ => Vec::new(),
                };
                services.extend(
                    B::execution_demand(device, arena, kernel, layout, closed)
                        .into_iter()
                        .map(|demand| demand.class),
                );
                for child in nested {
                    collect(device, arena, kernel, layout, child, services);
                }
            }
        }
        let mut services = BTreeSet::new();
        collect(self, arena, kernel, &layout, kernel.root(), &mut services);
        let raw = B::form_native_kernel_candidate(self, kernel, &layout)?;
        Ok(NativeKernelCandidate {
            raw,
            layout,
            abi,
            compatibility: self.compatibility_identity.clone(),
            services,
        })
    }

    /// Consumes a candidate and closes it against authoritative reflection.
    /// No candidate survives this transition and no reflected fact is copied
    /// back into the device contract.
    pub(crate) fn reconcile_native_kernel(
        &self,
        kernel: &crate::kernel::Kernel<B>,
        candidate: NativeKernelCandidate<B>,
    ) -> Result<NativeKernel<B>, NativeCompilationError> {
        let NativeKernelCandidate {
            raw,
            layout,
            abi,
            compatibility,
            services,
        } = candidate;
        let reflection = B::reflect_native_kernel(self, kernel, &layout, raw)?;
        assert_eq!(
            reflection.identity.compatibility, compatibility,
            "native reflection compatibility differs from the formation contract"
        );
        assert_eq!(
            reflection.abi, abi,
            "native reflection ABI differs from the canonical kernel ABI"
        );
        assert!(
            !reflection.launch.modes.is_empty(),
            "native reflection admits no launch mode"
        );
        if kernel.interface().uses_subgroup && reflection.launch.subgroup_width.is_none() {
            return Err(NativeCompilationError::MalformedToolchainOutput(
                "subgroup-using kernel reflection omitted its concrete subgroup width".into(),
            ));
        }
        assert!(
            reflection.launch.max_workgroup_threads != 0
                && reflection
                    .launch
                    .max_workgroup_size
                    .iter()
                    .all(|axis| *axis != 0)
                && reflection.launch.max_grid.iter().all(|axis| *axis != 0)
                && reflection
                    .launch
                    .subgroup_width
                    .is_none_or(|width| width != 0),
            "native reflection contains an empty launch domain"
        );
        if let NativeClusterDomain::Required { dimensions, .. } = reflection.launch.cluster {
            assert!(
                dimensions.into_iter().all(|axis| axis != 0),
                "native reflection contains an empty required cluster dimension"
            );
        }
        assert!(
            reflection.launch.max_workgroup_threads <= self.limits.max_workgroup_threads
                && reflection
                    .launch
                    .max_workgroup_size
                    .iter()
                    .zip(self.limits.max_workgroup_size)
                    .all(|(native, device)| *native <= device)
                && reflection
                    .launch
                    .max_grid
                    .iter()
                    .zip(self.limits.max_grid)
                    .all(|(native, device)| *native <= device)
                && reflection.launch.max_dynamic_local_bytes <= self.limits.max_workgroup_bytes,
            "native reflection exceeds a device-wide hard limit"
        );
        let static_local = match reflection.resources.static_local_bytes {
            NativeResourceUsage::NotApplicable => 0,
            NativeResourceUsage::Exact(bytes) | NativeResourceUsage::EnforcedUpperBound(bytes) => {
                bytes
            }
        };
        assert!(
            static_local
                .checked_add(reflection.launch.max_dynamic_local_bytes)
                .is_some_and(|total| total <= self.limits.max_workgroup_bytes),
            "native static and dynamic local-memory domains exceed the device limit"
        );
        let allowed_services = B::required_service_classes(&self.facts, &self.intrinsics);
        assert!(services.is_subset(&allowed_services));
        assert!(
            reflection.artifact.code_bytes != 0,
            "native reflection reports an empty code artifact"
        );
        let contract = NativeKernelContract {
            identity: reflection.identity,
            abi: reflection.abi,
            launch: reflection.launch,
            resources: reflection.resources,
            numerics: reflection.numerics,
            numerical_identity: reflection.numerical_identity,
            services,
            artifact: reflection.artifact,
        };
        Ok(NativeKernel {
            handle: reflection.handle,
            contract,
        })
    }

    pub(crate) fn from_parts(
        parts: DeviceContractParts<B>,
        capabilities: BTreeSet<CapabilityId>,
        intrinsics: BTreeSet<IntrinsicId>,
        registry: &'static CapabilityRegistry<B>,
    ) -> Self {
        let addressable_resources = B::addressable_resource_classes(&parts.facts);
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
        let compatibility_identity = parts.identity;
        let mut fingerprint = Sha256::new();
        fingerprint.update(b"seismic-target-compatibility-v2");
        fingerprint.update(compatibility_identity.fingerprint);
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
        let mut compatibility_identity = compatibility_identity;
        compatibility_identity.fingerprint = fingerprint.finalize().into();
        let identity = DeviceContractIdentity {
            backend: B::NAME,
            fingerprint: compatibility_identity.fingerprint,
        };
        let mut numerical = Sha256::new();
        numerical.update(b"seismic-numerical-environment-v1");
        numerical.update(B::NAME.as_str().as_bytes());
        numerical.update(compatibility_identity.backend_revision.as_bytes());
        numerical.update(compatibility_identity.toolchain.as_bytes());
        numerical.update([
            parts.numerics.contraction_available as u8,
            parts.numerics.flush_to_zero_available as u8,
            parts.numerics.denormals_preserved as u8,
        ]);
        for operation in &parts.numerics.approximate_transcendentals {
            numerical.update((operation.len() as u64).to_le_bytes());
            numerical.update(operation.as_bytes());
        }
        let numerical_identity = NumericalEnvironmentIdentity {
            backend: B::NAME,
            fingerprint: numerical.finalize().into(),
        };
        Self {
            identity,
            compatibility_identity,
            numerical_identity,
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
            registry,
        }
    }
}

impl<B: Backend> ExecutionProfile<B> {
    pub fn assemble(
        device: &DeviceContract<B>,
        parts: ExecutionProfileParts,
    ) -> Result<Self, TargetError> {
        internals::assemble_execution(device, parts)
    }

    pub fn identity(&self) -> &ExecutionProfileIdentity {
        &self.identity
    }

    pub fn device_identity(&self) -> &DeviceContractIdentity {
        &self.identity.device
    }

    pub fn service(&self, class: ServiceClassId) -> &ServiceDefinition {
        self.services
            .iter()
            .find(|definition| definition.class == class)
            .unwrap_or_else(|| {
                panic!(
                    "ExecutionProfile<{:?}> lacks required service `{}`",
                    B::NAME,
                    class.stable_name()
                )
            })
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

    /// Builds the one physical duration expression from a matching device
    /// contract, closed schedule, and machine-closed native kernels.
    pub fn derive_duration(
        &self,
        device: &DeviceContract<B>,
        arena: &mut ExprArena,
        schedule: &crate::schedule::ParametricSchedule,
        launch_layouts: &[crate::storage::LaunchLocalLayout],
        kernels: &[crate::kernel::Kernel<B>],
    ) -> QualifiedDuration {
        assert_eq!(
            &self.identity.device,
            device.identity(),
            "execution profile belongs to another device contract"
        );
        internals::derive_duration(
            device,
            self,
            arena,
            schedule,
            launch_layouts,
            internals::DurationKernels::Draft(kernels),
            None,
        )
    }

    pub(crate) fn derive_closed_duration(
        &self,
        device: &DeviceContract<B>,
        arena: &mut ExprArena,
        schedule: &crate::schedule::ParametricSchedule,
        launch_layouts: &[crate::storage::LaunchLocalLayout],
        kernels: &crate::kernel::KernelArena<B>,
        native: &[std::sync::Arc<NativeKernel<B>>],
    ) -> QualifiedDuration {
        assert_eq!(
            &self.identity.device,
            device.identity(),
            "execution profile belongs to another device contract"
        );
        internals::derive_duration(
            device,
            self,
            arena,
            schedule,
            launch_layouts,
            internals::DurationKernels::Closed(kernels),
            Some(native),
        )
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
        self.addressable_resource_capacity[id.0 as usize]
    }
}

/// One capability registration (§4.2): the stable id, the intrinsic
/// signatures this backend implements (each with a `TypedIntrinsic<B>`
/// lowering, its resource rules, and native emission), and the predicate
/// over target facts that narrows them per profile. A capability advertised
/// without an implemented lowering, or a lowering without a registered
/// signature, is a startup panic.
pub struct CapabilityRegistration<B: Backend> {
    pub capability: CapabilityId,
    /// Every signature this backend implements. Each has a typed lowering
    /// registered through [`crate::kernel::TypedIntrinsic`] and an emitter
    /// in the native compiler.
    pub implemented: Vec<IntrinsicId>,
    /// Which of `implemented` this target profile supports, given the facts.
    /// May only narrow `implemented`; an empty result means the capability
    /// is not advertised on this target.
    pub supported: fn(&B::Facts, &TargetLimits, &DataTypeSupport) -> BTreeSet<IntrinsicId>,
}

impl<B: Backend> fmt::Debug for CapabilityRegistration<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CapabilityRegistration")
            .field("capability", &self.capability)
            .finish()
    }
}

/// The sealed static registry of one backend, assembled once at compiler
/// initialization. Duplicate ids, a signature without an emitter, or an
/// emitter without a signature are startup panics (§4.2, §13.3.1).
pub struct CapabilityRegistry<B: Backend> {
    inner: internals::Registry<B>,
}

impl<B: Backend> CapabilityRegistry<B> {
    /// Assembles and seals. Panics on an inconsistent registration set.
    pub fn assemble(
        capabilities: Vec<CapabilityRegistration<B>>,
        structural: Vec<Box<dyn ImplementationFactory<B>>>,
    ) -> Self {
        Self {
            inner: internals::Registry::assemble(capabilities, structural),
        }
    }

    /// Every registered factory, capability-bound and structural, in a
    /// stable order.
    pub fn factories(&self) -> &[Box<dyn ImplementationFactory<B>>] {
        self.inner.factories()
    }

    pub fn capability(&self, id: CapabilityId) -> Option<&CapabilityRegistration<B>> {
        self.inner.capability(id)
    }
}

impl<B: Backend> fmt::Debug for CapabilityRegistry<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CapabilityRegistry").finish()
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

    pub(super) struct Registry<B: Backend> {
        capabilities: Vec<CapabilityRegistration<B>>,
        factories: Vec<Box<dyn ImplementationFactory<B>>>,
    }

    impl<B: Backend> Registry<B> {
        /// Seals one backend's registrations. Panics (§13.3.1) on:
        /// a capability registered twice; a capability of another backend;
        /// a capability advertised with no implemented signature; an
        /// implemented signature that is not a registered signature of that
        /// capability (an emitter without a signature); a signature
        /// implemented twice; two structural factories sharing an identity.
        pub(super) fn assemble(
            capabilities: Vec<CapabilityRegistration<B>>,
            structural: Vec<Box<dyn ImplementationFactory<B>>>,
        ) -> Self {
            for (position, registration) in capabilities.iter().enumerate() {
                let capability = registration.capability;
                if capabilities[..position]
                    .iter()
                    .any(|earlier| earlier.capability == capability)
                {
                    panic!(
                        "CapabilityRegistry<{:?}>: capability {capability:?} is registered twice",
                        B::NAME
                    );
                }
                let info = registry::capability_info(capability);
                if info.backend != B::NAME {
                    panic!(
                        "CapabilityRegistry<{:?}>: capability {capability:?} (`{}`) belongs to backend {:?}",
                        B::NAME,
                        info.name,
                        info.backend
                    );
                }
                if registration.implemented.is_empty() {
                    panic!(
                        "CapabilityRegistry<{:?}>: capability `{}` is advertised without an implemented signature",
                        B::NAME,
                        info.name
                    );
                }
                let signatures = registry::intrinsics(capability);
                for (index, intrinsic) in registration.implemented.iter().enumerate() {
                    if registration.implemented[..index].contains(intrinsic) {
                        panic!(
                            "CapabilityRegistry<{:?}>: signature {intrinsic:?} of `{}` is implemented twice",
                            B::NAME,
                            info.name
                        );
                    }
                    if !signatures
                        .iter()
                        .any(|signature| signature.id == *intrinsic)
                    {
                        panic!(
                            "CapabilityRegistry<{:?}>: `{}` implements {intrinsic:?}, which is not a registered signature of that capability (an emitter without a signature)",
                            B::NAME,
                            info.name
                        );
                    }
                }
            }
            for (position, factory) in structural.iter().enumerate() {
                let identity = factory.identity();
                if structural[..position]
                    .iter()
                    .any(|earlier| earlier.identity() == identity)
                {
                    panic!(
                        "CapabilityRegistry<{:?}>: structural factory `{}`/`{}` is registered twice",
                        B::NAME,
                        identity.name,
                        identity.revision
                    );
                }
            }
            let registered: BTreeSet<_> = capabilities
                .iter()
                .flat_map(|registration| registration.implemented.iter().copied())
                .collect();
            let emitted = B::emitted_intrinsics();
            assert_eq!(
                registered,
                emitted,
                "CapabilityRegistry<{:?}>: registered typed intrinsic signatures and native emitter signatures differ",
                B::NAME
            );
            Self {
                capabilities,
                factories: structural,
            }
        }

        pub(super) fn factories(&self) -> &[Box<dyn ImplementationFactory<B>>] {
            &self.factories
        }

        pub(super) fn capability(&self, id: CapabilityId) -> Option<&CapabilityRegistration<B>> {
            self.capabilities
                .iter()
                .find(|registration| registration.capability == id)
        }
    }

    /// The floor every backend profile must clear. Below it the device is
    /// `TargetError::UnsupportedDevice`; nothing in the compiler models a
    /// target without these facts.
    fn check_floor(parts: &DeviceContractParts<impl Backend>) -> Result<(), TargetError> {
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
    pub(super) fn assemble<B: Backend>(
        parts: DeviceContractParts<B>,
        registry: &'static CapabilityRegistry<B>,
    ) -> Result<DeviceContract<B>, TargetError> {
        if parts.identity.backend != B::NAME {
            panic!(
                "DeviceContract<{:?}>: the backend gathered an identity for {:?}",
                B::NAME,
                parts.identity.backend
            );
        }
        check_floor(&parts)?;
        let mut capabilities = BTreeSet::new();
        let mut intrinsics = BTreeSet::new();
        for registration in &registry.inner.capabilities {
            let supported = (registration.supported)(&parts.facts, &parts.limits, &parts.dtypes);
            for intrinsic in &supported {
                if !registration.implemented.contains(intrinsic) {
                    panic!(
                        "CapabilityRegistry<{:?}>: the predicate of capability {:?} advertises {intrinsic:?}, which the registration does not implement",
                        B::NAME,
                        registration.capability
                    );
                }
            }
            if !supported.is_empty() {
                capabilities.insert(registration.capability);
                intrinsics.extend(supported);
            }
        }
        Ok(DeviceContract::from_parts(
            parts,
            capabilities,
            intrinsics,
            registry,
        ))
    }

    pub(super) fn assemble_execution<B: Backend>(
        device: &DeviceContract<B>,
        parts: ExecutionProfileParts,
    ) -> Result<ExecutionProfile<B>, TargetError> {
        let mut required = core_service_classes();
        required.extend(B::required_service_classes(
            &device.facts,
            &device.intrinsics,
        ));
        let mut provided = BTreeSet::new();
        for definition in &parts.services {
            if definition.class.stable_name().is_empty() {
                panic!(
                    "ExecutionProfile<{:?}> contains an empty service identity",
                    B::NAME
                );
            }
            if !provided.insert(definition.class) {
                panic!(
                    "ExecutionProfile<{:?}> supplies service `{}` twice",
                    B::NAME,
                    definition.class.stable_name()
                );
            }
            assert!(
                !definition.correlation.stable_name().is_empty(),
                "ExecutionProfile<{:?}> service `{}` has an empty correlation identity",
                B::NAME,
                definition.class.stable_name()
            );
            assert!(
                definition.qualification.minimum_units != 0
                    && definition.qualification.minimum_units
                        <= definition.qualification.maximum_units
                    && definition.qualification.maximum_concurrent_uses != 0,
                "ExecutionProfile<{:?}> service `{}` has an empty qualification domain",
                B::NAME,
                definition.class.stable_name()
            );
            if definition.topology.resources == 0 || definition.topology.max_concurrency == 0 {
                panic!(
                    "ExecutionProfile<{:?}> service `{}` has an empty resource topology",
                    B::NAME,
                    definition.class.stable_name()
                );
            }
            if definition.saturated_capacity.regimes.is_empty() {
                panic!(
                    "ExecutionProfile<{:?}> service `{}` has no capacity regime",
                    B::NAME,
                    definition.class.stable_name()
                );
            }
            let mut previous = None;
            for (index, regime) in definition.saturated_capacity.regimes.iter().enumerate() {
                match (previous, regime.max_units) {
                    (_, None) if index + 1 == definition.saturated_capacity.regimes.len() => {}
                    (_, None) => panic!(
                        "ExecutionProfile<{:?}> service `{}` has a non-final unbounded regime",
                        B::NAME,
                        definition.class.stable_name()
                    ),
                    (Some(previous), Some(current)) if current <= previous => panic!(
                        "ExecutionProfile<{:?}> service `{}` has unordered regimes",
                        B::NAME,
                        definition.class.stable_name()
                    ),
                    (_, Some(current)) => previous = Some(current),
                }
            }
            if definition
                .saturated_capacity
                .regimes
                .last()
                .is_some_and(|regime| regime.max_units.is_some())
            {
                panic!(
                    "ExecutionProfile<{:?}> service `{}` lacks a final unbounded regime",
                    B::NAME,
                    definition.class.stable_name()
                );
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
                    panic!(
                        "ExecutionProfile<{:?}> service `{}` has an invalid duration interval",
                        B::NAME,
                        definition.class.stable_name()
                    );
                }
                if interval.upper_numerator != 0 {
                    let allowed_percent = match definition.accuracy {
                        ServiceAccuracyClass::Compute => 1u128,
                        ServiceAccuracyClass::MemoryOrTransfer => 2u128,
                    };
                    let width = u128::from(interval.upper_numerator - interval.lower_numerator);
                    let midpoint_twice =
                        u128::from(interval.upper_numerator) + u128::from(interval.lower_numerator);
                    assert!(
                        width * 100 <= midpoint_twice * allowed_percent,
                        "ExecutionProfile<{:?}> service `{}` interval {}/{}..{}/{} ns exceeds its {}% acquisition interval budget",
                        B::NAME,
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
                assert!(
                    !batches.is_empty(),
                    "ExecutionProfile<{:?}> service `{}` has no measurement batches",
                    B::NAME,
                    definition.class.stable_name()
                );
                for batch in batches {
                    assert!(
                        !batch.probe.is_empty()
                            && !batch.method.is_empty()
                            && batch.series.is_valid(),
                        "ExecutionProfile<{:?}> service `{}` has an incomplete measurement batch",
                        B::NAME,
                        definition.class.stable_name()
                    );
                    assert!(
                        batch.timer_resolution_ns.denominator != 0
                            && batch.timer_resolution_ns.lower_numerator
                                <= batch.timer_resolution_ns.upper_numerator,
                        "ExecutionProfile<{:?}> service `{}` has invalid timer resolution evidence",
                        B::NAME,
                        definition.class.stable_name()
                    );
                }
            }
        }
        assert_eq!(
            required,
            provided,
            "ExecutionProfile<{:?}> required/provided execution service sets differ",
            B::NAME
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

        fn scale_interval(interval: DurationInterval, units: u64) -> WideInterval {
            WideInterval {
                lower: u128::from(interval.lower_numerator)
                    .checked_mul(u128::from(units))
                    .expect("composition prediction lower bound overflows u128"),
                upper: u128::from(interval.upper_numerator)
                    .checked_mul(u128::from(units))
                    .expect("composition prediction upper bound overflows u128"),
                denominator: u128::from(interval.denominator),
            }
        }

        fn add_intervals(left: WideInterval, right: WideInterval) -> WideInterval {
            let common = gcd(left.denominator, right.denominator);
            let left_scale = right.denominator / common;
            let right_scale = left.denominator / common;
            WideInterval {
                lower: left
                    .lower
                    .checked_mul(left_scale)
                    .and_then(|value| {
                        right
                            .lower
                            .checked_mul(right_scale)
                            .and_then(|right| value.checked_add(right))
                    })
                    .expect("composition prediction lower sum overflows u128"),
                upper: left
                    .upper
                    .checked_mul(left_scale)
                    .and_then(|value| {
                        right
                            .upper
                            .checked_mul(right_scale)
                            .and_then(|right| value.checked_add(right))
                    })
                    .expect("composition prediction upper sum overflows u128"),
                denominator: left
                    .denominator
                    .checked_mul(left_scale)
                    .expect("composition prediction denominator overflows u128"),
            }
        }

        assert!(
            !parts.composition_qualification.cases.is_empty(),
            "ExecutionProfile<{:?}> has no held-out composition qualification cases",
            B::NAME
        );
        let mut composition_names = BTreeSet::new();
        let mut maximum_relative_error_basis_points = 0u16;
        for case in &parts.composition_qualification.cases {
            assert!(
                !case.stable_name.is_empty() && composition_names.insert(case.stable_name),
                "ExecutionProfile<{:?}> has an empty or duplicated composition case `{}`",
                B::NAME,
                case.stable_name
            );
            assert!(
                !case.demands.is_empty(),
                "ExecutionProfile<{:?}> composition case `{}` has no demands",
                B::NAME,
                case.stable_name
            );
            assert!(
                case.observed_ns.denominator != 0
                    && case.observed_ns.lower_numerator <= case.observed_ns.upper_numerator,
                "ExecutionProfile<{:?}> composition case `{}` has an invalid observation interval",
                B::NAME,
                case.stable_name
            );

            let mut predicted = WideInterval {
                lower: 0,
                upper: 0,
                denominator: 1,
            };
            for demand in &case.demands {
                assert!(
                    demand.units != 0,
                    "ExecutionProfile<{:?}> composition case `{}` has zero service demand",
                    B::NAME,
                    case.stable_name
                );
                let service = parts
                    .services
                    .iter()
                    .find(|service| service.class == demand.class)
                    .unwrap_or_else(|| {
                        panic!(
                            "ExecutionProfile<{:?}> composition case `{}` names absent service `{}`",
                            B::NAME,
                            case.stable_name,
                            demand.class.stable_name()
                        )
                    });
                let contribution = match demand.mode {
                    DemandMode::DependencyLatency => {
                        scale_interval(service.dependency_latency, demand.units)
                    }
                    DemandMode::SaturatedCapacity => {
                        let capacity = u64::from(service.topology.resources)
                            .checked_mul(u64::from(service.topology.max_concurrency))
                            .expect("composition service topology overflows u64");
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
                            .expect("validated service curve has no applicable final regime");
                        add_intervals(
                            scale_interval(service.saturated_capacity.setup, 1),
                            scale_interval(regime.per_unit, waves),
                        )
                    }
                };
                assert!(
                    demand.units >= service.qualification.minimum_units
                        && demand.units <= service.qualification.maximum_units,
                    "ExecutionProfile<{:?}> composition case `{}` is outside service `{}` qualification",
                    B::NAME,
                    case.stable_name,
                    demand.class.stable_name()
                );
                predicted = add_intervals(predicted, contribution);
            }

            let predicted_midpoint = predicted
                .lower
                .checked_add(predicted.upper)
                .expect("composition prediction midpoint overflows u128");
            let observed_midpoint = u128::from(case.observed_ns.lower_numerator)
                .checked_add(u128::from(case.observed_ns.upper_numerator))
                .expect("composition observation midpoint overflows u128");
            assert!(
                observed_midpoint != 0,
                "composition observation midpoint is zero"
            );
            let predicted_scaled = predicted_midpoint
                .checked_mul(u128::from(case.observed_ns.denominator))
                .expect("composition prediction comparison overflows u128");
            let observed_scaled = observed_midpoint
                .checked_mul(predicted.denominator)
                .expect("composition observation comparison overflows u128");
            let error = predicted_scaled
                .abs_diff(observed_scaled)
                .checked_mul(10_000)
                .expect("composition relative error overflows u128");
            let relative_error_basis_points = error.div_ceil(observed_scaled);
            assert!(
                relative_error_basis_points <= 500,
                "ExecutionProfile<{:?}> held-out composition `{}` predicted {}/{}..{}/{} ns versus observed {}/{}..{}/{} ns ({} bp) exceeds the 5% qualification budget",
                B::NAME,
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
        let mut execution = Sha256::new();
        execution.update(b"seismic-execution-profile-v2");
        execution.update(device.identity.fingerprint);
        execution.update(parts.probe_suite_revision.as_bytes());
        execution.update(parts.composition_qualification.suite_revision.as_bytes());
        execution.update(maximum_relative_error_basis_points.to_le_bytes());
        execution.update(parts.composition_qualification.observations_digest);
        for service in &parts.services {
            execution.update(service.class.stable_name().as_bytes());
            execution.update(service.correlation.stable_name().as_bytes());
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
                execution.update(regime.max_units.unwrap_or(u64::MAX).to_le_bytes());
            }
            execution.update(format!("{:?}", service.provenance).as_bytes());
            if let FactProvenance::Measured { batches } = &service.provenance {
                execution.update(b"measurement-batches/v1");
                execution.update((batches.len() as u64).to_le_bytes());
                for batch in batches {
                    execution.update(batch.probe.as_bytes());
                    execution.update(batch.method.as_bytes());
                    execution.update(batch.timer_resolution_ns.lower_numerator.to_le_bytes());
                    execution.update(batch.timer_resolution_ns.upper_numerator.to_le_bytes());
                    execution.update(batch.timer_resolution_ns.denominator.to_le_bytes());
                    execution.update(batch.observations_digest);
                    execution.update(batch.acquisition_duration_ns.to_le_bytes());
                    batch.series.update_identity(&mut execution);
                }
            }
        }
        let identity = ExecutionProfileIdentity {
            device: device.identity.clone(),
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
            _backend: std::marker::PhantomData,
        })
    }

    pub(super) fn bind_constants<B: Backend>(
        profile: &DeviceContract<B>,
        arena: &mut ExprArena,
    ) -> TargetConstants {
        fn nat(
            arena: &mut ExprArena,
            bindings: &mut Vec<(SymbolId, SymbolValue)>,
            value: u64,
        ) -> TargetConstantId {
            let (id, symbol) = arena.target_constant(SymbolSort::Nat);
            bindings.push((symbol, SymbolValue::Nat(value)));
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

    pub(super) enum DurationKernels<'a, B: Backend> {
        Draft(&'a [crate::kernel::Kernel<B>]),
        Closed(&'a crate::kernel::KernelArena<B>),
    }

    impl<'a, B: Backend> DurationKernels<'a, B> {
        fn kernel(&self, id: crate::kernel::KernelId) -> &'a crate::kernel::Kernel<B> {
            match self {
                Self::Draft(kernels) => kernels
                    .get(id.index() as usize)
                    .expect("launch kernel is outside the draft kernel set"),
                Self::Closed(kernels) => kernels.kernel(id),
            }
        }
    }

    pub(super) fn derive_duration<B: Backend>(
        device: &DeviceContract<B>,
        profile: &ExecutionProfile<B>,
        arena: &mut ExprArena,
        schedule: &crate::schedule::ParametricSchedule,
        launch_layouts: &[crate::storage::LaunchLocalLayout],
        kernels: DurationKernels<'_, B>,
        native_kernels: Option<&[std::sync::Arc<NativeKernel<B>>]>,
    ) -> QualifiedDuration {
        use seismic_lang::expr::{DurationExpr, DurationTerm};

        fn service_term<B: Backend>(
            profile: &ExecutionProfile<B>,
            arena: &mut ExprArena,
            class: ServiceClassId,
            units: seismic_lang::expr::NatExpr,
            mode: DemandMode,
            launch_ordinal: Option<u32>,
            qualification: &mut Vec<seismic_lang::expr::BoolExpr>,
            terms: &mut Vec<ServiceModelTerm>,
        ) -> DurationExpr {
            let definition = profile.service(class);
            let minimum = arena.nat(definition.qualification.minimum_units);
            let maximum = arena.nat(definition.qualification.maximum_units);
            qualification.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Ge, units, minimum));
            qualification.push(arena.nat_cmp(seismic_lang::expr::CmpOp::Le, units, maximum));
            terms.push(ServiceModelTerm {
                launch_ordinal,
                class,
                correlation: definition.correlation,
                units,
                mode,
            });
            let term = |arena: &mut ExprArena,
                        demand: seismic_lang::expr::NatExpr,
                        interval: DurationInterval| {
                let envelope = u64::from(
                    profile
                        .composition_qualification()
                        .maximum_relative_error_basis_points,
                );
                let denominator = interval
                    .denominator
                    .checked_mul(10_000)
                    .expect("duration composition-envelope denominator overflows u64");
                let lower_numerator = interval
                    .lower_numerator
                    .checked_mul(10_000 - envelope)
                    .expect("duration composition-envelope lower bound overflows u64");
                let upper_numerator = interval
                    .upper_numerator
                    .checked_mul(10_000 + envelope)
                    .expect("duration composition-envelope upper bound overflows u64");
                arena.duration(&[DurationTerm {
                    demand,
                    lower_numerator,
                    upper_numerator,
                    denominator,
                }])
            };
            match mode {
                DemandMode::DependencyLatency => term(arena, units, definition.dependency_latency),
                DemandMode::SaturatedCapacity => {
                    let one = arena.nat(1);
                    let setup = definition.saturated_capacity.setup;
                    let setup = term(arena, one, setup);
                    let mut regimes = definition.saturated_capacity.regimes.iter().rev();
                    let last = regimes
                        .next()
                        .expect("profile service capacity curve is non-empty");
                    let capacity = u64::from(definition.topology.resources)
                        .checked_mul(u64::from(definition.topology.max_concurrency))
                        .expect("service topology capacity overflows u64");
                    let capacity = arena.nat(capacity);
                    let waves = arena.nat_ceil_div(units, capacity);
                    let mut selected = term(arena, waves, last.per_unit);
                    for regime in regimes {
                        let maximum = regime
                            .max_units
                            .expect("only the final capacity regime is unbounded");
                        let maximum = arena.nat(maximum);
                        let condition =
                            arena.nat_cmp(seismic_lang::expr::CmpOp::Le, units, maximum);
                        let branch = term(arena, waves, regime.per_unit);
                        selected = arena.duration_select(condition, branch, selected);
                    }
                    arena.duration_add(setup, selected)
                }
            }
        }

        let one = arena.nat(1);
        let mut qualification = Vec::new();
        let mut terms = Vec::new();
        let mut launches = Vec::with_capacity(schedule.launches().len());
        assert_eq!(schedule.launches().len(), launch_layouts.len());
        for (launch_ordinal, (launch, local_layout)) in
            schedule.launches().iter().zip(launch_layouts).enumerate()
        {
            let kernel = kernels.kernel(launch.kernel);
            let native = native_kernels.map(|kernels| &*kernels[launch.kernel.index() as usize]);
            let emission = device.kernel_emission_layout(kernel);
            let grid = arena.nat_product(&launch.grid);
            let workgroup = arena.nat_product(&launch.workgroup);
            let participants = arena.nat_mul(grid, workgroup);
            let submission = service_term(
                profile,
                arena,
                SERVICE_SUBMISSION,
                one,
                DemandMode::DependencyLatency,
                Some(launch_ordinal as u32),
                &mut qualification,
                &mut terms,
            );
            let data = crate::kernel::internals::data(kernel);
            let mut duration = submission;
            for (block, multiplicity) in data.blocks().iter().zip(data.block_multiplicity()) {
                for op in &block.ops {
                    let closed = kernel.closed_op(op, &emission);
                    let demands = B::execution_demand(device, arena, kernel, &emission, closed);
                    for demand in demands {
                        let demands = match native {
                            Some(native) => B::refine_execution_demand(
                                device,
                                arena,
                                launch,
                                local_layout,
                                native,
                                demand,
                            ),
                            None => vec![demand],
                        };
                        for demand in demands {
                            let mut units = match demand.scope {
                                DemandScope::PerParticipant => {
                                    arena.nat_mul(demand.units, participants)
                                }
                                DemandScope::PerLaunch => demand.units,
                            };
                            if let Some(multiplicity) = multiplicity {
                                units = arena.nat_mul(units, *multiplicity);
                            }
                            let term = service_term(
                                profile,
                                arena,
                                demand.class,
                                units,
                                demand.mode,
                                Some(launch_ordinal as u32),
                                &mut qualification,
                                &mut terms,
                            );
                            duration = arena.duration_add(duration, term);
                        }
                    }
                }
            }
            launches.push(duration);
        }

        fn sequence<B: Backend>(
            profile: &ExecutionProfile<B>,
            arena: &mut ExprArena,
            launches: &[DurationExpr],
            steps: &[crate::schedule::ScheduleStep],
            qualification: &mut Vec<seismic_lang::expr::BoolExpr>,
            terms: &mut Vec<ServiceModelTerm>,
        ) -> DurationExpr {
            let one = arena.nat(1);
            let mut duration = arena.duration(&[]);
            for step in steps {
                let item = match step {
                    crate::schedule::ScheduleStep::Launch(id) => launches[id.index() as usize],
                    crate::schedule::ScheduleStep::Copy(copy) => service_term(
                        profile,
                        arena,
                        SERVICE_COPY,
                        copy.bytes,
                        DemandMode::SaturatedCapacity,
                        None,
                        qualification,
                        terms,
                    ),
                    crate::schedule::ScheduleStep::Fill(fill) => service_term(
                        profile,
                        arena,
                        SERVICE_FILL,
                        fill.bytes,
                        DemandMode::SaturatedCapacity,
                        None,
                        qualification,
                        terms,
                    ),
                    crate::schedule::ScheduleStep::ScalarRead(_) => service_term(
                        profile,
                        arena,
                        SERVICE_SCALAR_READ,
                        one,
                        DemandMode::DependencyLatency,
                        None,
                        qualification,
                        terms,
                    ),
                    crate::schedule::ScheduleStep::ScalarMove(_) => service_term(
                        profile,
                        arena,
                        SERVICE_SCALAR_MOVE,
                        one,
                        DemandMode::DependencyLatency,
                        None,
                        qualification,
                        terms,
                    ),
                    crate::schedule::ScheduleStep::Check(_) => service_term(
                        profile,
                        arena,
                        SERVICE_DATA_CHECK,
                        one,
                        DemandMode::DependencyLatency,
                        None,
                        qualification,
                        terms,
                    ),
                    crate::schedule::ScheduleStep::If {
                        condition,
                        then_steps,
                        else_steps,
                    } => {
                        let then_duration =
                            sequence(profile, arena, launches, then_steps, qualification, terms);
                        let else_duration =
                            sequence(profile, arena, launches, else_steps, qualification, terms);
                        arena.duration_select(*condition, then_duration, else_duration)
                    }
                    crate::schedule::ScheduleStep::Repeat {
                        binder,
                        symbol,
                        start,
                        end,
                        body,
                    } => {
                        let mut body_qualification = Vec::new();
                        let mut body_terms = Vec::new();
                        let body = sequence(
                            profile,
                            arena,
                            launches,
                            body,
                            &mut body_qualification,
                            &mut body_terms,
                        );
                        assert_eq!(body_qualification.len(), body_terms.len() * 2);
                        let upper = arena.nat_max(*end, *start);
                        let trips = arena.nat_sub(upper, *start);
                        let binder_dependent = arena
                            .free_symbols(seismic_lang::expr::AnyExpr::Duration(body))
                            .contains(symbol);
                        if binder_dependent {
                            let zero = arena.nat(0);
                            let one = arena.nat(1);
                            for condition in body_qualification {
                                let violation = arena.nat_select(condition, zero, one);
                                let violations = arena.nat_fold_range(
                                    seismic_lang::expr::FoldOp::Max,
                                    *binder,
                                    *start,
                                    trips,
                                    violation,
                                );
                                qualification.push(arena.nat_cmp(
                                    seismic_lang::expr::CmpOp::Eq,
                                    violations,
                                    zero,
                                ));
                            }
                            terms.extend(body_terms.into_iter().map(|mut term| {
                                term.units = arena.nat_fold_range(
                                    seismic_lang::expr::FoldOp::Sum,
                                    *binder,
                                    *start,
                                    trips,
                                    term.units,
                                );
                                term
                            }));
                            arena.duration_sum_range(*binder, *start, trips, body)
                        } else {
                            qualification.extend(body_qualification);
                            terms.extend(body_terms.into_iter().map(|mut term| {
                                term.units = arena.nat_mul(term.units, trips);
                                term
                            }));
                            arena.duration_scale(body, trips)
                        }
                    }
                    crate::schedule::ScheduleStep::Choose { decision, options } => {
                        let value = arena.decision_value(*decision);
                        let mut options = options.iter().rev();
                        let (_, last) = options
                            .next()
                            .expect("closed schedule choice has no options");
                        let mut selected =
                            sequence(profile, arena, launches, last, qualification, terms);
                        for (expected, branch) in options {
                            let expected = arena.int(*expected);
                            let condition =
                                arena.int_cmp(seismic_lang::expr::CmpOp::Eq, value, expected);
                            let branch =
                                sequence(profile, arena, launches, branch, qualification, terms);
                            selected = arena.duration_select(condition, branch, selected);
                        }
                        selected
                    }
                };
                duration = arena.duration_add(duration, item);
            }
            duration
        }

        let duration = sequence(
            profile,
            arena,
            &launches,
            schedule.steps(),
            &mut qualification,
            &mut terms,
        );
        QualifiedDuration {
            duration,
            qualification,
            terms,
        }
    }
}
