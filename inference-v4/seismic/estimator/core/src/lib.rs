//! Backend-independent execution-model facts and symbolic service algebra.
//! This crate performs no device discovery, compilation, or execution.

use seismic_lang::expr::ExprArena;
use sha2::{Digest, Sha256};

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

crate::analytical_services! {
    pub enum CoreService {
        Submission => "core.submission",
        Copy => "core.copy",
        Fill => "core.fill",
        ScalarRead => "core.scalar-read",
        ScalarMove => "core.scalar-move",
        AllocationInstance => "core.allocation-instance",
        TensorPublication => "core.tensor-publication",
        DataCheck => "core.data-check",
    }
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

    pub fn is_valid(&self) -> bool {
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

    pub fn update_identity(&self, digest: &mut Sha256) {
        fn tag(digest: &mut Sha256, value: &'static [u8]) {
            digest.update((value.len() as u64).to_le_bytes());
            digest.update(value);
        }
        match self {
            Self::Single(series) => {
                tag(digest, b"measurement-series/single/v1");
                digest.update(series.workload_units.to_le_bytes());
                digest.update((series.observations_ns.len() as u64).to_le_bytes());
                for observation in &series.observations_ns {
                    digest.update(observation.to_le_bytes());
                }
            }
            Self::PairedAdjacent(series) => {
                tag(digest, b"measurement-series/paired-adjacent/v1");
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

impl MeasurementBatch {
    pub fn update_identity(&self, digest: &mut Sha256) {
        update_text(digest, b"probe", self.probe);
        update_text(digest, b"method", self.method);
        update_interval(digest, b"timer-resolution", self.timer_resolution_ns);
        digest.update(self.observations_digest);
        digest.update(self.acquisition_duration_ns.to_le_bytes());
        self.series.update_identity(digest);
    }

    pub fn identity(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        update_tag(&mut digest, b"measurement-batch/v1");
        self.update_identity(&mut digest);
        digest.finalize().into()
    }
}

impl FactProvenance {
    /// Canonical, exhaustively tagged identity encoding. Adding a provenance
    /// variant requires extending this match rather than inheriting Debug text.
    pub fn update_identity(&self, digest: &mut Sha256) {
        match self {
            Self::Queried { api, field } => {
                update_tag(digest, b"fact-provenance/queried/v1");
                update_text(digest, b"api", api);
                update_text(digest, b"field", field);
            }
            Self::Derived { rule, inputs } => {
                update_tag(digest, b"fact-provenance/derived/v1");
                update_text(digest, b"rule", rule);
                digest.update((inputs.len() as u64).to_le_bytes());
                for input in inputs {
                    update_text(digest, b"input", input);
                }
            }
            Self::Measured { batches } => {
                update_tag(digest, b"fact-provenance/measured/v1");
                digest.update((batches.len() as u64).to_le_bytes());
                for batch in batches {
                    batch.update_identity(digest);
                }
            }
        }
    }
}

fn update_tag(digest: &mut Sha256, tag: &'static [u8]) {
    digest.update((tag.len() as u64).to_le_bytes());
    digest.update(tag);
}

fn update_text(digest: &mut Sha256, tag: &'static [u8], value: &str) {
    update_tag(digest, tag);
    digest.update((value.len() as u64).to_le_bytes());
    digest.update(value.as_bytes());
}

fn update_interval(digest: &mut Sha256, tag: &'static [u8], value: DurationInterval) {
    update_tag(digest, tag);
    digest.update(value.lower_numerator.to_le_bytes());
    digest.update(value.upper_numerator.to_le_bytes());
    digest.update(value.denominator.to_le_bytes());
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
pub struct ExecutionDemand<S = ServiceClassId> {
    pub class: S,
    pub units: seismic_lang::expr::NatExpr,
    pub mode: DemandMode,
    pub scope: DemandScope,
}

/// A structurally non-empty set of costs owned by one closed operation.
///
/// Backends cannot use an empty `Vec` to make an operation disappear.  An
/// operation that emits no runtime work must instead return an explicit
/// [`OperationCost::Elided`] witness.
#[derive(Clone, Debug)]
pub struct OperationDemands<S = ServiceClassId> {
    first: ExecutionDemand<S>,
    rest: Vec<ExecutionDemand<S>>,
}

impl<S> OperationDemands<S> {
    pub fn one(first: ExecutionDemand<S>) -> Self {
        Self {
            first,
            rest: Vec::new(),
        }
    }

    pub fn with_rest(first: ExecutionDemand<S>, rest: Vec<ExecutionDemand<S>>) -> Self {
        Self { first, rest }
    }

    pub fn push(&mut self, demand: ExecutionDemand<S>) {
        self.rest.push(demand);
    }

    pub fn iter(&self) -> impl Iterator<Item = &ExecutionDemand<S>> {
        std::iter::once(&self.first).chain(&self.rest)
    }

    pub fn into_iter(self) -> impl Iterator<Item = ExecutionDemand<S>> {
        std::iter::once(self.first).chain(self.rest)
    }
}

/// Closed reasons why a visited operation has no runtime cost.  Adding a new
/// reason is an explicit semantic decision, rather than an accidental empty
/// demand list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProvenElision {
    CompileTimeOnly,
}

/// A model cannot price an actual operation from its available semantic facts.
/// This does not reject the executable or invent a zero-cost contribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelLimitation {
    DeviceExtent { value: seismic_ir::kernel::ops::ErasedValue },
    HostQuantityWidth,
}

/// Total cost transfer for one closed operation.
#[derive(Clone, Debug)]
pub enum OperationCost<S = ServiceClassId> {
    Demands(OperationDemands<S>),
    Elided(ProvenElision),
}

impl<S> OperationCost<S> {
    pub fn one(demand: ExecutionDemand<S>) -> Self {
        Self::Demands(OperationDemands::one(demand))
    }

    pub fn demands(first: ExecutionDemand<S>, rest: Vec<ExecutionDemand<S>>) -> Self {
        Self::Demands(OperationDemands::with_rest(first, rest))
    }

    pub fn map_services<U>(self, mut map: impl FnMut(S) -> U) -> OperationCost<U> {
        match self {
            Self::Demands(demands) => {
                let mut demands = demands.into_iter().map(|demand| ExecutionDemand {
                    class: map(demand.class),
                    units: demand.units,
                    mode: demand.mode,
                    scope: demand.scope,
                });
                let first = demands
                    .next()
                    .expect("OperationDemands is structurally non-empty");
                OperationCost::demands(first, demands.collect())
            }
            Self::Elided(reason) => OperationCost::Elided(reason),
        }
    }

    pub fn try_map_services<U, E>(
        self,
        mut map: impl FnMut(S) -> Result<U, E>,
    ) -> Result<OperationCost<U>, E> {
        match self {
            Self::Demands(demands) => {
                let mut mapped = Vec::new();
                for demand in demands.into_iter() {
                    mapped.push(ExecutionDemand {
                        class: map(demand.class)?,
                        units: demand.units,
                        mode: demand.mode,
                        scope: demand.scope,
                    });
                }
                let first = mapped.remove(0);
                Ok(OperationCost::demands(first, mapped))
            }
            Self::Elided(reason) => Ok(OperationCost::Elided(reason)),
        }
    }
}

/// The invocation whose work produced a modeled contribution. This is
/// semantic provenance: it never contains a native kernel or artifact handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InvocationProvenance {
    KernelLaunch { ordinal: u32 },
    HostSchedule,
}

/// Symbolic region in which a modeled contribution applies. Guards are the
/// enclosing schedule choices, ordered outermost first. Candidate-family
/// ownership remains with the evaluator that requested this assessment.
#[derive(Clone, Debug)]
pub struct ContributionRegion {
    invocation: InvocationProvenance,
    guards: Vec<seismic_lang::expr::BoolExpr>,
}

impl ContributionRegion {
    pub fn invocation(&self) -> InvocationProvenance {
        self.invocation
    }
    pub fn guards(&self) -> &[seismic_lang::expr::BoolExpr] {
        &self.guards
    }

    /// The exact predicate under which this region is active. An unguarded
    /// region is active everywhere. Repeat composition replaces binder-local
    /// guards with an `ever active` predicate before exposing the assessment.
    pub fn active_predicate(&self, arena: &mut ExprArena) -> seismic_lang::expr::BoolExpr {
        arena.all(&self.guards)
    }
}

/// One indivisible piece of analytical evidence. Demand identity,
/// provenance, and its duration expression cannot be paired with data from
/// another contribution by position.
#[derive(Clone, Debug)]
pub struct ModeledContribution {
    region: ContributionRegion,
    class: ServiceClassId,
    correlation: ServiceCorrelationId,
    units: seismic_lang::expr::NatExpr,
    mode: DemandMode,
    duration: seismic_lang::expr::DurationExpr,
    evidence: FactProvenance,
}

impl ModeledContribution {
    pub fn region(&self) -> &ContributionRegion {
        &self.region
    }
    pub fn class(&self) -> ServiceClassId {
        self.class
    }
    pub fn correlation(&self) -> ServiceCorrelationId {
        self.correlation
    }
    pub fn units(&self) -> seismic_lang::expr::NatExpr {
        self.units
    }
    pub fn mode(&self) -> DemandMode {
        self.mode
    }
    pub fn duration(&self) -> seismic_lang::expr::DurationExpr {
        self.duration
    }
    pub fn evidence(&self) -> &FactProvenance {
        &self.evidence
    }
}

/// Total analytical result for one authoritative executable IR.  There is no
/// partial-success form: every visited operation and schedule step has a
/// modeled cost or an explicit semantic elision before this value exists.
#[derive(Clone, Debug)]
pub struct TotalPerformanceModel {
    estimate: seismic_lang::expr::DurationExpr,
    contributions: Vec<ModeledContribution>,
    maximum_relative_error_basis_points: u16,
}

impl TotalPerformanceModel {
    pub fn estimate(&self) -> seismic_lang::expr::DurationExpr {
        self.estimate
    }
    pub fn contributions(&self) -> &[ModeledContribution] {
        &self.contributions
    }
    pub fn maximum_relative_error_basis_points(&self) -> u16 {
        self.maximum_relative_error_basis_points
    }
}

/// Read-only coefficients used by the symbolic service model.
/// The profile owner establishes the validity and provenance of these facts.
pub trait ServiceModel {
    /// Infallible selection from a profile whose acquisition/assembly already
    /// proved that every service required by its closed operation vocabulary
    /// is present.
    fn service(&self, service: ServiceClassId) -> &ServiceDefinition;
    fn maximum_relative_error_basis_points(&self) -> u16;
}

use seismic_lang::expr::DurationTerm;

pub fn service_contribution<M: ServiceModel + ?Sized>(
    profile: &M,
    arena: &mut ExprArena,
    service: ServiceClassId,
    units: seismic_lang::expr::NatExpr,
    mode: DemandMode,
    invocation: InvocationProvenance,
) -> ModeledContribution {
    let region = ContributionRegion {
        invocation,
        guards: Vec::new(),
    };
    let definition = profile.service(service);
    let term =
        |arena: &mut ExprArena, demand: seismic_lang::expr::NatExpr, interval: DurationInterval| {
            let envelope = u64::from(profile.maximum_relative_error_basis_points());
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
    let duration = match mode {
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
                let condition = arena.nat_cmp(seismic_lang::expr::CmpOp::Le, units, maximum);
                let branch = term(arena, waves, regime.per_unit);
                selected = arena.duration_select(condition, branch, selected);
            }
            arena.duration_add(setup, selected)
        }
    };
    ModeledContribution {
        region,
        class: definition.class,
        correlation: definition.correlation,
        units,
        mode,
        duration,
        evidence: definition.provenance.clone(),
    }
}

mod execution;
pub use execution::{
    estimate, AnalyticalModelDefinition, AnalyticalService, AnalyticalServiceState, ExecutionModel,
};
