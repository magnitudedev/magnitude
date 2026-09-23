//! Pure proof of the Metal characterization boundary.
//!
//! This crate has no Metal, compiler, candidate, solver, or estimator
//! dependency. It snapshots the current estimator fact algebra, maps every
//! fact exhaustively to an acquisition obligation, validates an aggregate raw
//! bundle, and leaves physical interpretation to the exact model that declares
//! those obligations. It cannot manufacture a physical profile.

use std::fmt;

pub const PROTOCOL_REVISION: &str = "seismic-metal-characterization-boundary-v0";
pub const CURRENT_FACT_ALGEBRA_REVISION: &str = "seismic-current-service-algebra-2026-09-21";

/// Exact union of estimator/core's six service classes and
/// estimator/metal's 24 service classes at the audited snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MetalCostFact {
    Submission,
    Copy,
    Fill,
    ScalarRead,
    ScalarMove,
    DataCheck,
    Control,
    Integer,
    F32AddSub,
    F32Multiply,
    F32Divide,
    F32Remainder,
    F32MinMax,
    F32Fma,
    F32Compare,
    F32ToInteger,
    IntegerToF32,
    F32ToF16,
    F16ToF32,
    F32ToBF16,
    F16Strict,
    BF16Strict,
    ApproximateMath,
    GlobalMemory,
    WorkgroupMemory,
    Representation,
    Atomic,
    Barrier,
    Subgroup,
    Matrix,
}

impl MetalCostFact {
    pub const COUNT: usize = 30;
    pub const ALL: [Self; Self::COUNT] = [
        Self::Submission,
        Self::Copy,
        Self::Fill,
        Self::ScalarRead,
        Self::ScalarMove,
        Self::DataCheck,
        Self::Control,
        Self::Integer,
        Self::F32AddSub,
        Self::F32Multiply,
        Self::F32Divide,
        Self::F32Remainder,
        Self::F32MinMax,
        Self::F32Fma,
        Self::F32Compare,
        Self::F32ToInteger,
        Self::IntegerToF32,
        Self::F32ToF16,
        Self::F16ToF32,
        Self::F32ToBF16,
        Self::F16Strict,
        Self::BF16Strict,
        Self::ApproximateMath,
        Self::GlobalMemory,
        Self::WorkgroupMemory,
        Self::Representation,
        Self::Atomic,
        Self::Barrier,
        Self::Subgroup,
        Self::Matrix,
    ];

    pub const fn stable_name(self) -> &'static str {
        match self {
            Self::Submission => "core.submission",
            Self::Copy => "core.copy",
            Self::Fill => "core.fill",
            Self::ScalarRead => "core.scalar-read",
            Self::ScalarMove => "core.scalar-move",
            Self::DataCheck => "core.data-check",
            Self::Control => "metal.control",
            Self::Integer => "metal.integer",
            Self::F32AddSub => "metal.f32.add-sub",
            Self::F32Multiply => "metal.f32.multiply",
            Self::F32Divide => "metal.f32.divide",
            Self::F32Remainder => "metal.f32.remainder",
            Self::F32MinMax => "metal.f32.min-max",
            Self::F32Fma => "metal.f32.fma",
            Self::F32Compare => "metal.f32.compare",
            Self::F32ToInteger => "metal.f32.to-integer",
            Self::IntegerToF32 => "metal.integer.to-f32",
            Self::F32ToF16 => "metal.f32.to-f16",
            Self::F16ToF32 => "metal.f16.to-f32",
            Self::F32ToBF16 => "metal.f32.to-bf16",
            Self::F16Strict => "metal.f16.strict",
            Self::BF16Strict => "metal.bf16.strict",
            Self::ApproximateMath => "metal.approximate-math",
            Self::GlobalMemory => "metal.global-memory",
            Self::WorkgroupMemory => "metal.workgroup-memory",
            Self::Representation => "metal.representation",
            Self::Atomic => "metal.atomic",
            Self::Barrier => "metal.barrier",
            Self::Subgroup => "metal.subgroup",
            Self::Matrix => "metal.simdgroup-matrix",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Experiment {
    HostLifecycleBatch,
    HostTransferSweep,
    WorkgroupScopeSweep,
    MatrixCombinationSweep,
}

/// `NotObservable` is a pre-acquisition construction failure. It can never
/// appear in a successful or partial profile because no profile is created.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcquisitionObligation {
    DirectlyQueryable {
        authority: &'static str,
    },
    IsolatedMeasurable {
        entry_point: &'static str,
        experiment: Experiment,
        semantic_isolation: &'static str,
    },
    Derivable {
        exact_inputs: &'static [MetalCostFact],
    },
    NotObservable {
        blocker: &'static str,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FactProtocol {
    pub fact: MetalCostFact,
    pub obligation: AcquisitionObligation,
}

/// Exhaustive against `MetalCostFact`; deliberately has no default arm.
pub const fn protocol_for(fact: MetalCostFact) -> FactProtocol {
    use AcquisitionObligation as O;
    use Experiment as E;
    use MetalCostFact as F;
    let obligation = match fact {
        F::Submission => O::IsolatedMeasurable { entry_point: "characterize_submission", experiment: E::HostLifecycleBatch, semantic_isolation: "one queue and one empty command-buffer completion boundary" },
        F::Copy => O::IsolatedMeasurable { entry_point: "characterize_copy", experiment: E::HostTransferSweep, semantic_isolation: "declared bytes and storage state; completion charged separately" },
        F::Fill => O::IsolatedMeasurable { entry_point: "characterize_fill", experiment: E::HostTransferSweep, semantic_isolation: "declared bytes and storage state; completion charged separately" },
        F::ScalarRead => O::NotObservable { blocker: "current fact merges visibility, transfer, and possible completion; lifecycle owner is not singular" },
        F::ScalarMove => O::IsolatedMeasurable { entry_point: "characterize_host_scalar_move", experiment: E::HostLifecycleBatch, semantic_isolation: "host-only scalar operation with no queue transition" },
        F::DataCheck => O::IsolatedMeasurable { entry_point: "characterize_host_data_check", experiment: E::HostLifecycleBatch, semantic_isolation: "host predicate with operands resident" },
        F::Control => O::NotObservable { blocker: "one class conflates constants, parameters, branches, loops, selects, and renderer-added control" },
        F::Integer => O::NotObservable { blocker: "one class conflates 32/64-bit add, multiply, divide, shift, bit, compare, and helper work" },
        F::F32AddSub | F::F32Multiply | F::F32Divide | F::F32Remainder
        | F::F32MinMax | F::F32Fma | F::F16Strict | F::BF16Strict => O::NotObservable { blocker: "opaque strict helper has data-dependent integer/control/state-machine expansion hidden after tuning" },
        F::F32Compare => O::NotObservable { blocker: "Boolean result requires feedback scaffolding and strict comparison paths are hidden" },
        F::F32ToInteger | F::IntegerToF32 | F::F32ToF16 | F::F16ToF32
        | F::F32ToBF16 => O::NotObservable { blocker: "type-changing chain requires reverse transition; four-command residual is not isolated" },
        F::ApproximateMath => O::NotObservable { blocker: "one class conflates operation/dtype combinations and conversion sequences" },
        F::GlobalMemory => O::NotObservable { blocker: "resident/spill, reuse, coalescing, read/write, concurrency, and transitions are absent" },
        F::WorkgroupMemory => O::NotObservable { blocker: "bank pattern, access width, participants, and concurrency regimes are absent" },
        F::Representation => O::NotObservable { blocker: "recipe-specific loads, bit work, selects, conversions, and strict operations are hidden" },
        F::Atomic => O::NotObservable { blocker: "operation, dtype, contention, and CAS retry regimes are absent" },
        F::Barrier => O::IsolatedMeasurable { entry_point: "characterize_barrier", experiment: E::WorkgroupScopeSweep, semantic_isolation: "scope, participants, and outstanding-memory state are axes" },
        F::Subgroup => O::NotObservable { blocker: "one class conflates lane-index, shuffle, reductions, dtype, width, and participants" },
        F::Matrix => O::IsolatedMeasurable { entry_point: "characterize_matrix", experiment: E::MatrixCombinationSweep, semantic_isolation: "queried-supported exact dtype combination and tile shape" },
    };
    FactProtocol { fact, obligation }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompleteProbeManifest {
    entries: [FactProtocol; MetalCostFact::COUNT],
}

impl CompleteProbeManifest {
    pub fn current_estimator() -> Self {
        Self {
            entries: MetalCostFact::ALL.map(protocol_for),
        }
    }
    pub fn entries(&self) -> &[FactProtocol; MetalCostFact::COUNT] {
        &self.entries
    }
    pub fn constructibility_errors(&self) -> Vec<ConstructibilityError> {
        self.entries
            .iter()
            .filter_map(|entry| match entry.obligation {
                AcquisitionObligation::NotObservable { blocker } => Some(ConstructibilityError {
                    fact: entry.fact,
                    blocker,
                }),
                _ => None,
            })
            .collect()
    }
    pub fn is_constructible(&self) -> bool {
        self.constructibility_errors().is_empty()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstructibilityError {
    pub fact: MetalCostFact,
    pub blocker: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceIdentity {
    pub registry_id: u64,
    pub architecture: String,
    pub operating_system: String,
    pub metal_language: String,
    pub compiler_identity: String,
    pub numerical_mode: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolIdentity {
    pub protocol_revision: &'static str,
    pub fact_algebra_revision: &'static str,
    pub probe_library_digest: [u8; 32],
    pub measurement_endpoint: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcquisitionBudget {
    pub compile_ns: u64,
    pub warmup_ns: u64,
    pub measurement_ns: u64,
    pub total_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CharacterizationProtocol {
    identity: ProtocolIdentity,
    budget: AcquisitionBudget,
    manifest: CompleteProbeManifest,
}

impl CharacterizationProtocol {
    pub fn new(
        probe_library_digest: [u8; 32],
        budget: AcquisitionBudget,
    ) -> Result<Self, Vec<ConstructibilityError>> {
        let manifest = CompleteProbeManifest::current_estimator();
        let errors = manifest.constructibility_errors();
        if !errors.is_empty() {
            return Err(errors);
        }
        Ok(Self {
            identity: ProtocolIdentity {
                protocol_revision: PROTOCOL_REVISION,
                fact_algebra_revision: CURRENT_FACT_ALGEBRA_REVISION,
                probe_library_digest,
                measurement_endpoint: "MTLCommandBuffer.GPUStartTime/GPUEndTime",
            },
            budget,
            manifest,
        })
    }
    pub fn identity(&self) -> &ProtocolIdentity {
        &self.identity
    }
    pub fn budget(&self) -> AcquisitionBudget {
        self.budget
    }
    pub fn manifest(&self) -> &CompleteProbeManifest {
        &self.manifest
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcquisitionTelemetry {
    pub compile_ns: u64,
    pub warmup_ns: u64,
    pub measurement_ns: u64,
    pub total_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleSeries {
    pub endpoint: &'static str,
    pub workload: Vec<u64>,
    pub observations_ns: Vec<Vec<u64>>,
    pub timer_resolution_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeFailure {
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RawFactCapture {
    Queried {
        encoded_value: Vec<u8>,
        authority: &'static str,
    },
    Measured(SampleSeries),
    Derived,
    Failed(ProbeFailure),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawFactObservation {
    pub fact: MetalCostFact,
    pub capture: RawFactCapture,
}

/// Aggregate evidence only: no coefficient, curve, boundary, or estimator conclusion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawObservationBundle {
    pub device: DeviceIdentity,
    pub protocol: ProtocolIdentity,
    pub telemetry: AcquisitionTelemetry,
    pub facts: Vec<RawFactObservation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructuralValidationErrors(pub Vec<String>);
impl fmt::Display for StructuralValidationErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.join("; "))
    }
}

/// Valid evidence is still not a physical model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedRawBundle(RawObservationBundle);

pub fn validate_raw_bundle(
    protocol: &CharacterizationProtocol,
    expected_device: &DeviceIdentity,
    raw: RawObservationBundle,
) -> Result<ValidatedRawBundle, StructuralValidationErrors> {
    let mut errors = Vec::new();
    if &raw.device != expected_device {
        errors.push("device identity mismatch".into());
    }
    if &raw.protocol != protocol.identity() {
        errors.push("protocol identity mismatch".into());
    }
    let b = protocol.budget();
    if raw.telemetry.compile_ns > b.compile_ns {
        errors.push("probe-library compilation exceeded budget".into());
    }
    if raw.telemetry.warmup_ns > b.warmup_ns {
        errors.push("warmup exceeded budget".into());
    }
    if raw.telemetry.measurement_ns > b.measurement_ns {
        errors.push("measurement exceeded budget".into());
    }
    if raw.telemetry.total_ns > b.total_ns {
        errors.push("total acquisition exceeded budget".into());
    }
    let accounted = raw
        .telemetry
        .compile_ns
        .saturating_add(raw.telemetry.warmup_ns)
        .saturating_add(raw.telemetry.measurement_ns);
    if accounted > raw.telemetry.total_ns {
        errors.push("acquisition components exceed recorded total".into());
    }
    if raw.facts.len() != MetalCostFact::COUNT {
        errors.push(format!(
            "raw bundle has {} facts; expected {}",
            raw.facts.len(),
            MetalCostFact::COUNT
        ));
    }
    for (index, expected) in MetalCostFact::ALL.iter().enumerate() {
        let obligation = &protocol.manifest().entries()[index].obligation;
        match raw.facts.get(index) {
            Some(observed) if observed.fact == *expected => match (obligation, &observed.capture) {
                (
                    AcquisitionObligation::DirectlyQueryable { authority },
                    RawFactCapture::Queried {
                        encoded_value,
                        authority: observed_authority,
                    },
                ) => {
                    if encoded_value.is_empty() {
                        errors.push(format!(
                            "{}: queried value is empty",
                            expected.stable_name()
                        ));
                    }
                    if authority != observed_authority {
                        errors.push(format!(
                            "{}: query authority mismatch",
                            expected.stable_name()
                        ));
                    }
                }
                (
                    AcquisitionObligation::IsolatedMeasurable { .. },
                    RawFactCapture::Measured(series),
                ) => validate_series(*expected, series, &mut errors),
                (AcquisitionObligation::Derivable { .. }, RawFactCapture::Derived) => {}
                (AcquisitionObligation::NotObservable { blocker }, _) => errors.push(format!(
                    "{}: protocol is not constructible: {blocker}",
                    expected.stable_name()
                )),
                (_, RawFactCapture::Failed(failure)) => {
                    errors.push(format!("{}: {}", expected.stable_name(), failure.reason))
                }
                _ => errors.push(format!(
                    "{}: raw capture kind does not match acquisition obligation",
                    expected.stable_name()
                )),
            },
            Some(observed) => errors.push(format!(
                "slot {index} is {}; expected {}",
                observed.fact.stable_name(),
                expected.stable_name()
            )),
            None => errors.push(format!("missing {}", expected.stable_name())),
        }
    }
    if errors.is_empty() {
        Ok(ValidatedRawBundle(raw))
    } else {
        Err(StructuralValidationErrors(errors))
    }
}

fn validate_series(fact: MetalCostFact, series: &SampleSeries, errors: &mut Vec<String>) {
    let name = fact.stable_name();
    if series.timer_resolution_ns == 0 {
        errors.push(format!("{name}: timer resolution is zero"));
    }
    if series.workload.is_empty() {
        errors.push(format!("{name}: workload series is empty"));
    }
    if series.workload.len() != series.observations_ns.len() {
        errors.push(format!("{name}: workload/observation cardinality mismatch"));
    }
    if series.workload.contains(&0) {
        errors.push(format!("{name}: workload contains zero units"));
    }
    if series.observations_ns.iter().any(Vec::is_empty) {
        errors.push(format!("{name}: an observation set is empty"));
    }
}

/// A model-specific interpreter must define the typed profile and prove total
/// interpretation for every fact and invocation regime. There is intentionally
/// no generic curve fitter or generic `CertifiedMetalProfile` constructor.
pub trait MetalPhysicalModelInterpreter {
    type CertifiedMetalProfile;
    type Error;
    fn interpret(
        &self,
        evidence: ValidatedRawBundle,
    ) -> Result<Self::CertifiedMetalProfile, Self::Error>;
}

pub fn replay<I: MetalPhysicalModelInterpreter>(
    interpreter: &I,
    protocol: &CharacterizationProtocol,
    expected_device: &DeviceIdentity,
    raw: RawObservationBundle,
) -> Result<I::CertifiedMetalProfile, ReplayError<I::Error>> {
    let evidence =
        validate_raw_bundle(protocol, expected_device, raw).map_err(ReplayError::Structural)?;
    interpreter
        .interpret(evidence)
        .map_err(ReplayError::Physical)
}

#[derive(Debug, PartialEq, Eq)]
pub enum ReplayError<E> {
    Structural(StructuralValidationErrors),
    Physical(E),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_estimator_is_not_constructible() {
        let manifest = CompleteProbeManifest::current_estimator();
        assert_eq!(manifest.entries().len(), MetalCostFact::COUNT);
        assert!(!manifest.is_constructible());
        let blocked = manifest.constructibility_errors();
        assert!(blocked.iter().any(|e| e.fact == MetalCostFact::F32Compare));
        assert!(blocked
            .iter()
            .any(|e| e.fact == MetalCostFact::GlobalMemory));
        assert!(blocked.iter().any(|e| e.fact == MetalCostFact::Atomic));
    }

    #[test]
    fn protocol_creation_fails_before_hardware() {
        let budget = AcquisitionBudget {
            compile_ns: 1,
            warmup_ns: 1,
            measurement_ns: 1,
            total_ns: 3,
        };
        assert!(CharacterizationProtocol::new([0; 32], budget).is_err());
    }

    #[test]
    fn protocol_mapping_is_exhaustive_and_ordered() {
        let manifest = CompleteProbeManifest::current_estimator();
        for (fact, entry) in MetalCostFact::ALL.iter().zip(manifest.entries()) {
            assert_eq!(*fact, entry.fact);
        }
    }

    fn validation_fixture_protocol() -> CharacterizationProtocol {
        let entries = MetalCostFact::ALL.map(|fact| FactProtocol {
            fact,
            obligation: AcquisitionObligation::IsolatedMeasurable {
                entry_point: "fixture",
                experiment: Experiment::HostLifecycleBatch,
                semantic_isolation: "fixture only",
            },
        });
        CharacterizationProtocol {
            identity: ProtocolIdentity {
                protocol_revision: "fixture",
                fact_algebra_revision: "fixture",
                probe_library_digest: [1; 32],
                measurement_endpoint: "fixture",
            },
            budget: AcquisitionBudget {
                compile_ns: 10,
                warmup_ns: 10,
                measurement_ns: 10,
                total_ns: 30,
            },
            manifest: CompleteProbeManifest { entries },
        }
    }

    fn fixture_device() -> DeviceIdentity {
        DeviceIdentity {
            registry_id: 1,
            architecture: "fixture".into(),
            operating_system: "fixture".into(),
            metal_language: "fixture".into(),
            compiler_identity: "fixture".into(),
            numerical_mode: "fixture".into(),
        }
    }

    #[test]
    fn structural_validation_aggregates_all_errors() {
        let protocol = validation_fixture_protocol();
        let expected_device = fixture_device();
        let facts = MetalCostFact::ALL
            .map(|fact| RawFactObservation {
                fact,
                capture: RawFactCapture::Measured(SampleSeries {
                    endpoint: "fixture",
                    workload: vec![0],
                    observations_ns: vec![vec![]],
                    timer_resolution_ns: 0,
                }),
            })
            .into_iter()
            .collect();
        let raw = RawObservationBundle {
            device: DeviceIdentity {
                registry_id: 2,
                ..expected_device.clone()
            },
            protocol: protocol.identity().clone(),
            telemetry: AcquisitionTelemetry {
                compile_ns: 11,
                warmup_ns: 11,
                measurement_ns: 11,
                total_ns: 1,
            },
            facts,
        };
        let errors = validate_raw_bundle(&protocol, &expected_device, raw).unwrap_err();
        assert!(errors.0.iter().any(|e| e == "device identity mismatch"));
        assert!(errors.0.iter().any(|e| e.contains("compilation exceeded")));
        assert!(errors.0.iter().any(|e| e.contains("components exceed")));
        assert!(errors.0.iter().any(|e| e.contains("timer resolution")));
        assert!(errors.0.iter().any(|e| e.contains("zero units")));
        assert!(errors
            .0
            .iter()
            .any(|e| e.contains("observation set is empty")));
        assert!(errors.0.len() > MetalCostFact::COUNT * 2);
    }
}
