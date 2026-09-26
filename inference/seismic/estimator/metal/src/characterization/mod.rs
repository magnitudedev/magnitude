//! Isolated hardware-characterization boundary for the analytical Metal model.
//!
//! This module deliberately contains no device handle, compiler service,
//! candidate, solver, or planner type. A backend adapter may execute the fixed
//! protocol and return raw observations. Only the model-owned, pure
//! certification function can turn those observations into a profile.
//!
//! There is no partial-profile state. A `CertifiedMetalProfile<M>` contains
//! `M::CompleteParameters`, and `M::parameters` is total over `M::Fact`.

use std::fmt;
use std::marker::PhantomData;

/// Identity of the exact device/toolchain environment characterized by a run.
///
/// The native adapter derives this from immutable target discovery. It is a
/// value, not a native handle.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CharacterizationDeviceIdentity {
    pub device: [u8; 32],
    pub operating_system: Box<str>,
    pub metal_runtime: Box<str>,
    pub compiler: Box<str>,
    pub numerical_mode: [u8; 32],
}

/// Timer and synchronization semantics of every observation in one bundle.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MeasurementEndpoint {
    pub stable_name: Box<str>,
    pub timer: Box<str>,
    pub synchronization_boundary: Box<str>,
}

/// Explicit environmental controls retained with raw observations.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AcquisitionEnvironment {
    pub qos: Box<str>,
    pub thermal_policy: Box<str>,
    pub cache_policy: Box<str>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AcquisitionBudget {
    pub total_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AcquisitionMetrics {
    pub probe_compilation_ns: u64,
    pub probe_execution_ns: u64,
    pub total_ns: u64,
}

/// One failed fixed-probe obligation. Acquisition reports the whole batch.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProbeFailure {
    pub probe: Box<str>,
    pub reason: Box<str>,
}

/// Non-empty aggregate of fixed-probe failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbeFailures {
    items: Box<[ProbeFailure]>,
}

impl ProbeFailures {
    pub fn new(first: ProbeFailure, rest: Vec<ProbeFailure>) -> Self {
        let mut items = Vec::with_capacity(rest.len() + 1);
        items.push(first);
        items.extend(rest);
        Self {
            items: items.into_boxed_slice(),
        }
    }

    pub fn as_slice(&self) -> &[ProbeFailure] {
        &self.items
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AcquisitionError {
    DeviceMismatch,
    ProtocolMismatch,
    BudgetExceeded {
        budget_ns: u64,
        observed_ns: u64,
    },
    ProbeBatch {
        failures: ProbeFailures,
        metrics: AcquisitionMetrics,
    },
}

/// A certification failure is an all-or-nothing construction result. The
/// concrete model owns typed reasons because it owns the closed fact algebra.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileCertificationError<R> {
    reasons: Box<[R]>,
}

impl<R> ProfileCertificationError<R> {
    pub fn new(first: R, rest: Vec<R>) -> Self {
        let mut reasons = Vec::with_capacity(rest.len() + 1);
        reasons.push(first);
        reasons.extend(rest);
        Self {
            reasons: reasons.into_boxed_slice(),
        }
    }

    pub fn as_slice(&self) -> &[R] {
        &self.reasons
    }
}

mod private {
    pub trait Sealed {}
}

/// Model-owned closed characterization contract.
///
/// A production implementation lives beside the target-closed Metal physical
/// cost program. Its associated types are closed structs/enums rather than
/// open string maps:
///
/// - `CompleteProbeManifest` represents every queried, derived, or measured
///   obligation of the exact model version;
/// - `RawObservationBundle` contains every raw field required by certification;
/// - `CompleteParameters` contains every parameter required by evaluation;
/// - `parameters` is an exhaustive, infallible projection for every `Fact`.
///
/// The trait is sealed so a backend adapter cannot invent a second fact
/// vocabulary or certification policy.
pub trait MetalCharacterizationModel: private::Sealed + Sized + 'static {
    type Fact: Copy + fmt::Debug + Eq + 'static;
    type FactParameters: fmt::Debug + 'static;
    type CompleteProbeManifest: fmt::Debug + 'static;
    type RawObservationBundle: fmt::Debug + 'static;
    type CompleteParameters: fmt::Debug + 'static;
    type CertificationReason: fmt::Debug + 'static;

    fn protocol() -> CharacterizationProtocol<Self>;

    fn certify(
        device: &CharacterizationDeviceIdentity,
        protocol: &CharacterizationProtocol<Self>,
        observations: &Self::RawObservationBundle,
    ) -> Result<Self::CompleteParameters, ProfileCertificationError<Self::CertificationReason>>;

    fn parameters(profile: &Self::CompleteParameters, fact: Self::Fact) -> &Self::FactParameters;

    /// Content identity derived from the retained raw evidence. The model owns
    /// this derivation so a caller cannot attach an unrelated identity.
    fn evidence_fingerprint(observations: &Self::RawObservationBundle) -> [u8; 32];
}

/// Fixed, versioned protocol. It is created only by the sealed model.
pub struct CharacterizationProtocol<M: MetalCharacterizationModel> {
    identity: [u8; 32],
    model_version: Box<str>,
    probe_suite_version: Box<str>,
    manifest: M::CompleteProbeManifest,
    endpoint: MeasurementEndpoint,
    environment: AcquisitionEnvironment,
    budget: AcquisitionBudget,
}

impl<M: MetalCharacterizationModel> CharacterizationProtocol<M> {
    fn new(
        identity: [u8; 32],
        model_version: impl Into<Box<str>>,
        probe_suite_version: impl Into<Box<str>>,
        manifest: M::CompleteProbeManifest,
        endpoint: MeasurementEndpoint,
        environment: AcquisitionEnvironment,
        budget: AcquisitionBudget,
    ) -> Self {
        assert!(budget.total_ns != 0, "characterization budget is zero");
        Self {
            identity,
            model_version: model_version.into(),
            probe_suite_version: probe_suite_version.into(),
            manifest,
            endpoint,
            environment,
            budget,
        }
    }

    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
    pub fn model_version(&self) -> &str {
        &self.model_version
    }
    pub fn probe_suite_version(&self) -> &str {
        &self.probe_suite_version
    }
    pub fn manifest(&self) -> &M::CompleteProbeManifest {
        &self.manifest
    }
    pub fn endpoint(&self) -> &MeasurementEndpoint {
        &self.endpoint
    }
    pub fn environment(&self) -> &AcquisitionEnvironment {
        &self.environment
    }
    pub fn budget(&self) -> AcquisitionBudget {
        self.budget
    }
}

impl<M: MetalCharacterizationModel> fmt::Debug for CharacterizationProtocol<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharacterizationProtocol")
            .field("identity", &self.identity)
            .field("model_version", &self.model_version)
            .field("probe_suite_version", &self.probe_suite_version)
            .field("manifest", &self.manifest)
            .field("endpoint", &self.endpoint)
            .field("environment", &self.environment)
            .field("budget", &self.budget)
            .finish()
    }
}

/// Successful output of the backend adapter. The observations are raw and
/// retain provenance; no estimator coefficient or planner conclusion belongs
/// here.
#[derive(Debug)]
pub struct RawMetalObservationBundle<M: MetalCharacterizationModel> {
    device: CharacterizationDeviceIdentity,
    protocol: [u8; 32],
    endpoint: MeasurementEndpoint,
    environment: AcquisitionEnvironment,
    metrics: AcquisitionMetrics,
    observations: M::RawObservationBundle,
}

impl<M: MetalCharacterizationModel> RawMetalObservationBundle<M> {
    pub fn from_runner(
        device: CharacterizationDeviceIdentity,
        protocol: [u8; 32],
        endpoint: MeasurementEndpoint,
        environment: AcquisitionEnvironment,
        metrics: AcquisitionMetrics,
        observations: M::RawObservationBundle,
    ) -> Self {
        Self {
            device,
            protocol,
            endpoint,
            environment,
            metrics,
            observations,
        }
    }

    pub fn device(&self) -> &CharacterizationDeviceIdentity {
        &self.device
    }
    pub fn protocol(&self) -> [u8; 32] {
        self.protocol
    }
    pub fn endpoint(&self) -> &MeasurementEndpoint {
        &self.endpoint
    }
    pub fn environment(&self) -> &AcquisitionEnvironment {
        &self.environment
    }
    pub fn metrics(&self) -> AcquisitionMetrics {
        self.metrics
    }
    pub fn observations(&self) -> &M::RawObservationBundle {
        &self.observations
    }
}

/// Native adapter capability. The runner is already attached to one opened
/// device, so the contract grants no device-handle access to the model.
pub trait MetalProbeRunner<M: MetalCharacterizationModel> {
    fn device_identity(&self) -> CharacterizationDeviceIdentity;

    /// Compile the complete fixed manifest once and acquire one aggregate
    /// bundle. Candidate code, candidate identity, and candidate timing are
    /// absent from the interface.
    fn acquire(
        &mut self,
        protocol: &CharacterizationProtocol<M>,
    ) -> Result<RawMetalObservationBundle<M>, AcquisitionError>;
}

/// Immutable analytical-model input. Construction is private and possible
/// only after all protocol, identity, budget, and model-specific certification
/// checks succeed.
pub struct CertifiedMetalProfile<M: MetalCharacterizationModel> {
    device: CharacterizationDeviceIdentity,
    protocol: [u8; 32],
    evidence_fingerprint: [u8; 32],
    parameters: M::CompleteParameters,
    _model: PhantomData<fn() -> M>,
}

impl<M: MetalCharacterizationModel> CertifiedMetalProfile<M> {
    pub fn device(&self) -> &CharacterizationDeviceIdentity {
        &self.device
    }
    pub fn protocol(&self) -> [u8; 32] {
        self.protocol
    }
    pub fn evidence_fingerprint(&self) -> [u8; 32] {
        self.evidence_fingerprint
    }

    /// Total fact access. A successful profile has no absent/unassessed fact.
    pub fn parameters(&self, fact: M::Fact) -> &M::FactParameters {
        M::parameters(&self.parameters, fact)
    }
}

impl<M: MetalCharacterizationModel> fmt::Debug for CertifiedMetalProfile<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CertifiedMetalProfile")
            .field("device", &self.device)
            .field("protocol", &self.protocol)
            .field("evidence_fingerprint", &self.evidence_fingerprint)
            .field("parameters", &self.parameters)
            .finish()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum CertificationError<R> {
    DeviceMismatch,
    ProtocolMismatch,
    EndpointMismatch,
    EnvironmentMismatch,
    BudgetExceeded { budget_ns: u64, observed_ns: u64 },
    Model(ProfileCertificationError<R>),
}

/// Pure, replayable all-or-nothing certification. Recorded raw bundles may be
/// passed here repeatedly without compiling or running hardware.
pub fn certify_metal_profile<M: MetalCharacterizationModel>(
    expected_device: &CharacterizationDeviceIdentity,
    protocol: &CharacterizationProtocol<M>,
    raw: &RawMetalObservationBundle<M>,
) -> Result<CertifiedMetalProfile<M>, CertificationError<M::CertificationReason>> {
    if raw.device() != expected_device {
        return Err(CertificationError::DeviceMismatch);
    }
    if raw.protocol() != protocol.identity() {
        return Err(CertificationError::ProtocolMismatch);
    }
    if raw.endpoint() != protocol.endpoint() {
        return Err(CertificationError::EndpointMismatch);
    }
    if raw.environment() != protocol.environment() {
        return Err(CertificationError::EnvironmentMismatch);
    }
    if raw.metrics().total_ns > protocol.budget().total_ns {
        return Err(CertificationError::BudgetExceeded {
            budget_ns: protocol.budget().total_ns,
            observed_ns: raw.metrics().total_ns,
        });
    }
    let parameters = M::certify(expected_device, protocol, raw.observations())
        .map_err(CertificationError::Model)?;
    Ok(CertifiedMetalProfile {
        device: expected_device.clone(),
        protocol: protocol.identity(),
        evidence_fingerprint: M::evidence_fingerprint(raw.observations()),
        parameters,
        _model: PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Fixture;
    impl private::Sealed for Fixture {}

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Fact {
        Add,
        Load,
    }

    #[derive(Debug)]
    struct Parameters {
        add: u64,
        load: u64,
    }

    impl MetalCharacterizationModel for Fixture {
        type Fact = Fact;
        type FactParameters = u64;
        type CompleteProbeManifest = ();
        type RawObservationBundle = (u64, u64);
        type CompleteParameters = Parameters;
        type CertificationReason = &'static str;

        fn protocol() -> CharacterizationProtocol<Self> {
            CharacterizationProtocol::new(
                [3; 32],
                "fixture-model-v1",
                "fixture-probes-v1",
                (),
                MeasurementEndpoint {
                    stable_name: "fixture".into(),
                    timer: "host".into(),
                    synchronization_boundary: "complete".into(),
                },
                AcquisitionEnvironment {
                    qos: "fixed".into(),
                    thermal_policy: "fixed".into(),
                    cache_policy: "fixed".into(),
                },
                AcquisitionBudget { total_ns: 100 },
            )
        }

        fn certify(
            _device: &CharacterizationDeviceIdentity,
            _protocol: &CharacterizationProtocol<Self>,
            observations: &Self::RawObservationBundle,
        ) -> Result<Self::CompleteParameters, ProfileCertificationError<Self::CertificationReason>>
        {
            let mut reasons = Vec::new();
            if observations.0 == 0 {
                reasons.push("add is zero");
            }
            if observations.1 == 0 {
                reasons.push("load is zero");
            }
            if let Some(first) = reasons.first().copied() {
                return Err(ProfileCertificationError::new(
                    first,
                    reasons.into_iter().skip(1).collect(),
                ));
            }
            Ok(Parameters {
                add: observations.0,
                load: observations.1,
            })
        }

        fn parameters(profile: &Self::CompleteParameters, fact: Self::Fact) -> &u64 {
            match fact {
                Fact::Add => &profile.add,
                Fact::Load => &profile.load,
            }
        }

        fn evidence_fingerprint(observations: &Self::RawObservationBundle) -> [u8; 32] {
            let mut identity = [0; 32];
            identity[..8].copy_from_slice(&observations.0.to_le_bytes());
            identity[8..16].copy_from_slice(&observations.1.to_le_bytes());
            identity
        }
    }

    fn device() -> CharacterizationDeviceIdentity {
        CharacterizationDeviceIdentity {
            device: [1; 32],
            operating_system: "test-os".into(),
            metal_runtime: "test-metal".into(),
            compiler: "test-compiler".into(),
            numerical_mode: [2; 32],
        }
    }

    fn raw(
        protocol: &CharacterizationProtocol<Fixture>,
        observations: (u64, u64),
    ) -> RawMetalObservationBundle<Fixture> {
        RawMetalObservationBundle::from_runner(
            device(),
            protocol.identity(),
            protocol.endpoint().clone(),
            protocol.environment().clone(),
            AcquisitionMetrics {
                probe_compilation_ns: 10,
                probe_execution_ns: 20,
                total_ns: 30,
            },
            observations,
        )
    }

    #[test]
    fn successful_profile_has_total_infallible_fact_access() {
        let protocol = Fixture::protocol();
        let profile =
            certify_metal_profile(&device(), &protocol, &raw(&protocol, (7, 11))).unwrap();
        assert_eq!(*profile.parameters(Fact::Add), 7);
        assert_eq!(*profile.parameters(Fact::Load), 11);
    }

    #[test]
    fn certification_aggregates_model_failures_and_constructs_nothing() {
        let protocol = Fixture::protocol();
        let error =
            certify_metal_profile(&device(), &protocol, &raw(&protocol, (0, 0))).unwrap_err();
        assert_eq!(
            error,
            CertificationError::Model(ProfileCertificationError::new(
                "add is zero",
                vec!["load is zero"],
            ))
        );
    }

    #[test]
    fn certification_rejects_over_budget_bundle_before_model_interpretation() {
        let protocol = Fixture::protocol();
        let mut raw = raw(&protocol, (7, 11));
        raw.metrics.total_ns = 101;
        assert_eq!(
            certify_metal_profile(&device(), &protocol, &raw).unwrap_err(),
            CertificationError::BudgetExceeded {
                budget_ns: 100,
                observed_ns: 101,
            }
        );
    }
}
