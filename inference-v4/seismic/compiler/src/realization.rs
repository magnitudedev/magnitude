//! Demand-driven native realization and opaque handle ownership.
//!
//! Structural candidate domains contain no native artifacts. Preparation
//! hands an exact canonical coordinate to [`Realizer`], which compiles only
//! the kernels retained by IR native specialization. The private registry
//! owns one shared handle for each native artifact identity and records dense
//! active manifests without exposing handles to evaluation or planning.

use crate::errors::PreparationError;
use crate::implementation::ImplementationIdentity;
use crate::refinement::{CandidateFamily, CandidateFamilyIdentity};
use seismic_ir::kernel::KernelId;
use seismic_ir::schedule::LaunchId;
use seismic_lang::expr::{AnyExpr, DecisionId, ExprArena, PartialAssignment, SymbolValue};
use seismic_target::{
    CompatibilityIdentity, DeviceDescription, DeviceDescriptionIdentity, NativeArtifactMetrics,
    NativeCompiler, NativeKernelDescription, NativeKernelIdentity, TargetFamily,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;

/// Exact pre-native identity of one canonical family assignment.
///
/// The family identity names the structural candidate. `assignment` is
/// digested in the family's declared decision order, so caller order and raw
/// arena ordinals cannot create aliases.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CandidateRealizationIdentity {
    family: CandidateFamilyIdentity,
    assignment: [u8; 32],
}

/// An exact family assignment from the checked structural domain.
///
/// The request owns the family and assignment rather than a specialization
/// borrowing the family. This lets reconciliation retain the family in its
/// coordinate-exact output while native specialization remains a local,
/// inspectable transition inside [`Realizer::realize`].
pub(crate) struct CanonicalRealizationRequest<T: TargetFamily> {
    identity: CandidateRealizationIdentity,
    family: Arc<CandidateFamily<T>>,
    assignment: PartialAssignment,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum RealizationRequestError {
    ForeignFamily,
    DuplicateChoice(DecisionId),
    ForeignChoice(DecisionId),
    ValueOutsideAxis { decision: DecisionId, value: i64 },
}

impl<T: TargetFamily> CanonicalRealizationRequest<T> {
    /// Adapter for `CandidateDomain::check`: the domain supplies its checked
    /// family and canonical coordinate together. The coordinate already
    /// contains exactly the active choices in declaration order.
    pub(crate) fn from_checked_coordinate(
        family: Arc<CandidateFamily<T>>,
        arena: &ExprArena,
        coordinate: &crate::candidate_domain::CandidateCoordinate,
    ) -> Result<Self, RealizationRequestError> {
        if coordinate.family() != family.identity() {
            return Err(RealizationRequestError::ForeignFamily);
        }
        let mut fixed = PartialAssignment::new();
        for (decision, value) in coordinate.choices() {
            let Some(_) = family
                .choices()
                .iter()
                .find(|candidate| candidate.decision() == *decision)
            else {
                return Err(RealizationRequestError::ForeignChoice(*decision));
            };
            let symbol = arena.decision_symbol(*decision);
            if fixed.get(symbol).is_some() {
                return Err(RealizationRequestError::DuplicateChoice(*decision));
            }
            if !arena.decision_domain(*decision).values().contains(value) {
                return Err(RealizationRequestError::ValueOutsideAxis {
                    decision: *decision,
                    value: *value,
                });
            }
            fixed.bind(symbol, SymbolValue::Int(*value));
        }

        let mut digest = Sha256::new();
        digest.update(b"seismic-canonical-realization-assignment-v1");
        for (ordinal, (_, value)) in coordinate.choices().iter().enumerate() {
            digest.update((ordinal as u64).to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        let identity = CandidateRealizationIdentity {
            family: family.identity().clone(),
            assignment: digest.finalize().into(),
        };
        Ok(Self {
            identity,
            family,
            assignment: fixed,
        })
    }
}

/// Stable lookup key for one emitted kernel template. Coordinate choices do
/// not enter this key: two coordinates reuse an artifact exactly when they
/// select the same original family kernel under the same native compatibility
/// contract.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct NativeArtifactRequestKey {
    compatibility: CompatibilityIdentity,
    family: CandidateFamilyIdentity,
    kernel_ordinal: u32,
}

/// A deterministic, candidate-specific post-reflection rejection. These are
/// cached. NativeCompilationError is deliberately absent and always remains a
/// retryable preparation/service failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CandidateRejection {
    reason: String,
}

impl CandidateRejection {
    pub(crate) fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    pub(crate) fn reason(&self) -> &str {
        &self.reason
    }
}

/// Handle-free dense active native set. `kernels` is in first-active-launch
/// order and retains each original KernelId, which is the exact remap later
/// executable translation consumes after schedule-Choose elimination.
#[derive(Debug)]
pub(crate) struct RealizedNativeSet<T: TargetFamily> {
    identity: CandidateRealizationIdentity,
    active_launches: Box<[LaunchId]>,
    kernels: Box<[RealizedKernel<T>]>,
}

#[derive(Debug)]
struct RealizedKernel<T: TargetFamily> {
    original: KernelId,
    artifact: NativeKernelIdentity,
    description: NativeKernelDescription<T>,
}

impl<T: TargetFamily> RealizedNativeSet<T> {
    pub(crate) fn identity(&self) -> &CandidateRealizationIdentity {
        &self.identity
    }

    pub(crate) fn assignment_identity(&self) -> [u8; 32] {
        self.identity.assignment
    }

    pub(crate) fn active_launches(&self) -> &[LaunchId] {
        &self.active_launches
    }

    pub(crate) fn descriptions(
        &self,
    ) -> impl ExactSizeIterator<Item = &NativeKernelDescription<T>> {
        self.kernels.iter().map(|kernel| &kernel.description)
    }

    pub(crate) fn original_kernels(&self) -> impl ExactSizeIterator<Item = KernelId> + '_ {
        self.kernels.iter().map(|kernel| kernel.original)
    }

    pub(crate) fn description(&self, original: KernelId) -> Option<&NativeKernelDescription<T>> {
        self.kernels
            .iter()
            .find(|kernel| kernel.original == original)
            .map(|kernel| &kernel.description)
    }

    pub(crate) fn native_kernel_index(&self, original: KernelId) -> Option<u32> {
        self.kernels
            .iter()
            .position(|kernel| kernel.original == original)
            .map(|index| u32::try_from(index).expect("active native kernel count exceeds u32"))
    }

    fn artifact_identities(&self) -> impl ExactSizeIterator<Item = &NativeKernelIdentity> {
        self.kernels.iter().map(|kernel| &kernel.artifact)
    }
}

#[cfg(test)]
pub(crate) mod demand_driven_tests {
    use super::*;
    use crate::numerics::NumericalTransfer;
    use crate::refinement::{
        CandidateFamilyParts, ChoiceDeclaration, ConstructionAuthority, FactoryIdentity,
        ImplementationProvenance,
    };
    use seismic_ir::construction::{AllocationPlan, Construction};
    use seismic_ir::kernel::{Kernel, KernelId};
    use seismic_ir::target::{
        IntrinsicIdentityBuilder, IntrinsicNumericalSemantics, KernelAbiFootprint, KernelAbiLayout,
        KernelAbiModel, KernelEmissionLayout, LocalRealization, LocalRealizationPolicy,
        NumericalEnvironment, TargetLimits, VectorSupport,
    };
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::entry::{ElementBindings, NumericalRole};
    use seismic_lang::expr::{FiniteDomain, PartialAssignment, TargetPredicate};
    use seismic_lang::registry::{BackendName, IntrinsicSignature};
    use seismic_target::{
        CompatibilityIdentity, DeviceDescriptionParts, NativeClusterDomain, NativeCompilationError,
        NativeKernelReflection, NativeLaunchDomain, NativeNumericalModeIdentity,
        NativeResourceUsage, NativeResources, NumericalEnvironmentIdentity,
    };
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Debug)]
    pub(crate) struct FakeTarget;

    #[derive(Clone, Debug, PartialEq)]
    pub(crate) struct FakeAbi;

    #[derive(Clone, Debug)]
    pub(crate) enum NoIntrinsic {}

    impl seismic_ir::target::KernelDialect for FakeTarget {
        const NAME: BackendName = BackendName::Cpu;
        type Facts = ();
        type Intrinsic = NoIntrinsic;

        fn write_intrinsic_identity(op: &NoIntrinsic, _: &mut IntrinsicIdentityBuilder) {
            match *op {}
        }

        fn intrinsic_numerics(
            _: &(),
            _: &IntrinsicSignature,
            op: &NoIntrinsic,
        ) -> IntrinsicNumericalSemantics {
            match *op {}
        }

        fn intrinsic_addressable_resources(
            op: &NoIntrinsic,
        ) -> Vec<seismic_ir::kernel::ops::AddressableResourceHandle> {
            match *op {}
        }
    }

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
        type NativeLaunchMode = ();
        type NativeNumericalMode = ();
        type NativeProperties = ();
    }

    pub(crate) struct CountingCompiler {
        forms: AtomicUsize,
        fail_next: AtomicBool,
    }

    impl CountingCompiler {
        pub(crate) fn new() -> Self {
            Self {
                forms: AtomicUsize::new(0),
                fail_next: AtomicBool::new(false),
            }
        }

        pub(crate) fn form_count(&self) -> usize {
            self.forms.load(Ordering::SeqCst)
        }
    }

    impl NativeCompiler<FakeTarget> for CountingCompiler {
        type Context = ();
        type Candidate = usize;
        type Handle = usize;

        fn form(
            &self,
            _: &(),
            _: &DeviceDescription<FakeTarget>,
            _: &Kernel<FakeTarget>,
            _: &KernelEmissionLayout,
        ) -> Result<Self::Candidate, NativeCompilationError> {
            let ordinal = self.forms.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_next.swap(false, Ordering::SeqCst) {
                return Err(NativeCompilationError::ToolchainFailure(
                    "injected infrastructure failure".into(),
                ));
            }
            Ok(ordinal)
        }

        fn reflect(
            &self,
            target: &DeviceDescription<FakeTarget>,
            kernel: &Kernel<FakeTarget>,
            _: &KernelEmissionLayout,
            candidate: Self::Candidate,
        ) -> Result<NativeKernelReflection<FakeTarget, Self::Handle>, NativeCompilationError>
        {
            let mut digest = [0; 32];
            digest[..8].copy_from_slice(&(candidate as u64).to_le_bytes());
            Ok(NativeKernelReflection::new(
                candidate,
                NativeKernelDescription {
                    identity: NativeKernelIdentity {
                        compatibility: target.compatibility_identity().clone(),
                        artifact_digest: digest,
                    },
                    abi: target.kernel_abi_layout(kernel),
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
                        fingerprint: [7; 32],
                    },
                    properties: (),
                },
                NativeArtifactMetrics {
                    compilation_ns: 1,
                    code_bytes: 1,
                    metadata_bytes: 1,
                },
            ))
        }
    }

    struct Accept;
    impl NativeCandidateReconciler<FakeTarget> for Accept {
        type Output = ();

        fn reconcile(
            &mut self,
            _: &mut ExprArena,
            _: ReconciliationInput<FakeTarget>,
        ) -> Result<Self::Output, NativeReconciliationError> {
            Ok(())
        }
    }

    struct Reject {
        calls: Arc<AtomicUsize>,
    }
    impl NativeCandidateReconciler<FakeTarget> for Reject {
        type Output = ();

        fn reconcile(
            &mut self,
            _: &mut ExprArena,
            _: ReconciliationInput<FakeTarget>,
        ) -> Result<Self::Output, NativeReconciliationError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(NativeReconciliationError::Rejected(
                CandidateRejection::new("reflected limit"),
            ))
        }
    }

    pub(crate) fn device() -> DeviceDescription<FakeTarget> {
        DeviceDescription::new(DeviceDescriptionParts {
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
            vectors: VectorSupport::default(),
            numerics: NumericalEnvironment {
                contraction_available: false,
                flush_to_zero_available: false,
                approximate_transcendentals: BTreeSet::new(),
                denormals_preserved: true,
            },
            capabilities: BTreeSet::new(),
            intrinsics: BTreeSet::new(),
            facts: (),
            kernel_abi: FakeAbi,
            local_realization: LocalRealizationPolicy {
                workgroup: LocalRealization::NativeDynamic,
                participant: LocalRealization::NativeDynamic,
                register: LocalRealization::NativeDynamic,
            },
            addressable_resources: Vec::new(),
        })
        .unwrap()
    }

    pub(crate) fn choice_family(
        arena: &mut ExprArena,
    ) -> (Arc<CandidateFamily<FakeTarget>>, DecisionId, [KernelId; 3]) {
        let mut construction = Construction::<FakeTarget>::new(arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let first = construction
            .portable_kernel(arena, &(), &[], &vectors)
            .close();
        let second = construction
            .portable_kernel(arena, &(), &[], &vectors)
            .close();
        let shared = construction
            .portable_kernel(arena, &(), &[], &vectors)
            .close();
        let decision = arena.decision(FiniteDomain::new(vec![0, 1]).unwrap());
        let mut schedule = construction.schedule(arena, 0);
        schedule.choose(decision, |choice| {
            choice.option(0, |selected| {
                selected.launch_sequential(first);
                selected.launch_sequential(shared);
            });
            choice.option(1, |selected| {
                selected.launch_sequential(second);
                selected.launch_sequential(shared);
            });
        });
        let token = schedule.close();
        let executable = construction
            .close(token)
            .analyze_allocations()
            .apply_allocation_plan(arena, AllocationPlan::distinct())
            .finish()
            .close_execution(arena);
        let always = arena.bool(true);
        let coverage = TargetPredicate::new(arena, always).unwrap();
        let launch_count = executable.schedule().launches().len();
        let family = Arc::new(CandidateFamily::from_parts(CandidateFamilyParts {
            authority: ConstructionAuthority::Optimized,
            numerical_role: NumericalRole::Alternative,
            identity: family_identity(),
            semantic_coverage: coverage,
            executable,
            launch_scratch: vec![
                seismic_ir::storage::LaunchScratchRequirements::default();
                launch_count
            ],
            launch_abi: vec![Vec::new(); launch_count],
            choices: vec![ChoiceDeclaration {
                decision,
                meaning: "selected branch",
                active_when: always,
            }],
            hard_constraints: always,
            numerical_transfer: NumericalTransfer::new(
                Vec::new(),
                Vec::new(),
                Vec::new(),
                BTreeMap::new(),
                Vec::new(),
            ),
            provenance: ImplementationProvenance {
                root: fixture_function_identity(),
                callees: Vec::new(),
            },
            result_publications: Vec::new(),
        }));
        (family, decision, [first, second, shared])
    }

    fn fixture_function_identity() -> seismic_lang::ids::StableFunctionId {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "realizer-fixture.seismic".into(),
            text: "fn probe(x: f32) -> f32:\n    return x\n".into(),
        }]))
        .unwrap();
        let entry = module
            .entry(
                module.entry_named("probe").unwrap(),
                &ElementBindings::default(),
            )
            .unwrap();
        let identity = entry.program().functions().next().unwrap().1.stable();
        identity
    }

    fn request(
        family: Arc<CandidateFamily<FakeTarget>>,
        identity: u8,
        assignment: PartialAssignment,
    ) -> CanonicalRealizationRequest<FakeTarget> {
        CanonicalRealizationRequest {
            identity: CandidateRealizationIdentity {
                family: family.identity().clone(),
                assignment: [identity; 32],
            },
            family,
            assignment,
        }
    }

    fn assignment(arena: &ExprArena, decision: DecisionId, value: i64) -> PartialAssignment {
        let mut assignment = PartialAssignment::new();
        assignment.bind(arena.decision_symbol(decision), SymbolValue::Int(value));
        assignment
    }

    fn family_identity() -> CandidateFamilyIdentity {
        CandidateFamilyIdentity {
            factory: FactoryIdentity {
                name: "counting",
                revision: "v1",
            },
            structure: [9; 32],
        }
    }

    #[test]
    fn realization_is_selected_only_cached_and_retries_infrastructure_failure() {
        let mut arena = ExprArena::new();
        let (family, decision, [first, second, shared]) = choice_family(&mut arena);
        let target = device();
        let compiler = CountingCompiler::new();
        let mut realizer = Realizer::new(&compiler, &(), &target, Accept);
        assert_eq!(compiler.form_count(), 0, "construction must not compile");

        let zero_assignment = assignment(&arena, decision, 0);
        let zero = request(family.clone(), 0, zero_assignment.clone());
        let RealizationOutcome::Ready { candidate, .. } =
            realizer.realize(&mut arena, zero).unwrap()
        else {
            panic!("selected candidate was rejected")
        };
        let native = candidate.native();
        assert_eq!(compiler.form_count(), 2);
        assert_eq!(native.native_kernel_index(first), Some(0));
        assert_eq!(native.native_kernel_index(shared), Some(1));
        assert_eq!(native.native_kernel_index(second), None);

        let zero = request(family.clone(), 0, zero_assignment.clone());
        let _ = realizer.realize(&mut arena, zero).unwrap();
        assert_eq!(compiler.form_count(), 2, "candidate cache must be exact");

        let one_assignment = assignment(&arena, decision, 1);
        let one = request(family.clone(), 1, one_assignment);
        let RealizationOutcome::Ready { candidate, .. } =
            realizer.realize(&mut arena, one).unwrap()
        else {
            panic!("selected candidate was rejected")
        };
        let selected_native = candidate.native().clone();
        let native = selected_native.as_ref();
        assert_eq!(compiler.form_count(), 3, "shared kernel must be reused");
        assert_eq!(native.native_kernel_index(first), None);
        assert_eq!(native.native_kernel_index(second), Some(0));
        assert_eq!(native.native_kernel_index(shared), Some(1));

        let implementation = ImplementationIdentity {
            factory: family.identity().factory.clone(),
            structure: [4; 32],
        };
        let expected = native
            .descriptions()
            .map(|description| description.identity.clone())
            .collect::<Vec<_>>();
        let mut registry = realizer.into_registry();
        registry
            .bind_implementation(implementation.clone(), native)
            .unwrap();
        assert_eq!(
            registry
                .resolve(target.identity(), &implementation, &expected)
                .len(),
            2
        );

        let retry_compiler = CountingCompiler::new();
        retry_compiler.fail_next.store(true, Ordering::SeqCst);
        let mut retry = Realizer::new(&retry_compiler, &(), &target, Accept);
        let failed = request(family.clone(), 0, zero_assignment.clone());
        assert!(matches!(
            retry.realize(&mut arena, failed),
            Err(PreparationError::NativeCompilation(
                NativeCompilationError::ToolchainFailure(_)
            ))
        ));
        assert_eq!(retry_compiler.form_count(), 1);
        let retried = request(family, 0, zero_assignment);
        assert!(matches!(
            retry.realize(&mut arena, retried),
            Ok(RealizationOutcome::Ready { .. })
        ));
        assert_eq!(retry_compiler.form_count(), 3);
    }

    #[test]
    fn deterministic_rejection_is_cached_after_reflection() {
        let mut arena = ExprArena::new();
        let (family, decision, _) = choice_family(&mut arena);
        let target = device();
        let compiler = CountingCompiler::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let mut realizer = Realizer::new(
            &compiler,
            &(),
            &target,
            Reject {
                calls: calls.clone(),
            },
        );
        let fixed = assignment(&arena, decision, 0);
        let first = request(family.clone(), 0, fixed.clone());
        let RealizationOutcome::Rejected {
            rejection,
            newly_formed,
        } = realizer.realize(&mut arena, first).unwrap()
        else {
            panic!("candidate should be rejected")
        };
        assert_eq!(rejection.reason(), "reflected limit");
        assert_eq!(compiler.form_count(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(newly_formed.code_bytes, 2);
        assert_eq!(newly_formed.metadata_bytes, 2);
        assert_eq!(newly_formed.compilation_ns, 2);
        let mut budget = crate::preparation_budget::PreparationBudgetTracker::new(
            crate::preparation_budget::PreparationBudget {
                native_code_bytes: 1,
                ..crate::preparation_budget::PreparationBudget::default()
            },
        );
        assert!(!budget.record_native_artifact(newly_formed));

        let again = request(family, 0, fixed);
        let RealizationOutcome::Rejected { newly_formed, .. } =
            realizer.realize(&mut arena, again).unwrap()
        else {
            panic!("cached candidate should be rejected")
        };
        assert_eq!(newly_formed.code_bytes, 0);
        assert_eq!(newly_formed.metadata_bytes, 0);
        assert_eq!(newly_formed.compilation_ns, 0);
        assert_eq!(compiler.form_count(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

pub(crate) struct RealizedCandidate<T: TargetFamily, O> {
    native: Arc<RealizedNativeSet<T>>,
    reconciled: Arc<O>,
}

impl<T: TargetFamily, O> RealizedCandidate<T, O> {
    pub(crate) fn native(&self) -> &Arc<RealizedNativeSet<T>> {
        &self.native
    }

    pub(crate) fn reconciled(&self) -> &Arc<O> {
        &self.reconciled
    }
}

pub(crate) enum RealizationOutcome<T: TargetFamily, O> {
    Ready {
        candidate: Arc<RealizedCandidate<T, O>>,
        newly_formed: NativeArtifactMetrics,
    },
    Rejected {
        rejection: Arc<CandidateRejection>,
        newly_formed: NativeArtifactMetrics,
    },
}

enum CachedCandidate<T: TargetFamily, O> {
    Ready(Arc<RealizedCandidate<T, O>>),
    Rejected(Arc<CandidateRejection>),
}

pub(crate) enum NativeReconciliationError {
    Rejected(CandidateRejection),
    Preparation(PreparationError),
}

/// Complete owned input to the one post-reflection reconciliation transition.
/// It carries the same checked family and exact assignment used to derive the
/// dense native set, so an output can retain them without reconstructing or
/// re-identifying the candidate.
pub(crate) struct ReconciliationInput<T: TargetFamily> {
    family: Arc<CandidateFamily<T>>,
    assignment: PartialAssignment,
    native: Arc<RealizedNativeSet<T>>,
}

impl<T: TargetFamily> ReconciliationInput<T> {
    pub(crate) fn into_parts(
        self,
    ) -> (
        Arc<CandidateFamily<T>>,
        PartialAssignment,
        Arc<RealizedNativeSet<T>>,
    ) {
        (self.family, self.assignment, self.native)
    }
}

/// The one compiler-owned post-reflection legality transition used by a
/// Realizer. Keeping it in the service object makes cached rejection
/// independent of whichever search strategy requested a coordinate.
pub(crate) trait NativeCandidateReconciler<T: TargetFamily> {
    type Output;

    fn reconcile(
        &mut self,
        arena: &mut ExprArena,
        input: ReconciliationInput<T>,
    ) -> Result<Self::Output, NativeReconciliationError>;
}

/// Demand-driven native formation for one target/compiler/context triple.
/// These live services and the sole handle registry never cross into the
/// structural domain, evaluator, or planner.
pub(crate) struct Realizer<
    'a,
    T: TargetFamily,
    C: NativeCompiler<T>,
    R: NativeCandidateReconciler<T>,
> {
    compiler: &'a C,
    context: &'a C::Context,
    target: &'a DeviceDescription<T>,
    reconciler: R,
    registry: RealizationRegistry<T, C::Handle>,
    candidates: HashMap<CandidateRealizationIdentity, CachedCandidate<T, R::Output>>,
}

impl<'a, T, C, R> Realizer<'a, T, C, R>
where
    T: TargetFamily,
    C: NativeCompiler<T>,
    R: NativeCandidateReconciler<T>,
{
    pub(crate) fn new(
        compiler: &'a C,
        context: &'a C::Context,
        target: &'a DeviceDescription<T>,
        reconciler: R,
    ) -> Self {
        Self {
            compiler,
            context,
            target,
            reconciler,
            registry: RealizationRegistry::new(target.identity().clone()),
            candidates: HashMap::new(),
        }
    }

    /// Specializes the exact assignment and realizes only active kernels.
    /// Reconciliation is the sole post-reflection eligibility transition:
    /// returning Rejected caches that deterministic result, while every
    /// native infrastructure error returns before the candidate cache changes.
    pub(crate) fn realize(
        &mut self,
        arena: &mut ExprArena,
        request: CanonicalRealizationRequest<T>,
    ) -> Result<RealizationOutcome<T, R::Output>, PreparationError> {
        // Validate the authoritative arena even on a candidate-cache hit.
        // Reconciled outputs may retain nodes added to this arena, so serving
        // them under a different arena would violate their identity contract.
        arena.view(AnyExpr::Bool(request.family.hard_constraints()));
        if let Some(cached) = self.candidates.get(&request.identity) {
            return Ok(match cached {
                CachedCandidate::Ready(candidate) => RealizationOutcome::Ready {
                    candidate: candidate.clone(),
                    newly_formed: empty_metrics(),
                },
                CachedCandidate::Rejected(rejection) => RealizationOutcome::Rejected {
                    rejection: rejection.clone(),
                    newly_formed: empty_metrics(),
                },
            });
        }

        let CanonicalRealizationRequest {
            identity,
            family,
            assignment,
        } = request;
        let (native, formed) = {
            let specialization = family
                .executable
                .specialize_for_native(&*arena, &assignment)
                .unwrap_or_else(|error| {
                    panic!("checked candidate failed native specialization: {error}")
                });
            let mut formed = empty_metrics();
            let mut realized = Vec::with_capacity(specialization.kernels().len());
            for (original, kernel) in specialization.kernels() {
                let key = NativeArtifactRequestKey {
                    compatibility: self.target.compatibility_identity().clone(),
                    family: identity.family.clone(),
                    kernel_ordinal: original.ordinal(),
                };
                let artifact = if let Some(identity) = self.registry.artifact_requests.get(&key) {
                    identity.clone()
                } else {
                    let (native, metrics) = crate::implementation::native::realize_kernel(
                        self.compiler,
                        self.context,
                        kernel,
                        self.target,
                    )?;
                    add_metrics(&mut formed, metrics)?;
                    let artifact = self.registry.artifacts.retain_one(
                        self.target.compatibility_identity(),
                        identity.family.factory.name,
                        native,
                    )?;
                    self.registry
                        .artifact_requests
                        .insert(key, artifact.clone());
                    artifact
                };
                let description = self
                    .registry
                    .artifacts
                    .description(&artifact)
                    .unwrap_or_else(|| panic!("artifact request cache references absent storage"))
                    .clone();
                realized.push(RealizedKernel {
                    original,
                    artifact,
                    description,
                });
            }
            let native = Arc::new(RealizedNativeSet {
                identity: identity.clone(),
                active_launches: specialization
                    .launches()
                    .map(|launch| launch.id)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                kernels: realized.into_boxed_slice(),
            });
            (native, formed)
        };
        let reconciliation = ReconciliationInput {
            family,
            assignment,
            native: native.clone(),
        };
        let reconciled = match self.reconciler.reconcile(arena, reconciliation) {
            Ok(reconciled) => reconciled,
            Err(NativeReconciliationError::Rejected(rejection)) => {
                let rejection = Arc::new(rejection);
                self.candidates
                    .insert(identity, CachedCandidate::Rejected(rejection.clone()));
                return Ok(RealizationOutcome::Rejected {
                    rejection,
                    newly_formed: formed,
                });
            }
            Err(NativeReconciliationError::Preparation(error)) => return Err(error),
        };
        let candidate = Arc::new(RealizedCandidate {
            native: native.clone(),
            reconciled: Arc::new(reconciled),
        });
        let replaced = self.registry.realized.insert(identity.clone(), native);
        assert!(
            replaced.is_none(),
            "realized candidate manifest changed after successful reconciliation"
        );
        self.candidates
            .insert(identity, CachedCandidate::Ready(candidate.clone()));
        Ok(RealizationOutcome::Ready {
            candidate,
            newly_formed: formed,
        })
    }

    pub(crate) fn into_registry(self) -> RealizationRegistry<T, C::Handle> {
        self.registry
    }
}

fn empty_metrics() -> NativeArtifactMetrics {
    NativeArtifactMetrics {
        compilation_ns: 0,
        code_bytes: 0,
        metadata_bytes: 0,
    }
}

fn add_metrics(
    total: &mut NativeArtifactMetrics,
    artifact: NativeArtifactMetrics,
) -> Result<(), PreparationError> {
    let overflow = |quantity: &'static str| {
        PreparationError::NativeCompilation(
            seismic_target::NativeCompilationError::ToolchainResourceExhausted(format!(
                "aggregate native {quantity} exceeds u64"
            )),
        )
    };
    total.compilation_ns = total
        .compilation_ns
        .checked_add(artifact.compilation_ns)
        .ok_or_else(|| overflow("compilation duration in nanoseconds"))?;
    total.code_bytes = total
        .code_bytes
        .checked_add(artifact.code_bytes)
        .ok_or_else(|| overflow("code size in bytes"))?;
    total.metadata_bytes = total
        .metadata_bytes
        .checked_add(artifact.metadata_bytes)
        .ok_or_else(|| overflow("metadata size in bytes"))?;
    Ok(())
}

/// The sole owner of candidate-native handles before materialization.
///
/// `artifacts` is content-addressed native storage. `manifests` is the exact
/// implementation-to-artifact relation. Insertion validates a complete batch
/// before either map changes, so failed insertion cannot leave partial state.
#[derive(Debug)]
pub(crate) struct RealizationRegistry<T: TargetFamily, H> {
    device: DeviceDescriptionIdentity,
    artifacts: NativeArtifactStore<T, H>,
    manifests: ImplementationManifest,
    artifact_requests: HashMap<NativeArtifactRequestKey, NativeKernelIdentity>,
    realized: HashMap<CandidateRealizationIdentity, Arc<RealizedNativeSet<T>>>,
}

/// Content-addressed native handle ownership. This type alone decides whether
/// a repeated identity denotes the same artifact and performs handle dedupe.
#[derive(Debug)]
struct NativeArtifactStore<T: TargetFamily, H> {
    by_identity:
        HashMap<seismic_target::NativeKernelIdentity, Arc<seismic_target::NativeKernel<T, H>>>,
}

/// Owns implementation membership and the exact semantic-kernel-ordinal to
/// artifact-identity relation. Slice order is semantic kernel ordinal.
#[derive(Debug)]
struct ImplementationManifest {
    by_implementation: HashMap<ImplementationIdentity, Box<[seismic_target::NativeKernelIdentity]>>,
}

impl ImplementationManifest {
    fn new() -> Self {
        Self {
            by_implementation: HashMap::new(),
        }
    }

    fn validate_absent(
        &self,
        implementation: &ImplementationIdentity,
    ) -> Result<(), PreparationError> {
        if self.by_implementation.contains_key(implementation) {
            return Err(PreparationError::InvalidCandidateDomain(format!(
                "duplicate implementation identity from factory `{}`",
                implementation.factory.name
            )));
        }
        Ok(())
    }

    /// Publishes a manifest only after every referenced artifact is retained.
    fn commit(
        &mut self,
        implementation: ImplementationIdentity,
        artifacts: Box<[seismic_target::NativeKernelIdentity]>,
    ) {
        let replaced = self.by_implementation.insert(implementation, artifacts);
        assert!(
            replaced.is_none(),
            "realization manifest changed after successful batch preflight"
        );
    }

    fn resolve_exact<'a>(
        &'a self,
        implementation: &ImplementationIdentity,
        expected: &[seismic_target::NativeKernelIdentity],
    ) -> &'a [seismic_target::NativeKernelIdentity] {
        let manifest = self
            .by_implementation
            .get(implementation)
            .unwrap_or_else(|| {
                panic!("planned implementation is absent from its realization registry")
            });
        assert_eq!(
            manifest.as_ref(),
            expected,
            "planned implementation artifact manifest differs from its realized manifest"
        );
        manifest
    }
}

impl<T: TargetFamily, H> NativeArtifactStore<T, H> {
    fn new() -> Self {
        Self {
            by_identity: HashMap::new(),
        }
    }

    fn retain_one(
        &mut self,
        compatibility: &CompatibilityIdentity,
        factory: &str,
        kernel: seismic_target::NativeKernel<T, H>,
    ) -> Result<NativeKernelIdentity, PreparationError> {
        let description = kernel.description();
        if &description.identity.compatibility != compatibility {
            return Err(PreparationError::InvalidCandidateDomain(format!(
                "implementation from factory `{factory}` contains a native artifact for another compatibility domain"
            )));
        }
        let identity = description.identity.clone();
        if let Some(existing) = self.by_identity.get(&identity) {
            if existing.description() != description {
                return Err(PreparationError::InvalidCandidateDomain(format!(
                    "native artifact identity collision from factory `{factory}` has conflicting reflected descriptions"
                )));
            }
            return Ok(identity);
        }
        self.by_identity.insert(identity.clone(), Arc::new(kernel));
        Ok(identity)
    }

    fn get(
        &self,
        identity: &seismic_target::NativeKernelIdentity,
    ) -> Option<Arc<seismic_target::NativeKernel<T, H>>> {
        self.by_identity.get(identity).cloned()
    }

    fn description(&self, identity: &NativeKernelIdentity) -> Option<&NativeKernelDescription<T>> {
        self.by_identity
            .get(identity)
            .map(|kernel| kernel.description())
    }
}

impl<T: TargetFamily, H> RealizationRegistry<T, H> {
    pub(crate) fn new(device: DeviceDescriptionIdentity) -> Self {
        Self {
            device,
            artifacts: NativeArtifactStore::new(),
            manifests: ImplementationManifest::new(),
            artifact_requests: HashMap::new(),
            realized: HashMap::new(),
        }
    }

    /// Binds a reconciled implementation identity to the dense native set
    /// previously published by this registry's Realizer. All validation
    /// completes before the implementation manifest changes.
    pub(crate) fn bind_implementation(
        &mut self,
        implementation: ImplementationIdentity,
        native: &RealizedNativeSet<T>,
    ) -> Result<(), PreparationError> {
        self.manifests.validate_absent(&implementation)?;
        let published = self.realized.get(native.identity()).ok_or_else(|| {
            PreparationError::InvalidCandidateDomain(format!(
                "implementation from factory `{}` references an unpublished native candidate",
                implementation.factory.name
            ))
        })?;
        if !std::ptr::eq(published.as_ref(), native) {
            return Err(PreparationError::InvalidCandidateDomain(format!(
                "implementation from factory `{}` references a native candidate published by another registry",
                implementation.factory.name
            )));
        }
        let artifacts = native
            .artifact_identities()
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        if artifacts
            .iter()
            .any(|identity| !self.artifacts.by_identity.contains_key(identity))
        {
            return Err(PreparationError::InvalidCandidateDomain(format!(
                "implementation from factory `{}` references an absent native artifact",
                implementation.factory.name
            )));
        }
        self.manifests.commit(implementation, artifacts);
        Ok(())
    }

    pub(crate) fn resolve(
        &self,
        device: &DeviceDescriptionIdentity,
        implementation: &ImplementationIdentity,
        expected: &[seismic_target::NativeKernelIdentity],
    ) -> Vec<Arc<seismic_target::NativeKernel<T, H>>> {
        assert_eq!(
            device, &self.device,
            "planned policy and private realization registry have different device identities"
        );
        let manifest = self.manifests.resolve_exact(implementation, expected);
        manifest
            .iter()
            .map(|artifact| {
                self.artifacts.get(artifact).unwrap_or_else(|| {
                    panic!("implementation manifest references an absent native artifact")
                })
            })
            .collect()
    }
}
