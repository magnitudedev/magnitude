//! Demand-driven native realization and opaque handle ownership.
//!
//! Structural candidate domains contain no native artifacts. Preparation
//! hands an exact canonical coordinate to [`Realizer`], which compiles the
//! kernels the member launches. The private registry
//! owns each formed native handle; reconciled candidates retain their exact
//! ordered native set without exposing handles to evaluation or planning.

use crate::errors::PreparationError;
use crate::refinement::ConstructedCandidate;
use seismic_ir::kernel::KernelId;
use seismic_lang::expr::{AnyExpr, ExprArena, PartialAssignment, SymbolValue};
use seismic_target::{
    CompatibilityIdentity, DeviceDescription, DeviceDescriptionIdentity, NativeArtifactMetrics,
    NativeCompiler, NativeKernelDescription, TargetFamily,
};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;

/// Preparation-local exact cache key for a materialized family assignment.
/// The digest remains an external semantic label, but never decides whether
/// two resident requests can share a formed handle.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CandidateRealizationIdentity {
    family_materialization: u64,
    choices: Vec<(crate::refinement::PhysicalChoice, i64)>,
    assignment: [u8; 32],
}

/// An exact family assignment from the checked structural domain.
///
/// The request owns the family and assignment. This lets reconciliation
/// retain the family in its coordinate-exact output.
pub(crate) struct CanonicalRealizationRequest<T: TargetFamily> {
    identity: CandidateRealizationIdentity,
    family: Arc<ConstructedCandidate<T>>,
    assignment: PartialAssignment,
}

impl<T: TargetFamily> CanonicalRealizationRequest<T> {
    /// Consume the domain's coherent checked selection without reconstructing
    /// or revalidating its family/coordinate relationship.
    pub(crate) fn from_checked_candidate(
        read: &crate::candidate_domain::ConstructedCandidateRead<'_, T>,
    ) -> Self {
        let checked = read.candidate();
        let arena = read.arena();
        let family = checked.family().clone();
        let coordinate = checked.coordinate();
        let mut fixed = PartialAssignment::new();
        for (decision, value) in coordinate
            .decision_choices(family.choices())
            .expect("checked physical axes")
        {
            fixed.bind(
                arena.decision_symbol(decision),
                SymbolValue::Int((value).into()),
            );
        }

        let mut digest = Sha256::new();
        digest.update(b"seismic-canonical-realization-assignment-v1");
        for (ordinal, (_, value)) in coordinate.choices().iter().enumerate() {
            digest.update((ordinal as u64).to_le_bytes());
            digest.update(value.to_le_bytes());
        }
        let identity = CandidateRealizationIdentity {
            family_materialization: family.materialization_id(),
            choices: coordinate.choices().to_vec(),
            assignment: digest.finalize().into(),
        };
        Self {
            identity,
            family,
            assignment: fixed,
        }
    }
}

/// Stable lookup key for one emitted kernel template. Coordinate choices do
/// not enter this key: two coordinates reuse an artifact exactly when they
/// select the same original family kernel under the same native compatibility
/// contract.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeArtifactRequestKey {
    compatibility: CompatibilityIdentity,
    family_materialization: u64,
    kernel_ordinal: u32,
}

/// Preparation-local identity of one successfully formed native handle.
/// Matching image digests and reflection do not merge separate formations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NativeArtifactInstanceId(usize);

/// Read-only native artifact requirements of a candidate, derived exactly as
/// realization derives them.
/// Missing units have unknown formation cost; this does not initiate work.
#[derive(Clone, Debug)]
pub struct NativeRequirements {
    pub required: Vec<NativeArtifactRequestKey>,
    pub missing: Vec<NativeArtifactRequestKey>,
    pub rejected: bool,
}

/// A deterministic, candidate-specific post-reflection rejection. These are
/// cached. NativeCompilationError is deliberately absent and always remains a
/// retryable preparation/service failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateRejection {
    reason: String,
}

impl CandidateRejection {
    pub(crate) fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// Every kernel a member launches, in first-launch order without duplicates.
/// Each of these kernels is realized.
fn member_kernels<T: TargetFamily>(family: &ConstructedCandidate<T>) -> Vec<KernelId> {
    let mut kernels = Vec::new();
    for launch in family.schedule().launches() {
        if !kernels.contains(&launch.kernel) {
            kernels.push(launch.kernel);
        }
    }
    kernels
}

/// Handle-free dense native set. `kernels` is in first-launch order and
/// retains each original KernelId, which is the exact remap later executable
/// translation consumes.
#[derive(Debug)]
pub(crate) struct RealizedNativeSet<T: TargetFamily> {
    identity: CandidateRealizationIdentity,
    kernels: Box<[RealizedKernel<T>]>,
}

#[derive(Debug)]
struct RealizedKernel<T: TargetFamily> {
    original: KernelId,
    instance: NativeArtifactInstanceId,
    description: NativeKernelDescription<T>,
}

impl<T: TargetFamily> RealizedNativeSet<T> {
    pub(crate) fn assignment_identity(&self) -> [u8; 32] {
        self.identity.assignment
    }

    pub(crate) fn descriptions(
        &self,
    ) -> impl ExactSizeIterator<Item = &NativeKernelDescription<T>> {
        self.kernels.iter().map(|kernel| &kernel.description)
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
            .map(|index| u32::try_from(index).expect("native kernel count exceeds u32"))
    }

    pub(crate) fn retained_metadata_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            .saturating_add(std::mem::size_of_val(self.kernels.as_ref()))
            .saturating_add(
                self.identity.choices.capacity()
                    * std::mem::size_of::<(crate::refinement::PhysicalChoice, i64)>(),
            )
    }

    pub(crate) fn artifact_instances(
        &self,
    ) -> impl ExactSizeIterator<Item = NativeArtifactInstanceId> + '_ {
        self.kernels.iter().map(|kernel| kernel.instance)
    }
}

#[cfg(test)]
pub(crate) mod demand_driven_tests {
    use super::*;
    use crate::numerics::NumericalApplicability;
    use crate::refinement::{
        ChoiceDeclaration, ConstructedCandidateIdentity, ConstructedCandidateParts,
        ImplementationProvenance,
    };
    use seismic_lang::expr::DecisionId;
    use seismic_ir::construction::{AllocationPlan, Construction};
    use seismic_ir::kernel::{Kernel, KernelId};
    use seismic_ir::target::{
        IntrinsicIdentityBuilder, IntrinsicNumericalSemantics, KernelAbiFootprint, KernelAbiLayout,
        KernelAbiModel, KernelEmissionLayout, LocalRealization, LocalRealizationPolicy,
        NumericalEnvironment, TargetLimits, VectorSupport,
    };
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::entry::ElementBindings;
    use seismic_lang::expr::{FiniteDomain, PartialAssignment, TargetPredicate};
    use seismic_lang::registry::{BackendName, IntrinsicSignature};
    use seismic_target::NativeKernelIdentity;
    use seismic_target::{
        CompatibilityIdentity, DeviceDescriptionParts, NativeClusterDomain, NativeCompilationError,
        NativeKernelReflection, NativeLaunchDomain, NativeNumericalModeIdentity,
        NativeResourceUsage, NativeResources, NumericalEnvironmentIdentity,
    };
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[derive(Debug)]
    pub(crate) struct FakeTarget;

    #[derive(Clone, Debug, PartialEq)]
    pub(crate) struct FakeAbi;

    #[derive(Clone, Debug)]
    pub(crate) enum NoIntrinsic {}

    impl seismic_ir::target::PhysicalDialect for FakeTarget {
        type LaunchDescriptor = bool;
        fn ordinary_launch() -> Self::LaunchDescriptor {
            false
        }

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
        type NativeNumericalMode = ();
        type NativeProperties = ();
    }

    pub(crate) struct CountingCompiler {
        forms: AtomicUsize,
        fail_next: AtomicBool,
        fail_on_call: AtomicUsize,
        same_digest: AtomicBool,
        varying_reflection: AtomicBool,
        pub(crate) reject_launch_descriptor: AtomicBool,
    }

    impl CountingCompiler {
        pub(crate) fn new() -> Self {
            Self {
                forms: AtomicUsize::new(0),
                fail_next: AtomicBool::new(false),
                fail_on_call: AtomicUsize::new(0),
                same_digest: AtomicBool::new(false),
                varying_reflection: AtomicBool::new(false),
                reject_launch_descriptor: AtomicBool::new(false),
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
            if self.fail_next.swap(false, Ordering::SeqCst)
                || self.fail_on_call.load(Ordering::SeqCst) == ordinal
            {
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
            if !self.same_digest.load(Ordering::SeqCst) {
                digest[..8].copy_from_slice(&(candidate as u64).to_le_bytes());
            }
            Ok(NativeKernelReflection::new(
                candidate,
                NativeKernelDescription {
                    identity: NativeKernelIdentity {
                        compatibility: target.compatibility_identity().clone(),
                        artifact_digest: digest,
                    },
                    abi: target.kernel_abi_layout(kernel),
                    launch: NativeLaunchDomain {
                        modes: vec![self.reject_launch_descriptor.load(Ordering::SeqCst)],
                        subgroup_width: None,
                        cluster: NativeClusterDomain::NotApplicable,
                        max_grid: [1, 1, 1],
                        max_workgroup_size: [1, 1, 1],
                        max_workgroup_threads: 1,
                        max_dynamic_local_bytes: 0,
                    },
                    resources: NativeResources {
                        registers_per_participant: if self.varying_reflection.load(Ordering::SeqCst)
                        {
                            NativeResourceUsage::Exact(candidate as u64)
                        } else {
                            NativeResourceUsage::NotApplicable
                        },
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

    /// Accepts every candidate and retains its realized native set, which the
    /// tests inspect through the reconciled output.
    struct Accept;
    impl NativeCandidateReconciler<FakeTarget> for Accept {
        type Output = Arc<RealizedNativeSet<FakeTarget>>;

        fn reconcile(
            &mut self,
            _: &mut ExprArena,
            input: ReconciliationInput<FakeTarget>,
        ) -> Result<Self::Output, NativeReconciliationError> {
            Ok(input.into_parts().2)
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
        DeviceDescription::new(device_parts()).unwrap()
    }

    pub(crate) fn registry() -> crate::target::CompilerRegistry<FakeTarget> {
        crate::target::CompilerRegistry::assemble(crate::target::CompilerRegistryParts {
            capabilities: Vec::new(),

            native_launch_constraints: |_, _, _, _, _, _| Vec::new(),
            addressable_resources: |_| Vec::new(),
            emitted_intrinsics: Default::default(),
        })
    }

    pub(crate) fn device_parts() -> DeviceDescriptionParts<FakeTarget> {
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
        }
    }

    pub(crate) fn choice_family(
        arena: &mut ExprArena,
    ) -> (
        Arc<ConstructedCandidate<FakeTarget>>,
        DecisionId,
        [KernelId; 2],
    ) {
        let mut construction = Construction::<FakeTarget>::new(arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let first = construction
            .portable_kernel(arena, &(), &[], &vectors)
            .close();
        let shared = construction
            .portable_kernel(arena, &(), &[], &vectors)
            .close();
        let decision = arena.decision(FiniteDomain::new(vec![0, 1]).unwrap());
        let mut schedule = construction.schedule(arena, 0);
        schedule.launch_sequential(first);
        schedule.launch_sequential(shared);
        let token = schedule.close();
        let executable = construction
            .close(token)
            .normalize_launches(arena, u64::MAX, 64)
            .unwrap()
            .analyze_allocations()
            .apply_allocation_plan(arena, AllocationPlan::distinct())
            .finish()
            .close_execution(arena, device().local_realization(), &FakeAbi);
        let always = arena.bool(true);
        let coverage = TargetPredicate::new(arena, always).unwrap();
        let family = Arc::new(ConstructedCandidate::test_from_parts(
            ConstructedCandidateParts {
                identity: family_identity(),
                semantic_coverage: coverage,
                native_index_bits: device().limits().max_index_bits,
                executable,
                choices: vec![ChoiceDeclaration {
                    kind: crate::refinement::ChoiceKind::WorkgroupSize,
                    decision,
                    meaning: "selected branch",
                    active_when: always,
                }],
                hard_constraints: always,
                numerical_applicability: NumericalApplicability::unresolved_fixture(),
                provenance: ImplementationProvenance {
                    root: fixture_function_identity(),
                    callees: Vec::new(),
                },
                bindings: crate::portable::FrozenBindings::default(),
                result_publications: Vec::new(),
            },
        ));
        (family, decision, [first, shared])
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
        family: Arc<ConstructedCandidate<FakeTarget>>,
        identity: u8,
        assignment: PartialAssignment,
    ) -> CanonicalRealizationRequest<FakeTarget> {
        CanonicalRealizationRequest {
            identity: CandidateRealizationIdentity {
                family_materialization: family.materialization_id(),
                choices: vec![(
                    crate::refinement::PhysicalChoice {
                        ordinal: 0,
                        kind: crate::refinement::ChoiceKind::WorkgroupSize,
                    },
                    identity as i64,
                )],
                assignment: [identity; 32],
            },
            family,
            assignment,
        }
    }

    fn assignment(arena: &ExprArena, decision: DecisionId, value: i64) -> PartialAssignment {
        let mut assignment = PartialAssignment::new();
        assignment.bind(
            arena.decision_symbol(decision),
            SymbolValue::Int((value).into()),
        );
        assignment
    }

    fn family_identity() -> ConstructedCandidateIdentity {
        ConstructedCandidateIdentity { structure: [9; 32] }
    }

    #[test]
    fn inspection_is_read_only_and_reports_exact_shared_units() {
        let mut arena = ExprArena::new();
        let (family, decision, _) = choice_family(&mut arena);
        let target = device();
        let compiler = CountingCompiler::new();
        let mut realizer = Realizer::new(&compiler, &(), &target, Accept);
        let zero = request(family.clone(), 0, assignment(&arena, decision, 0));
        let one = request(family.clone(), 1, assignment(&arena, decision, 1));
        let before = realizer.inspect(&arena, &zero);
        assert_eq!(before.required.len(), 2);
        assert_eq!(before.missing, before.required);
        assert!(!before.rejected);
        assert_eq!(compiler.form_count(), 0);
        realizer.realize(&mut arena, zero).unwrap();
        let zero = request(family, 0, assignment(&arena, decision, 0));
        assert!(realizer.inspect(&arena, &zero).missing.is_empty());
        let other = realizer.inspect(&arena, &one);
        assert_eq!(other.required, before.required);
        assert!(other.missing.is_empty());
        assert_eq!(compiler.form_count(), 2);
        realizer.realize(&mut arena, one).unwrap();
        assert_eq!(compiler.form_count(), 2);
    }

    #[test]
    fn equal_family_digests_do_not_share_native_requests_across_materializations() {
        let mut arena = ExprArena::new();
        let (first_family, first_decision, _) = choice_family(&mut arena);
        let (second_family, second_decision, _) = choice_family(&mut arena);
        assert_eq!(first_family.identity(), second_family.identity());
        assert_ne!(
            first_family.materialization_id(),
            second_family.materialization_id()
        );
        let target = device();
        let compiler = CountingCompiler::new();
        let mut realizer = Realizer::new(&compiler, &(), &target, Accept);
        let first = request(first_family, 0, assignment(&arena, first_decision, 0));
        let second = request(second_family, 0, assignment(&arena, second_decision, 0));
        let RealizationOutcome::Ready {
            candidate: first, ..
        } = realizer.realize(&mut arena, first).unwrap()
        else {
            panic!("first family should be ready")
        };
        let RealizationOutcome::Ready {
            candidate: second, ..
        } = realizer.realize(&mut arena, second).unwrap()
        else {
            panic!("second family should be ready")
        };
        assert_eq!(compiler.form_count(), 4);
        assert_ne!(
            first.reconciled().as_ref().artifact_instances().collect::<Vec<_>>(),
            second.reconciled().as_ref().artifact_instances().collect::<Vec<_>>()
        );
    }

    #[test]
    fn equal_assignment_digests_do_not_share_distinct_choice_vectors() {
        let mut arena = ExprArena::new();
        let (family, decision, _) = choice_family(&mut arena);
        let target = device();
        let compiler = CountingCompiler::new();
        let mut realizer = Realizer::new(&compiler, &(), &target, Accept);
        let zero = request(family.clone(), 0, assignment(&arena, decision, 0));
        let mut one = request(family, 1, assignment(&arena, decision, 1));
        one.identity.assignment = zero.identity.assignment;

        let RealizationOutcome::Ready {
            candidate: zero, ..
        } = realizer.realize(&mut arena, zero).unwrap()
        else {
            panic!("zero choice should be ready")
        };
        let RealizationOutcome::Ready { candidate: one, .. } =
            realizer.realize(&mut arena, one).unwrap()
        else {
            panic!("one choice should be ready")
        };
        assert_eq!(compiler.form_count(), 2);
        assert!(!Arc::ptr_eq(&zero, &one));
    }

    #[test]
    fn realization_is_selected_only_cached_and_retries_infrastructure_failure() {
        let mut arena = ExprArena::new();
        let (family, decision, [first, shared]) = choice_family(&mut arena);
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
        let native = candidate.reconciled().as_ref();
        assert_eq!(compiler.form_count(), 2);
        assert_eq!(native.native_kernel_index(first), Some(0));
        assert_eq!(native.native_kernel_index(shared), Some(1));
        let shared_instance = native.artifact_instances().nth(1).unwrap();

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
        let selected_native = candidate.reconciled().as_ref().clone();
        let native = selected_native.as_ref();
        assert_eq!(compiler.form_count(), 2, "formed kernels must be reused");
        assert_eq!(native.native_kernel_index(first), Some(0));
        assert_eq!(native.native_kernel_index(shared), Some(1));
        assert_eq!(native.artifact_instances().nth(1), Some(shared_instance));

        assert_eq!(realizer.registry().resolve(target.identity(), native).len(), 2);

        let retry_compiler = CountingCompiler::new();
        retry_compiler.fail_next.store(true, Ordering::SeqCst);
        let mut retry = Realizer::new(&retry_compiler, &(), &target, Accept);
        let failed = request(family.clone(), 0, zero_assignment.clone());
        assert!(matches!(
            retry.realize(&mut arena, failed),
            Err(RealizationFailure {
                error: PreparationError::NativeCompilation(
                    NativeCompilationError::ToolchainFailure(_)
                ),
                ..
            })
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
    fn later_kernel_failure_reports_earlier_formation_and_reuses_its_instance() {
        let mut arena = ExprArena::new();
        let (family, decision, _) = choice_family(&mut arena);
        let target = device();
        let compiler = CountingCompiler::new();
        compiler.fail_on_call.store(2, Ordering::SeqCst);
        let mut realizer = Realizer::new(&compiler, &(), &target, Accept);
        let fixed = assignment(&arena, decision, 0);
        let first = request(family.clone(), 0, fixed.clone());
        let failure = match realizer.realize(&mut arena, first) {
            Err(failure) => failure,
            Ok(_) => panic!("second kernel formation should fail"),
        };
        assert!(matches!(
            failure.error,
            PreparationError::NativeCompilation(NativeCompilationError::ToolchainFailure(_))
        ));
        assert_eq!(failure.newly_formed.code_bytes, 1);
        assert_eq!(failure.newly_formed.metadata_bytes, 1);
        assert_eq!(compiler.form_count(), 2);

        let retry = request(family, 0, fixed);
        assert_eq!(realizer.inspect(&arena, &retry).missing.len(), 1);
        let RealizationOutcome::Ready {
            newly_formed,
            candidate,
        } = realizer.realize(&mut arena, retry).unwrap()
        else {
            panic!("retry should succeed")
        };
        assert_eq!(newly_formed.code_bytes, 1);
        assert_eq!(compiler.form_count(), 3);
        assert_eq!(candidate.reconciled().as_ref().artifact_instances().count(), 2);
    }

    #[test]
    fn matching_native_digests_preserve_distinct_formed_handles() {
        let mut arena = ExprArena::new();
        let (family, decision, _) = choice_family(&mut arena);
        let target = device();
        let compiler = CountingCompiler::new();
        compiler.same_digest.store(true, Ordering::SeqCst);
        let mut realizer = Realizer::new(&compiler, &(), &target, Accept);
        let request = request(family, 0, assignment(&arena, decision, 0));
        let RealizationOutcome::Ready { candidate, .. } =
            realizer.realize(&mut arena, request).unwrap()
        else {
            panic!("candidate should be ready")
        };
        let native = candidate.reconciled().as_ref();
        let instances = native.artifact_instances().collect::<Vec<_>>();
        assert_eq!(instances.len(), 2);
        assert_ne!(instances[0], instances[1]);
        let descriptions = native.descriptions().collect::<Vec<_>>();
        assert_eq!(descriptions[0], descriptions[1]);
        let handles = realizer
            .registry()
            .resolve(target.identity(), native)
            .into_iter()
            .map(|kernel| *kernel.handle())
            .collect::<Vec<_>>();
        assert_eq!(handles, vec![1, 2]);
        assert_eq!(realizer.registry().retained_sizes(instances), (2, 2));
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

pub(crate) struct RealizedCandidate<O> {
    reconciled: Arc<O>,
}

impl<O> RealizedCandidate<O> {
    pub(crate) fn reconciled(&self) -> &Arc<O> {
        &self.reconciled
    }
}

pub(crate) enum RealizationOutcome<O> {
    Ready {
        candidate: Arc<RealizedCandidate<O>>,
        newly_formed: NativeArtifactMetrics,
    },
    Rejected {
        rejection: Arc<CandidateRejection>,
        newly_formed: NativeArtifactMetrics,
    },
}

/// A failed candidate attempt may already have formed and retained earlier
/// kernels. Return their resource use so preparation cannot silently lose it.
#[derive(Debug)]
pub(crate) struct RealizationFailure {
    pub(crate) error: PreparationError,
    pub(crate) newly_formed: NativeArtifactMetrics,
}

enum CachedCandidate<O> {
    Ready(Arc<RealizedCandidate<O>>),
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
    family: Arc<ConstructedCandidate<T>>,
    assignment: PartialAssignment,
    native: Arc<RealizedNativeSet<T>>,
}

impl<T: TargetFamily> ReconciliationInput<T> {
    pub(crate) fn into_parts(
        self,
    ) -> (
        Arc<ConstructedCandidate<T>>,
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
    candidates: HashMap<CandidateRealizationIdentity, CachedCandidate<R::Output>>,
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

    pub(crate) fn inspect(
        &self,
        arena: &ExprArena,
        request: &CanonicalRealizationRequest<T>,
    ) -> NativeRequirements {
        arena.view(AnyExpr::Bool(request.family.hard_constraints()));
        let required = member_kernels(&request.family)
            .into_iter()
            .map(|original| NativeArtifactRequestKey {
                compatibility: self.target.compatibility_identity().clone(),
                family_materialization: request.identity.family_materialization,
                kernel_ordinal: original.ordinal(),
            })
            .collect::<Vec<_>>();
        let missing = required
            .iter()
            .filter(|key| !self.registry.artifact_requests.contains_key(*key))
            .cloned()
            .collect();
        NativeRequirements {
            required,
            missing,
            rejected: matches!(
                self.candidates.get(&request.identity),
                Some(CachedCandidate::Rejected(_))
            ),
        }
    }

    /// Realizes every kernel the member launches.
    /// Reconciliation is the sole post-reflection eligibility transition:
    /// returning Rejected caches that deterministic result, while every
    /// native infrastructure error returns before the candidate cache changes.
    pub(crate) fn realize(
        &mut self,
        arena: &mut ExprArena,
        request: CanonicalRealizationRequest<T>,
    ) -> Result<RealizationOutcome<R::Output>, RealizationFailure> {
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
            let originals = member_kernels(&family);
            let mut formed = empty_metrics();
            let mut realized = Vec::with_capacity(originals.len());
            for original in originals {
                let kernel = family.executable().kernels().kernel(original);
                let key = NativeArtifactRequestKey {
                    compatibility: self.target.compatibility_identity().clone(),
                    family_materialization: identity.family_materialization,
                    kernel_ordinal: original.ordinal(),
                };
                let instance = if let Some(instance) = self.registry.artifact_requests.get(&key) {
                    *instance
                } else {
                    let (native, metrics) = crate::implementation::native::realize_kernel(
                        self.compiler,
                        self.context,
                        kernel,
                        self.target,
                    )
                    .map_err(|error| RealizationFailure {
                        error,
                        newly_formed: formed,
                    })?;
                    add_metrics(&mut formed, metrics).map_err(|error| RealizationFailure {
                        error,
                        newly_formed: formed,
                    })?;
                    let instance = self
                        .registry
                        .artifacts
                        .retain_one(self.target.compatibility_identity(), native, metrics)
                        .map_err(|error| RealizationFailure {
                            error,
                            newly_formed: formed,
                        })?;
                    self.registry.artifact_requests.insert(key, instance);
                    instance
                };
                let description = self
                    .registry
                    .artifacts
                    .description(instance)
                    .unwrap_or_else(|| panic!("artifact request cache references absent storage"))
                    .clone();
                realized.push(RealizedKernel {
                    original,
                    instance,
                    description,
                });
            }
            let native = Arc::new(RealizedNativeSet {
                identity: identity.clone(),
                kernels: realized.into_boxed_slice(),
            });
            (native, formed)
        };
        let reconciliation = ReconciliationInput {
            family,
            assignment,
            native,
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
            Err(NativeReconciliationError::Preparation(error)) => {
                return Err(RealizationFailure {
                    error,
                    newly_formed: formed,
                });
            }
        };
        let candidate = Arc::new(RealizedCandidate {
            reconciled: Arc::new(reconciled),
        });
        let replaced = self
            .candidates
            .insert(identity, CachedCandidate::Ready(candidate.clone()));
        assert!(
            replaced.is_none(),
            "cached candidate changed after realization"
        );
        Ok(RealizationOutcome::Ready {
            candidate,
            newly_formed: formed,
        })
    }

    pub(crate) fn registry(&self) -> &RealizationRegistry<T, C::Handle> {
        &self.registry
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
    let compilation_ns = total
        .compilation_ns
        .checked_add(artifact.compilation_ns)
        .ok_or_else(|| overflow("compilation duration in nanoseconds"))?;
    let code_bytes = total
        .code_bytes
        .checked_add(artifact.code_bytes)
        .ok_or_else(|| overflow("code size in bytes"))?;
    let metadata_bytes = total
        .metadata_bytes
        .checked_add(artifact.metadata_bytes)
        .ok_or_else(|| overflow("metadata size in bytes"))?;
    *total = NativeArtifactMetrics {
        compilation_ns,
        code_bytes,
        metadata_bytes,
    };
    Ok(())
}

/// The sole owner of candidate-native handles before materialization.
///
/// `artifacts` retains formed native instances and `artifact_requests` owns
/// canonical request reuse. Candidates carry their reconciled native set.
#[derive(Debug)]
pub(crate) struct RealizationRegistry<T: TargetFamily, H> {
    device: DeviceDescriptionIdentity,
    artifacts: NativeArtifactStore<T, H>,
    artifact_requests: HashMap<NativeArtifactRequestKey, NativeArtifactInstanceId>,
}

/// Every formed request has a resident handle. Request-cache reuse shares
/// that instance without treating artifact digests as handle identity.
#[derive(Debug)]
struct NativeArtifactStore<T: TargetFamily, H> {
    by_instance: Vec<ResidentArtifact<T, H>>,
}

#[derive(Debug)]
struct ResidentArtifact<T: TargetFamily, H> {
    kernel: Arc<seismic_target::NativeKernel<T, H>>,
    metrics: NativeArtifactMetrics,
}

impl<T: TargetFamily, H> NativeArtifactStore<T, H> {
    fn new() -> Self {
        Self {
            by_instance: Vec::new(),
        }
    }

    fn retain_one(
        &mut self,
        compatibility: &CompatibilityIdentity,
        kernel: seismic_target::NativeKernel<T, H>,
        metrics: NativeArtifactMetrics,
    ) -> Result<NativeArtifactInstanceId, PreparationError> {
        let description = kernel.description();
        if &description.identity.compatibility != compatibility {
            return Err(PreparationError::InvalidCandidateDomain(format!(
                "constructed implementation contains a native artifact for another compatibility domain"
            )));
        }
        let instance = NativeArtifactInstanceId(self.by_instance.len());
        self.by_instance.push(ResidentArtifact {
            kernel: Arc::new(kernel),
            metrics,
        });
        Ok(instance)
    }

    fn get(
        &self,
        instance: NativeArtifactInstanceId,
    ) -> Option<Arc<seismic_target::NativeKernel<T, H>>> {
        self.by_instance
            .get(instance.0)
            .map(|artifact| artifact.kernel.clone())
    }

    fn description(
        &self,
        instance: NativeArtifactInstanceId,
    ) -> Option<&NativeKernelDescription<T>> {
        self.by_instance
            .get(instance.0)
            .map(|artifact| artifact.kernel.description())
    }
}

impl<T: TargetFamily, H> RealizationRegistry<T, H> {
    /// Union of the policy's native instances, excluding resident search losers
    /// and counting request-cache reuse once. Formation time is not retained storage.
    pub(crate) fn retained_sizes<'a>(
        &self,
        instances: impl IntoIterator<Item = NativeArtifactInstanceId>,
    ) -> (u64, u64) {
        let mut seen = std::collections::HashSet::new();
        let mut code = 0u64;
        let mut metadata = 0u64;
        for instance in instances {
            if seen.insert(instance) {
                let artifact = self
                    .artifacts
                    .by_instance
                    .get(instance.0)
                    .expect("retained policy references a realized native artifact");
                code = code.saturating_add(artifact.metrics.code_bytes);
                metadata = metadata.saturating_add(artifact.metrics.metadata_bytes);
            }
        }
        (code, metadata)
    }

    pub(crate) fn new(device: DeviceDescriptionIdentity) -> Self {
        Self {
            device,
            artifacts: NativeArtifactStore::new(),
            artifact_requests: HashMap::new(),
        }
    }

    pub(crate) fn resolve(
        &self,
        device: &DeviceDescriptionIdentity,
        native: &RealizedNativeSet<T>,
    ) -> Vec<Arc<seismic_target::NativeKernel<T, H>>> {
        assert_eq!(
            device, &self.device,
            "planned policy and realization registry have different devices"
        );
        native
            .kernels
            .iter()
            .map(|realized| {
                let request = NativeArtifactRequestKey {
                    compatibility: realized.description.identity.compatibility.clone(),
                    family_materialization: native.identity.family_materialization,
                    kernel_ordinal: realized.original.ordinal(),
                };
                assert_eq!(
                    self.artifact_requests.get(&request),
                    Some(&realized.instance),
                    "native candidate does not belong to this realization registry"
                );
                let kernel = self
                    .artifacts
                    .get(realized.instance)
                    .expect("native candidate references an absent artifact");
                assert_eq!(
                    kernel.description(),
                    &realized.description,
                    "native instance differs from its reconciled reflection"
                );
                kernel
            })
            .collect()
    }
}
