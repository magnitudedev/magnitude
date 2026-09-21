//! Sole production compilation seam (frozen by C0; package P1 owns the
//! planning call it makes).
//!
//! The compiler calls core `form_plan_space` exactly once, passes the sealed
//! space to the solver, resolves exactly one complete assignment, then
//! performs exhaustive encoding and native assembly. Backends supply a
//! profile, a mapping catalog, an exhaustive mechanical encoder over sealed
//! launches, and a native assembler. There is no `Backend::elaborate`, no
//! retry, fallback, repair, or alternate selector.

use crate::planning::{self, Budget};
use seismic_lang::{
    logical::{
        self, specialization::SpecializationDomain, ApplicabilityReport, EffectiveTargetIdentity,
        LogicalProgram,
    },
    precision::PrecisionPolicy,
    sir::{IntrinsicUse, Program},
};
use seismic_realization::{
    failure::CompilerDefect,
    ids::{BranchIx, CallIx, GuardIx, LaunchIx, RepeatIx, StorageIx},
    kernel::ExecutableDialect,
    numerics::NumericalEvidence,
    physical::{PhysicalPlan, PhysicalStep, SealedGuard, SealedLaunch},
    plan_space::NumericalContext,
    strategy::MappingCatalog,
    target::EffectiveTargetProfile,
};
use std::sync::Arc;

/// Semantic diagnostics of a program that cannot yield a valid
/// specialization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostics(pub Vec<String>);

/// Native toolchain failure report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolchainReport(pub String);

/// Host system failure report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemReport(pub String);

/// A backend's closed native-assembly failure taxonomy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssemblyFailure {
    CompilerInvariant(CompilerDefect),
    Toolchain(ToolchainReport),
    SystemPreparation(SystemReport),
}

impl std::fmt::Display for AssemblyFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CompilerInvariant(defect) => write!(f, "{defect} during assembly"),
            Self::Toolchain(report) => write!(f, "native toolchain failure: {}", report.0),
            Self::SystemPreparation(report) => {
                write!(f, "native system preparation failed: {}", report.0)
            }
        }
    }
}

impl std::error::Error for AssemblyFailure {}

/// The closed production failure taxonomy. Exactly one of these (or a
/// compiled artifact) is the result of compiling a checked program.
#[derive(Debug)]
pub enum CompileFailure {
    InvalidSemanticProgram(Diagnostics),
    NoApplicableImplementation(ApplicabilityReport),
    PlanningInfeasible,
    CompilerBug(CompilerDefect),
    ToolchainFailure(ToolchainReport),
    SystemFailure(SystemReport),
}

impl std::fmt::Display for CompileFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSemanticProgram(diagnostics) => {
                write!(f, "invalid semantic program: {}", diagnostics.0.join("; "))
            }
            Self::NoApplicableImplementation(report) => {
                write!(f, "no applicable implementation for entry `{}`", report.entry)?;
                for rejection in &report.occurrences {
                    write!(f, "; {rejection:?}")?;
                }
                Ok(())
            }
            Self::PlanningInfeasible => write!(f, "planning is infeasible"),
            Self::CompilerBug(defect) => write!(f, "{defect}"),
            Self::ToolchainFailure(report) => write!(f, "toolchain failure: {}", report.0),
            Self::SystemFailure(report) => write!(f, "system failure: {}", report.0),
        }
    }
}

impl std::error::Error for CompileFailure {}

/// The compiler backend seam. Formation and encoding are total because their
/// inputs are sealed; catalog construction validates that every declared
/// intrinsic family has an encoder.
pub trait Backend {
    type Dialect: ExecutableDialect;
    type Catalog: MappingCatalog<Self::Dialect>;
    type EncodedLaunch;
    type NativeArtifact;

    fn profile(&self) -> &EffectiveTargetProfile;
    fn catalog(&self) -> &Self::Catalog;
    /// Applicability of one exact typed capability use on this target.
    fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String>;
    /// Exhaustive mechanical encoding of one sealed launch. Total.
    fn encode(&self, launch: &SealedLaunch<Self::Dialect>) -> Self::EncodedLaunch;
    /// Compile every encoded launch, reflect native facts, validate them
    /// against the selected contract, and seal one structured native tree.
    fn assemble(
        &self,
        plan: EncodedPlan<Self::Dialect, Self::EncodedLaunch>,
    ) -> Result<Self::NativeArtifact, AssemblyFailure>;
}

/// Encoded mirror of the sealed physical schedule: every `PhysicalStep`
/// exactly once, launches paired with their encoded form. Construction is
/// private to `encode_plan`, which builds the tree from the plan it pairs;
/// the mirroring invariant is asserted at that one construction site.
pub struct EncodedPlan<D: ExecutableDialect, L> {
    physical: Arc<PhysicalPlan<D>>,
    steps: Vec<EncodedStep<L>>,
}

impl<D: ExecutableDialect, L> EncodedPlan<D, L> {
    /// The sealed physical plan this encoded tree mirrors.
    pub fn physical(&self) -> &Arc<PhysicalPlan<D>> {
        &self.physical
    }

    /// The encoded tree, mirroring `physical().schedule()` exactly.
    pub fn steps(&self) -> &[EncodedStep<L>] {
        &self.steps
    }

    /// The only constructor: pairs an encoded tree with the plan it was
    /// encoded from, asserting that the tree mirrors the plan's schedule
    /// step for step.
    pub(crate) fn seal(
        physical: Arc<PhysicalPlan<D>>,
        steps: Vec<EncodedStep<L>>,
    ) -> EncodedPlan<D, L> {
        debug_assert!(mirrors(&steps, &physical.schedule().steps));
        EncodedPlan { physical, steps }
    }
}

/// Whether an encoded tree pairs with the physical schedule it claims to
/// mirror: identical step structure, identical dense ids, launches encoded
/// in tree order.
fn mirrors<D: ExecutableDialect, L>(
    encoded: &[EncodedStep<L>],
    physical: &[PhysicalStep<D>],
) -> bool {
    if encoded.len() != physical.len() {
        return false;
    }
    for (encoded, physical) in encoded.iter().zip(physical) {
        let paired = match (encoded, physical) {
            (
                EncodedStep::Launch { launch, .. },
                PhysicalStep::Launch(sealed),
            ) => *launch == sealed.id,
            (EncodedStep::Guard { guard }, PhysicalStep::Guard(sealed)) => *guard == sealed.id,
            (
                EncodedStep::Call { call, body },
                PhysicalStep::Call(sealed),
            ) => *call == sealed.id && mirrors(body, &sealed.body.steps),
            (
                EncodedStep::If {
                    branch,
                    then_steps,
                    else_steps,
                },
                PhysicalStep::If(sealed),
            ) => {
                *branch == sealed.id
                    && mirrors(then_steps, &sealed.then_schedule.steps)
                    && mirrors(else_steps, &sealed.else_schedule.steps)
            }
            (
                EncodedStep::Repeat { repeat, body },
                PhysicalStep::Repeat(sealed),
            ) => *repeat == sealed.id && mirrors(body, &sealed.body.steps),
            (EncodedStep::Fill { storage }, PhysicalStep::Fill(sealed)) => {
                *storage == sealed.storage
            }
            _ => false,
        };
        if !paired {
            return false;
        }
    }
    true
}

pub enum EncodedStep<L> {
    Launch { launch: LaunchIx, encoded: L },
    Guard { guard: GuardIx },
    Call { call: CallIx, body: Vec<EncodedStep<L>> },
    If { branch: BranchIx, then_steps: Vec<EncodedStep<L>>, else_steps: Vec<EncodedStep<L>> },
    Repeat { repeat: RepeatIx, body: Vec<EncodedStep<L>> },
    /// A host-side fill (the pull-counter reset): present in the encoded
    /// tree so it mirrors the physical schedule exactly once, but never
    /// backend-encoded — the host zeroes the storage before the launch.
    Fill { storage: StorageIx },
}

/// The compiled artifact: logical program, sealed physical plan, and native
/// artifact. `CompiledPlan` (runtime, package R1) wraps it with its
/// invocation contract.
pub struct Compiled<D: ExecutableDialect, A> {
    pub logical: Arc<LogicalProgram>,
    pub physical: Arc<PhysicalPlan<D>>,
    pub native: A,
}

/// Compile one entry under one complete specialization domain and precision
/// policy for one backend.
pub fn compile<B: Backend>(
    program: &Program,
    domain: &SpecializationDomain,
    precision: &PrecisionPolicy,
    backend: &B,
    numerical_evidence: &[NumericalEvidence],
    budget: Budget,
) -> Result<Compiled<B::Dialect, B::NativeArtifact>, CompileFailure> {
    let profile = backend.profile();
    let target = EffectiveTargetIdentity {
        backend: profile.backend.clone(),
        capability_fingerprint: profile.capability_fingerprint.clone(),
    };
    let supports = |intrinsic: &IntrinsicUse| backend.supports_intrinsic(intrinsic);
    let logical = logical::construct(program, &target, &supports, domain).map_err(|error| {
        match error {
            logical::LogicalConstructionError::NoApplicableImplementation(report) => {
                CompileFailure::NoApplicableImplementation(report)
            }
            logical::LogicalConstructionError::InvalidProgram(reason) => {
                CompileFailure::InvalidSemanticProgram(Diagnostics(vec![reason]))
            }
        }
    })?;
    let logical = Arc::new(logical);
    let space = seismic_realization::form_plan_space(&logical, profile, backend.catalog())
        .map_err(CompileFailure::CompilerBug)?;
    let numerics = NumericalContext {
        precision,
        evidence: numerical_evidence,
    };
    let assignment = planning::plan(&space, &numerics, budget).map_err(|error| match error {
        planning::PlanningFailure::Infeasible => CompileFailure::PlanningInfeasible,
        planning::PlanningFailure::CompilerBug(defect) => CompileFailure::CompilerBug(defect),
        planning::PlanningFailure::Solver(reason) => {
            CompileFailure::SystemFailure(SystemReport(reason))
        }
    })?;
    let physical = Arc::new(space.resolve(assignment));
    let encoded = encode_plan(Arc::clone(&physical), backend);
    let native = backend.assemble(encoded).map_err(|failure| match failure {
        AssemblyFailure::CompilerInvariant(defect) => CompileFailure::CompilerBug(defect),
        AssemblyFailure::Toolchain(report) => CompileFailure::ToolchainFailure(report),
        AssemblyFailure::SystemPreparation(report) => CompileFailure::SystemFailure(report),
    })?;
    Ok(Compiled {
        logical,
        physical,
        native,
    })
}

/// Encode every launch once, mirroring the sealed schedule exactly. Total.
fn encode_plan<B: Backend>(
    physical: Arc<PhysicalPlan<B::Dialect>>,
    backend: &B,
) -> EncodedPlan<B::Dialect, B::EncodedLaunch> {
    let steps = encode_steps(&physical.schedule().steps, backend);
    EncodedPlan::seal(physical, steps)
}

fn encode_steps<B: Backend>(
    steps: &[PhysicalStep<B::Dialect>],
    backend: &B,
) -> Vec<EncodedStep<B::EncodedLaunch>> {
    steps
        .iter()
        .map(|step| match step {
            PhysicalStep::Launch(launch) => EncodedStep::Launch {
                launch: launch.id,
                encoded: backend.encode(launch),
            },
            PhysicalStep::Guard(SealedGuard { id, .. }) => EncodedStep::Guard { guard: *id },
            PhysicalStep::Call(call) => EncodedStep::Call {
                call: call.id,
                body: encode_steps(&call.body.steps, backend),
            },
            PhysicalStep::If(branch) => EncodedStep::If {
                branch: branch.id,
                then_steps: encode_steps(&branch.then_schedule.steps, backend),
                else_steps: encode_steps(&branch.else_schedule.steps, backend),
            },
            PhysicalStep::Repeat(repeat) => EncodedStep::Repeat {
                repeat: repeat.id,
                body: encode_steps(&repeat.body.steps, backend),
            },
            PhysicalStep::Fill(fill) => EncodedStep::Fill {
                storage: fill.storage,
            },
        })
        .collect()
}
