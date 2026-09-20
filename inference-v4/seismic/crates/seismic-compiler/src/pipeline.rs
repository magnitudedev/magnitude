//! Sole production compilation seam.
//!
//! Backends construct complete plan families, encode one resolved launch at
//! a time, and assemble the hierarchy already encoded by the compiler core.
//! The pipeline consumes the one structured schedule authority: encoded
//! items mirror `ResolvedStep::{Launch, Call, If, Repeat}`, never a flattened
//! phase list. Failure is one closed taxonomy; there is no retry, fallback,
//! or repair path.

use crate::planning::{self, Budget};
use seismic_lang::{
    logical::{self, ApplicabilityReport, EffectiveTargetIdentity, LogicalProgram},
    precision::PrecisionPolicy,
    sir::{IntrinsicUse, Program},
    types::Elem,
};
use seismic_realization::executable::{
    EffectiveTargetProfile, ExecutableDialect, InvariantReport, PlanFamily, ResolvedLaunch,
    ResolvedPlan, ResolvedScheduleIf, ResolvedScheduleRepeat, ResolvedStep,
};
use std::{collections::BTreeMap, sync::Arc};

/// Concrete specialization of the entry: shape and element parameters plus
/// the caller's whole-program precision policy, and the expected runtime
/// value of `index` parameters that bound runtime domains.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Workload {
    pub shapes: BTreeMap<String, i64>,
    pub elems: BTreeMap<String, Elem>,
    pub precision: PrecisionPolicy,
    /// Expected runtime value of each `index` parameter used as a loop bound,
    /// keyed by entry parameter name. Prices runtime domains for planning;
    /// an unstated parameter leaves its domains priced at capacity. Never
    /// affects semantics, geometry, or resources.
    pub extents: BTreeMap<String, i64>,
}

/// Semantic diagnostics of a program that cannot yield a valid specialization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostics(pub Vec<String>);

/// Native toolchain failure report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolchainReport(pub String);

/// Host system failure report.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemReport(pub String);

/// The closed production failure taxonomy. Exactly one of these (or a
/// `ResolvedPlan`) is the result of compiling a checked program.
#[derive(Debug)]
pub enum CompileFailure {
    InvalidSemanticProgram(Diagnostics),
    NoApplicableImplementation(ApplicabilityReport),
    PlanningInfeasible,
    CompilerBug(InvariantReport),
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
                write!(
                    f,
                    "no applicable implementation for entry `{}`",
                    report.entry
                )?;
                for rejection in &report.occurrences {
                    write!(f, "; {:?}", rejection)?;
                }
                Ok(())
            }
            Self::PlanningInfeasible => write!(f, "planning is infeasible"),
            Self::CompilerBug(report) => write!(f, "compiler bug: {}", report.0),
            Self::ToolchainFailure(report) => write!(f, "toolchain failure: {}", report.0),
            Self::SystemFailure(report) => write!(f, "system failure: {}", report.0),
        }
    }
}

impl std::error::Error for CompileFailure {}

pub trait Backend {
    type Dialect: ExecutableDialect;
    type EncodedLaunch;
    type NativeArtifact;

    fn target(&self) -> &'static str;
    fn capability_fingerprint(&self) -> String;
    fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String>;
    fn target_profile(&self) -> &EffectiveTargetProfile;
    /// Construct the complete plan family for one logical program: every
    /// applicable portable alternative must receive a universal physical
    /// alternative.
    fn elaborate(&self, logical: &LogicalProgram) -> Result<PlanFamily<Self::Dialect>, String>;
    fn encode_launch(
        &self,
        launch: &ResolvedLaunch<Self::Dialect>,
    ) -> Result<Self::EncodedLaunch, String>;
    fn assemble(
        &self,
        encoded: EncodedPlan<Self::Dialect, Self::EncodedLaunch>,
    ) -> Result<Self::NativeArtifact, String>;
}

/// Encoded mirror of the resolved structured schedule.
pub struct EncodedPlan<D: ExecutableDialect, L> {
    pub resolved: Arc<ResolvedPlan<D>>,
    pub steps: Vec<EncodedStep<D, L>>,
}

pub enum EncodedStep<D: ExecutableDialect, L> {
    Launch {
        resolved: ResolvedLaunch<D>,
        encoded: L,
    },
    Call {
        call: seismic_realization::executable::ResolvedCall<D>,
        encoded: Box<EncodedPlanBody<D, L>>,
    },
    If {
        resolved: ResolvedScheduleIf<D>,
        then_steps: Vec<EncodedStep<D, L>>,
        else_steps: Vec<EncodedStep<D, L>>,
    },
    Repeat {
        resolved: ResolvedScheduleRepeat<D>,
        body: Vec<EncodedStep<D, L>>,
    },
}

/// One encoded nested body (no second ABI or arena).
pub struct EncodedPlanBody<D: ExecutableDialect, L> {
    pub steps: Vec<EncodedStep<D, L>>,
}

pub struct Compiled<D: ExecutableDialect, A> {
    pub logical: Arc<LogicalProgram>,
    pub physical: Arc<ResolvedPlan<D>>,
    pub native: A,
}

pub fn compile<B: Backend>(
    program: &Program,
    entry: &str,
    workload: &Workload,
    backend: &B,
    numerical_evidence: &[planning::NumericalEvidence],
    budget: Budget,
) -> Result<Compiled<B::Dialect, B::NativeArtifact>, CompileFailure> {
    let bug = |reason: String| CompileFailure::CompilerBug(InvariantReport(reason));
    let target = EffectiveTargetIdentity {
        backend: backend.target().to_string(),
        capability_fingerprint: backend.capability_fingerprint(),
    };
    let supports = |intrinsic: &IntrinsicUse| backend.supports_intrinsic(intrinsic);
    let mut logical = logical::construct(
        program,
        entry,
        &target,
        &supports,
        workload.shapes.clone(),
        workload.elems.clone(),
    )
    .map_err(|error| match error {
        logical::LogicalConstructionError::NoApplicableImplementation(report) => {
            CompileFailure::NoApplicableImplementation(report)
        }
        logical::LogicalConstructionError::InvalidProgram(reason) => {
            CompileFailure::InvalidSemanticProgram(Diagnostics(vec![reason]))
        }
    })?;
    logical
        .verify()
        .map_err(|reasons| bug(reasons.join("; ")))?;
    logical.install_expected_extents(&workload.extents);
    if logical.target != target {
        return Err(bug(
            "the logical target identity changed during specialization".into(),
        ));
    }
    let family = backend.elaborate(&logical).map_err(bug)?;
    let resolved = planning::plan(
        &logical,
        &family,
        &planning::Context {
            target: backend.target_profile(),
            precision: &workload.precision,
            numerical_evidence,
        },
        budget,
    )
    .map_err(|error| match error {
        planning::PlanningFailure::Infeasible => CompileFailure::PlanningInfeasible,
        planning::PlanningFailure::CompilerBug(reason) => {
            CompileFailure::CompilerBug(InvariantReport(reason))
        }
        planning::PlanningFailure::Solver(reason) => {
            CompileFailure::SystemFailure(SystemReport(reason))
        }
    })?;
    let physical = Arc::new(resolved);
    let encoded = encode_plan(physical.clone(), backend).map_err(bug)?;
    let native = backend
        .assemble(encoded)
        .map_err(|reason| CompileFailure::ToolchainFailure(ToolchainReport(reason)))?;
    Ok(Compiled {
        logical: Arc::new(logical),
        physical,
        native,
    })
}

fn encode_plan<B: Backend>(
    resolved: Arc<ResolvedPlan<B::Dialect>>,
    backend: &B,
) -> Result<EncodedPlan<B::Dialect, B::EncodedLaunch>, String> {
    let steps = encode_steps(&resolved.entry.schedule, backend)?;
    Ok(EncodedPlan { resolved, steps })
}

fn encode_steps<B: Backend>(
    schedule: &seismic_realization::executable::ResolvedSchedule<B::Dialect>,
    backend: &B,
) -> Result<Vec<EncodedStep<B::Dialect, B::EncodedLaunch>>, String> {
    let mut steps = Vec::new();
    for step in schedule.steps.iter() {
        steps.push(match step {
            ResolvedStep::Launch(launch) => EncodedStep::Launch {
                encoded: backend.encode_launch(launch)?,
                resolved: launch.clone(),
            },
            ResolvedStep::Call(call) => {
                let encoded = Box::new(EncodedPlanBody {
                    steps: encode_steps(&call.body.schedule, backend)?,
                });
                EncodedStep::Call {
                    call: call.clone(),
                    encoded,
                }
            }
            ResolvedStep::If(if_step) => EncodedStep::If {
                resolved: if_step.clone(),
                then_steps: encode_steps(&if_step.then_schedule, backend)?,
                else_steps: encode_steps(&if_step.else_schedule, backend)?,
            },
            ResolvedStep::Repeat(repeat) => EncodedStep::Repeat {
                resolved: repeat.clone(),
                body: encode_steps(&repeat.body, backend)?,
            },
        });
    }
    Ok(steps)
}
