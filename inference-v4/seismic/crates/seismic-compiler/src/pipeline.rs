//! Sole production compilation seam.
//!
//! Backends construct complete schedules, encode one resolved launch at a
//! time, and assemble the hierarchy already encoded by the compiler core.

use crate::planning::{self, Budget};
use seismic_lang::{
    family::{TargetEnvironment, Workload},
    logical::{self, LogicalProgram},
    sir::{IntrinsicUse, Program},
};
use seismic_realization::executable::{
    ExecutableDialect, ExecutableTargetProfile, PlanFamily, ResolvedLaunch, ResolvedPhase,
    ResolvedPlan, ResolvedScheduleItem, ResolvedSubplan,
};
use std::sync::Arc;

pub trait Backend {
    type Dialect: ExecutableDialect;
    type EncodedLaunch;
    type NativeArtifact;

    fn target(&self) -> &'static str;
    fn capability_fingerprint(&self) -> String;
    fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String>;
    fn target_profile(
        &self,
    ) -> &ExecutableTargetProfile<<Self::Dialect as ExecutableDialect>::Capability>;
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

pub struct EncodedPlan<D: ExecutableDialect, L> {
    pub resolved: Arc<ResolvedPlan<D>>,
    pub items: Vec<EncodedScheduleItem<D, L>>,
}

pub enum EncodedScheduleItem<D: ExecutableDialect, L> {
    Phase(EncodedPhase<D, L>),
    Subplan(EncodedSubplan<D, L>),
}

pub struct EncodedPhase<D: ExecutableDialect, L> {
    pub resolved: ResolvedPhase<D>,
    pub launches: Vec<L>,
}

pub struct EncodedSubplan<D: ExecutableDialect, L> {
    pub resolved: ResolvedSubplan<D>,
    pub encoded: Box<EncodedPlan<D, L>>,
}

pub struct Compiled<D: ExecutableDialect, A> {
    pub logical: Arc<LogicalProgram>,
    pub physical: Arc<ResolvedPlan<D>>,
    pub native: A,
}

#[derive(Debug)]
pub enum CompileError {
    Specialization(String),
    Elaboration(String),
    Planning(planning::PlanningError),
    Emission(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Specialization(message) => write!(f, "logical specialization failed: {message}"),
            Self::Elaboration(message) => write!(f, "physical elaboration failed: {message}"),
            Self::Planning(error) => error.fmt(f),
            Self::Emission(message) => write!(f, "native emission failed: {message}"),
        }
    }
}

impl std::error::Error for CompileError {}

pub fn compile<B: Backend>(
    program: &Program,
    entry: &str,
    workload: &Workload,
    backend: &B,
    numerical_evidence: &[planning::NumericalEvidence],
    budget: Budget,
) -> Result<Compiled<B::Dialect, B::NativeArtifact>, CompileError> {
    let capability_fingerprint = backend.capability_fingerprint();
    let supports = |intrinsic: &IntrinsicUse| backend.supports_intrinsic(intrinsic);
    let environment = TargetEnvironment {
        target: backend.target(),
        capability_fingerprint: &capability_fingerprint,
        supports_intrinsic: &supports,
    };
    let logical = logical::specialize_entry_contract(program, entry, &environment, workload)
        .map_err(CompileError::Specialization)?;
    logical.verify().map_err(CompileError::Specialization)?;
    if logical.target != backend.target()
        || logical.capability_fingerprint != capability_fingerprint
    {
        return Err(CompileError::Specialization(
            "logical target identity changed during specialization".into(),
        ));
    }
    let family = backend
        .elaborate(&logical)
        .map_err(CompileError::Elaboration)?;
    let resolved = planning::plan(
        &logical,
        family,
        planning::Context {
            target: backend.target_profile(),
            precision: &workload.precision,
            numerical_evidence,
        },
        budget,
    )
    .map_err(CompileError::Planning)?;
    let physical = Arc::new(resolved);
    let encoded = encode_plan(physical.clone(), backend).map_err(CompileError::Emission)?;
    let native = backend.assemble(encoded).map_err(CompileError::Emission)?;
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
    let mut items = Vec::with_capacity(resolved.items().len());
    for item in resolved.items().iter() {
        match item {
            ResolvedScheduleItem::Phase(phase) => {
                let launches = phase
                    .launches
                    .iter()
                    .map(|launch| backend.encode_launch(launch))
                    .collect::<Result<Vec<_>, _>>()?;
                items.push(EncodedScheduleItem::Phase(EncodedPhase {
                    resolved: phase.clone(),
                    launches,
                }));
            }
            ResolvedScheduleItem::Subplan(subplan) => {
                let child = Arc::new((*subplan.plan).clone());
                let encoded = encode_plan(child, backend)?;
                items.push(EncodedScheduleItem::Subplan(EncodedSubplan {
                    resolved: subplan.clone(),
                    encoded: Box::new(encoded),
                }));
            }
        }
    }
    Ok(EncodedPlan { resolved, items })
}
