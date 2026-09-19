//! Joint selection of authored implementations, contiguous fusion groups and numerical
//! sites (spec sections 9-12). The family comes from `seismic-lang`; the backend adds
//! domains, legality, fusion intervals and local cost factors; `magnitude-solver`
//! searches; the witness is validated, instantiated and handed to the backend to realize.
//!
//! Nothing here or in a backend hook ranks alternatives by profitability outside the
//! solver objective, and nothing after selection changes the witness.
mod analyze;
mod backend;
mod export;
mod greedy;
/// Target-neutral halves of the backend hooks (domains, intervals, limits, factors, seed).
pub mod mapping;
/// Derived quantities over numerical site values.
pub mod quantity;
mod search;
/// The shared structural walk of candidate bodies, parameterized by a backend `Accounting`.
pub mod structure;

pub use analyze::{analyze, analyze_with, SearchAnalysis};
pub use backend::{Backend, Constraint, Factor, Interval, IntervalRef};
pub use search::{replay, select, select_qualified, Budget, Strategy};

use seismic_lang::family::{Family, Witness};
use seismic_lang::precision::NumericalAssessment;
use seismic_lang::types::Elem;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProofStatus {
    /// Complete checked execution; search ended by budget or with open obligations.
    Feasible,
    /// Additionally optimal over the stated family under the stated estimate model.
    ModelOptimal,
}

/// The executable boundary: a checked feasible witness with its realization.
#[derive(Debug)]
pub struct Selected<E> {
    pub execution: E,
    pub family: Arc<Family>,
    pub witness: Witness,
    /// Estimated cost of `witness` in the backend's estimate units.
    pub estimate: u64,
    /// The validated constructive seed and its estimate, retained for comparison.
    pub seed: Witness,
    pub seed_estimate: u64,
    pub status: ProofStatus,
    /// Identity of the estimate model, e.g. `metal-estimate-unqualified-v0`.
    pub estimate_model: String,
    /// Numerical status of the complete selected witness under the requested precision policy.
    pub numerical_assessment: NumericalAssessment,
    /// Identity of the whole-witness qualification selected for execution, when any.
    pub qualification: Option<QualificationIdentity>,
    /// Lower bound proved by the solver over the exported family, in the same units.
    pub lower_bound: u64,
    pub unresolved: Vec<String>,
    /// Wall time of every selection phase of this entry.
    pub timings: Timings,
    pub search: SearchStats,
}

/// Numerical evidence for one complete witness. Bounds belong to the composition as a whole;
/// they are never copied to another witness or specialization.
#[derive(Clone, Debug)]
pub struct Qualification {
    pub program: [u8; 32],
    pub target: String,
    pub numerical_environment: String,
    pub estimate_model: String,
    pub entry: String,
    pub shapes: BTreeMap<String, i64>,
    pub elems: BTreeMap<String, Elem>,
    pub witness: Witness,
    pub assessment: NumericalAssessment,
    pub corpus: String,
    pub method: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QualificationIdentity {
    pub numerical_environment: String,
    pub corpus: String,
    pub method: String,
}

impl Qualification {
    /// Create a policy-bound record for one complete witness. Replay still validates the
    /// witness structurally when the record is consumed.
    pub fn new<B: Backend>(
        program: &seismic_lang::sir::Program,
        backend: &B,
        entry: impl Into<String>,
        workload: &seismic_lang::family::Workload,
        witness: Witness,
        assessment: NumericalAssessment,
        corpus: impl Into<String>,
        method: impl Into<String>,
    ) -> Result<Self, String> {
        if assessment.evidence != seismic_lang::precision::EvidenceClass::Qualified {
            return Err("qualification records require qualified numerical evidence".into());
        }
        if !assessment.satisfies(&workload.precision) {
            return Err("qualified outputs do not satisfy the workload precision policy".into());
        }
        let entry = entry.into();
        let family = seismic_lang::family::construct(program, &entry, backend.target(), workload)?;
        family.validate(&witness)?;
        let root = family.occurrences.first().ok_or("qualification entry has no root occurrence")?;
        let reference = root.candidates.iter().find(|candidate| candidate.reference)
            .ok_or("qualification entry has no portable reference body")?;
        let definition = program.definition(reference.via);
        let mut expected: std::collections::BTreeSet<String> = definition.params.iter()
            .filter(|parameter| !matches!(parameter.mode, seismic_lang::syntax::ast::Mode::In))
            .map(|parameter| parameter.name.clone())
            .collect();
        if !matches!(definition.result, seismic_lang::types::Ty::Void) {
            expected.insert("$return".into());
        }
        let observed: std::collections::BTreeSet<String> = assessment.outputs.iter()
            .map(|output| output.output.clone())
            .collect();
        if observed != expected || assessment.outputs.len() != expected.len() {
            return Err(format!(
                "qualification outputs differ from entry outputs: expected {:?}, observed {:?}",
                expected, observed
            ));
        }
        Ok(Self {
            program: program.identity(),
            target: backend.target().into(),
            numerical_environment: backend.numerical_environment(),
            estimate_model: backend.estimate_model(),
            entry,
            shapes: workload.shapes.clone(),
            elems: workload.elems.clone(),
            witness,
            assessment,
            corpus: corpus.into(),
            method: method.into(),
        })
    }

    pub fn identity(&self) -> QualificationIdentity {
        QualificationIdentity {
            numerical_environment: self.numerical_environment.clone(),
            corpus: self.corpus.clone(),
            method: self.method.clone(),
        }
    }
}

/// Wall time per selection phase. `search` is the solver (or the greedy sweeps); it
/// excludes `seed`, whose validation is charged to `seed`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timings {
    /// `family::construct`.
    pub family: Duration,
    /// `bind_structure` + `constraints` + `intervals` + `factors`.
    pub backend_hooks: Duration,
    /// Tabulation of constraints and factors into the solver model.
    pub export: Duration,
    /// `Backend::seed` and its validation against the exported model.
    pub seed: Duration,
    pub search: Duration,
    pub instantiate: Duration,
    pub realize: Duration,
}

impl Timings {
    /// Everything before instantiation: what choosing the witness cost.
    pub fn solve(&self) -> Duration {
        self.family + self.backend_hooks + self.export + self.seed + self.search
    }
}

/// One solver phase as `magnitude-solver` reports it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Phase {
    pub time: Duration,
    pub work: u64,
    pub nodes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    pub strategy: Strategy,
    /// Variables and factors of the exported model.
    pub variables: usize,
    pub factors: usize,
    /// `Strategy::Exact`: the exact slice, and the neighborhood slice when it ran.
    pub exact: Phase,
    pub neighborhood: Option<Phase>,
    /// `Strategy::Greedy`: completed sweeps and validated trial assignments.
    pub greedy_sweeps: u32,
    pub greedy_trials: u64,
}

/// Distinct outcomes (spec section 13.3). None of these recommends or performs a fallback.
#[derive(Debug)]
pub enum SelectionError {
    /// Type, effect, ownership or declaration error in the linked program.
    InvalidSource(String),
    /// No applicable implementation covers a reached call on this target.
    MissingCoverage(String),
    /// The backend has no mapping for an otherwise meaningful structure.
    UnsupportedMapping(String),
    /// Required representation/participation/lifetime interfaces disagree.
    IncompatibleComposition(String),
    /// The exported joint family is proved to have no solution.
    Infeasible,
    /// Family/interface construction stopped with unexplored obligations.
    ConstructionIncomplete(Vec<String>),
    /// Budget ended without any checked usable configuration.
    SelectionIncomplete(String),
    /// A required quantity or estimate has no supported derivation.
    AnalysisUnavailable(String),
    /// A bounded request permits qualification, but no matching accepted witness exists.
    MissingQualification(String),
    /// Reconstruction disagreed with the witness: a compiler defect.
    Reconstruction(String),
}

impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionError::InvalidSource(m) => write!(f, "invalid source: {m}"),
            SelectionError::MissingCoverage(m) => write!(f, "missing target coverage: {m}"),
            SelectionError::UnsupportedMapping(m) => {
                write!(f, "unsupported structural mapping: {m}")
            }
            SelectionError::IncompatibleComposition(m) => {
                write!(f, "incompatible composition: {m}")
            }
            SelectionError::Infeasible => {
                write!(f, "no feasible configuration in the stated family")
            }
            SelectionError::ConstructionIncomplete(o) => {
                write!(f, "construction incomplete: {}", o.join("; "))
            }
            SelectionError::SelectionIncomplete(m) => write!(f, "selection incomplete: {m}"),
            SelectionError::AnalysisUnavailable(m) => write!(f, "analysis unavailable: {m}"),
            SelectionError::MissingQualification(m) => write!(f, "missing numerical qualification: {m}"),
            SelectionError::Reconstruction(m) => write!(f, "reconstruction defect: {m}"),
        }
    }
}

impl std::error::Error for SelectionError {}
