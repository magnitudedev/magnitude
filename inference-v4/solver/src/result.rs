use crate::model::{Error, Model, Result};
use std::sync::Arc;

/// A witness evaluated against the original immutable model. Construction is private.
#[derive(Clone, Debug)]
pub struct FeasibleSolution {
    pub(crate) model: Arc<Model>,
    pub(crate) values: Vec<i64>,
    pub(crate) cost: u64,
}
impl FeasibleSolution {
    /// Validates a complete assignment against the exact model and constructs
    /// the feasible witness. Domain, constraint and coverage violations are
    /// rejected.
    pub fn from_assignment(model: Arc<Model>, values: &[i64]) -> Result<Self> {
        let assessment = model.validate_assignment(values)?;
        if assessment.infeasible {
            return Err(Error::InvalidAssignment(
                "assignment violates the model".into(),
            ));
        }
        let cost = assessment
            .exact_cost
            .ok_or_else(|| Error::InvalidAssignment("assignment has unresolved coverage".into()))?;
        Ok(Self {
            model,
            values: values.to_vec(),
            cost,
        })
    }
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn values(&self) -> &[i64] {
        &self.values
    }
    pub fn cost(&self) -> u64 {
        self.cost
    }
}
/// A validated witness whose optimality has been established over the whole model.
/// Diagnostic incumbents cannot be converted to this type by library consumers.
#[derive(Clone, Debug)]
pub struct Solution {
    pub(crate) feasible: FeasibleSolution,
}
impl Solution {
    pub fn model(&self) -> &Model {
        self.feasible.model()
    }
    pub fn values(&self) -> &[i64] {
        self.feasible.values()
    }
    pub fn cost(&self) -> u64 {
        self.feasible.cost()
    }
    pub fn feasible(&self) -> &FeasibleSolution {
        &self.feasible
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StopReason {
    Work,
    Time,
    Memory,
    Coverage(Vec<crate::model::Obligation>),
}
#[derive(Clone, Debug)]
pub struct Progress {
    pub lower_bound: u64,
    pub incumbent: Option<FeasibleSolution>,
    pub reason: StopReason,
    pub stats: crate::search::Stats,
}
#[derive(Clone, Debug)]
pub enum Outcome {
    Optimal(Solution),
    Infeasible,
    Incomplete(Progress),
}
/// Result of one budgeted solve. Budget limits optimization only: feasibility
/// is decided before any optimization budget is spent, so `Suspended` appears
/// only when retained memory or a coverage obligation blocked the search,
/// never because work or time ran out before feasibility was decided.
#[derive(Clone, Debug)]
pub enum Budgeted {
    /// Optimality proved over the whole model.
    Optimal(Solution),
    /// Every complete assignment violates the model.
    Infeasible,
    /// Best feasible incumbent; optimality was not established within budget.
    Incumbent {
        solution: FeasibleSolution,
        lower_bound: u64,
    },
    /// Feasibility undecided by memory pressure or a coverage obligation.
    Suspended(Progress),
}
/// Exported feasible assignment for the exact model. Constructible only from
/// a solver-validated `FeasibleSolution`; consumers cannot forge one.
#[derive(Clone, Debug)]
pub struct FeasibleAssignment {
    solution: FeasibleSolution,
}
impl FeasibleAssignment {
    pub fn new(solution: FeasibleSolution) -> Self {
        Self { solution }
    }
    pub fn solution(&self) -> &FeasibleSolution {
        &self.solution
    }
    pub fn values(&self) -> &[i64] {
        self.solution.values()
    }
    pub fn cost(&self) -> u64 {
        self.solution.cost()
    }
    pub fn model(&self) -> &Model {
        self.solution.model()
    }
}
impl From<FeasibleSolution> for FeasibleAssignment {
    fn from(solution: FeasibleSolution) -> Self {
        Self { solution }
    }
}
