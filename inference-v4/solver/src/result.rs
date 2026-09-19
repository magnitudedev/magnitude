use crate::model::Model;
use std::sync::Arc;

/// A witness evaluated against the original immutable model. Construction is private.
#[derive(Clone, Debug)]
pub struct FeasibleSolution {
    pub(crate) model: Arc<Model>,
    pub(crate) values: Vec<i64>,
    pub(crate) cost: u64,
}
impl FeasibleSolution {
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
