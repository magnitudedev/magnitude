//! Compact repetition of an independent additive body.
//!
//! Each occurrence is a distinct copy of the complete body model and has no
//! shared variables, resources, retained state or cross-occurrence constraints.
//! The repeated objective is the sum of occurrence objectives. This makes solving
//! one body and multiplying its optimum exact, rather than a wave-time estimate.
//! Serial executions with reset state fit this model. [`CoupledRepetition`]
//! represents repeated finite activity templates with shared capacities and
//! cross-occurrence precedence. It uses proved compact reductions where they
//! apply and a resumable, complete joint finite model everywhere else.

mod coupled;
pub use coupled::{
    CoupledActivity, CoupledIncumbent, CoupledOutcome, CoupledProgress, CoupledRepetition,
    CoupledSolution, CoupledStats, CoupledTemplate, RepeatedPrecedence, RepetitionMethod,
    RepetitionStrategy,
};

use crate::result::FeasibleSolution;
use crate::result::StopReason;
use crate::{Error, Limits, Model, Options, Outcome, Result, Search, Solution, Stats};
use std::sync::Arc;

/// A repeated witness; one assignment is reused for independent occurrences.
/// The occurrences still each contribute their cost: memo reuse is not sharing.
#[derive(Clone, Debug)]
pub struct RepeatedSolution {
    count: u64,
    body: Option<Solution>,
    cost: u64,
}

impl RepeatedSolution {
    pub fn count(&self) -> u64 {
        self.count
    }
    pub fn body(&self) -> Option<&Solution> {
        self.body.as_ref()
    }
    pub fn cost(&self) -> u64 {
        self.cost
    }
}

#[derive(Clone, Debug)]
pub struct RepetitionProgress {
    pub lower_bound: u64,
    pub incumbent: Option<RepeatedIncumbent>,
    pub reason: StopReason,
    pub stats: Stats,
}

/// A feasible repeated assignment without completed exclusion of improvements.
/// This type cannot be used to construct an optimal repetition result.
#[derive(Clone, Debug)]
pub struct RepeatedIncumbent {
    count: u64,
    body: FeasibleSolution,
    cost: u64,
}

impl RepeatedIncumbent {
    pub fn count(&self) -> u64 {
        self.count
    }
    pub fn body(&self) -> &FeasibleSolution {
        &self.body
    }
    pub fn cost(&self) -> u64 {
        self.cost
    }
}

#[derive(Clone, Debug)]
pub enum RepetitionOutcome {
    Optimal(RepeatedSolution),
    Infeasible,
    Incomplete(RepetitionProgress),
}

pub struct IndependentRepetition {
    count: u64,
    search: Search,
}

impl IndependentRepetition {
    pub fn new(body: Arc<Model>, count: u64, options: Options) -> Result<Self> {
        Ok(Self {
            count,
            search: Search::new(body, options)?,
        })
    }

    pub fn count(&self) -> u64 {
        self.count
    }
    pub fn body_search(&self) -> &Search {
        &self.search
    }

    pub fn advance(&mut self, limits: Limits) -> Result<RepetitionOutcome> {
        if self.count == 0 {
            return Ok(RepetitionOutcome::Optimal(RepeatedSolution {
                count: 0,
                body: None,
                cost: 0,
            }));
        }
        Ok(match self.search.advance(limits)? {
            Outcome::Optimal(solution) => {
                RepetitionOutcome::Optimal(repeat_solution(self.count, solution)?)
            }
            Outcome::Infeasible => RepetitionOutcome::Infeasible,
            Outcome::Incomplete(progress) => RepetitionOutcome::Incomplete(RepetitionProgress {
                lower_bound: multiply(progress.lower_bound, self.count)?,
                incumbent: progress
                    .incumbent
                    .map(|body| {
                        let cost = multiply(body.cost(), self.count)?;
                        Ok::<_, Error>(RepeatedIncumbent {
                            count: self.count,
                            body,
                            cost,
                        })
                    })
                    .transpose()?,
                reason: progress.reason,
                stats: progress.stats,
            }),
        })
    }
}

fn repeat_solution(count: u64, solution: Solution) -> Result<RepeatedSolution> {
    let cost = multiply(solution.cost(), count)?;
    Ok(RepeatedSolution {
        count,
        body: Some(solution),
        cost,
    })
}

fn multiply(cost: u64, count: u64) -> Result<u64> {
    cost.checked_mul(count)
        .ok_or_else(|| Error::Overflow("repeated objective".into()))
}
