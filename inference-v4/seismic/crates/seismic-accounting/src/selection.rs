//! Selection over the implementation's own choices and derived constraints.
//!
//! Search state is private. Bounds, coverage and schedule feasibility are checked
//! where they are computed, without a second proof graph or certificate language.
use crate::schedule::{self, Model, SearchOutcome, Solution};
mod domain;
pub use domain::{Choices, Domain, IntegerRange, Region};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Context {
    pub program: String,
    pub workload: String,
    pub target: String,
    pub contracts: String,
    pub execution_form: String,
    pub objective: String,
    pub seconds_numerator: u64,
    pub seconds_denominator: u64,
}
impl Context {
    fn validate(&self) -> Result<(), String> {
        if [
            &self.program,
            &self.workload,
            &self.target,
            &self.contracts,
            &self.execution_form,
            &self.objective,
        ]
        .iter()
        .any(|s| s.is_empty())
            || self.seconds_numerator == 0
            || self.seconds_denominator == 0
        {
            return Err("selection requires identified inputs and a positive time unit".into());
        }
        Ok(())
    }
    fn check_model(&self, model: &Model) -> Result<(), String> {
        if u128::from(model.timebase.seconds_numerator) * u128::from(self.seconds_denominator)
            != u128::from(self.seconds_numerator) * u128::from(model.timebase.seconds_denominator)
        {
            return Err("execution analysis has a different objective time unit".into());
        }
        model.relationship.require_feasible_upper()
    }
}

/// An interval for this analysis's optimum, never an independently supplied cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    lower: u64,
    upper: u64,
}
impl Cost {
    pub fn lower(self) -> u64 {
        self.lower
    }
    pub fn upper(self) -> u64 {
        self.upper
    }
    pub fn is_exact(self) -> bool {
        self.lower == self.upper
    }
}

/// A violated capacity derived by the implementation's own construction rules.
/// This is not a hardware fact or a separately supplied proof of infeasibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CapacityViolation {
    pub resource: String,
    pub required: u64,
    pub available: u64,
}
impl CapacityViolation {
    fn validate(&self) -> Result<(), String> {
        if self.resource.is_empty() || self.required <= self.available {
            return Err("infeasible branch has no violated implementation capacity".into());
        }
        Ok(())
    }
}
pub enum Node<E> {
    Choice { name: String, alternatives: Domain },
    Realization(E),
    Infeasible(CapacityViolation),
}
pub trait Space {
    type Execution;
    fn context(&self) -> &Context;
    fn expand(&self, prefix: &[usize]) -> Result<Node<Self::Execution>, String>;
    /// Derive constraints from this exact execution and bound hardware facts.
    fn analyze(&self, execution: &Self::Execution) -> Result<Model, String>;
    /// Resolve the selected schedule in the emitted execution and validate that
    /// transformation against the source before returning it.
    fn materialize(
        &self,
        execution: &Self::Execution,
        objective: &Objective,
    ) -> Result<Self::Execution, String>;
}

/// Read-only projection of a scheduling analysis tied to its exact constraints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Objective {
    solution: Solution,
}
impl Objective {
    pub fn model(&self) -> &Model {
        self.solution.model()
    }
    pub fn schedule(&self) -> &schedule::Schedule {
        self.solution.schedule()
    }
    pub fn cost(&self) -> Cost {
        Cost {
            lower: self.solution.lower_bound(),
            upper: self.schedule().completion,
        }
    }
}
pub struct Selected<E> {
    execution: E,
    path: Vec<usize>,
    objective: Objective,
}
impl<E> Selected<E> {
    pub fn execution(&self) -> &E {
        &self.execution
    }
    pub fn selected_path(&self) -> &[usize] {
        &self.path
    }
    pub fn cost(&self) -> Cost {
        let upper = self.objective.cost().upper();
        Cost {
            lower: upper,
            upper,
        }
    }
    pub fn objective(&self) -> &Objective {
        &self.objective
    }
    pub fn into_execution(self) -> E {
        self.execution
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// New choice/execution nodes per call; this never truncates a legal domain.
    pub nodes: usize,
    /// Proposed start assignments per leaf; resumption increases this allowance.
    pub schedule_assignments: u64,
}
#[derive(Clone, Debug, PartialEq)]
enum Analysis {
    Choice(Domain),
    Execution {
        model: Model,
        outcome: SearchOutcome,
        allowance: u64,
    },
    Infeasible(CapacityViolation),
}
struct Record {
    path: Vec<usize>,
    analysis: Analysis,
}
struct Incumbent<E> {
    record: usize,
    execution: E,
    upper: u64,
}
/// Coverage is maintained by the search itself. Cached analyses are rederived
/// from their exact executions on resumption before they can justify exclusions.
pub struct Progress<E> {
    context: Context,
    pending: Vec<Region>,
    records: Vec<Record>,
    incumbent: Option<Incumbent<E>>,
}
impl<E> Progress<E> {
    pub fn nodes_visited(&self) -> usize {
        self.records.len()
    }
    pub fn frontier(&self) -> &[Region] {
        &self.pending
    }
    pub fn incumbent(&self) -> Option<&E> {
        self.incumbent.as_ref().map(|i| &i.execution)
    }
    pub fn feasible_upper(&self) -> Option<u64> {
        self.incumbent.as_ref().map(|i| i.upper)
    }
    pub fn lower_bound(&self) -> Result<u64, String> {
        // One selected implementation cannot establish a region-wide bound.
        if !self.pending.is_empty() {
            return Ok(0);
        }
        self.records
            .iter()
            .filter_map(|record| match &record.analysis {
                Analysis::Execution { outcome, .. } => schedule_lower(outcome),
                _ => None,
            })
            .min()
            .ok_or_else(|| "execution domain has no feasible member".into())
    }
}
pub enum Outcome<E> {
    Optimal(Selected<E>),
    Incomplete(Progress<E>),
    Infeasible,
}

pub fn select<S: Space>(space: &S, budget: Budget) -> Result<Outcome<S::Execution>, String> {
    space.context().validate()?;
    resume(
        space,
        Progress {
            context: space.context().clone(),
            pending: vec![Region::Branch { path: Vec::new() }],
            records: Vec::new(),
            incumbent: None,
        },
        budget,
    )
}

pub fn resume<S: Space>(
    space: &S,
    mut progress: Progress<S::Execution>,
    budget: Budget,
) -> Result<Outcome<S::Execution>, String> {
    space.context().validate()?;
    if space.context() != &progress.context {
        return Err("selection inputs changed".into());
    }
    // These are cache consistency checks in the existing analysis path, not a
    // second interpretation of the computation or independent proof replay.
    for index in 0..progress.records.len() {
        let record = &mut progress.records[index];
        match (&mut record.analysis, space.expand(&record.path)?) {
            (
                Analysis::Choice(alternatives),
                Node::Choice {
                    alternatives: domain,
                    ..
                },
            ) if *alternatives == domain => {}
            (Analysis::Infeasible(saved), Node::Infeasible(actual)) if *saved == actual => {}
            (
                Analysis::Execution {
                    model: saved,
                    outcome,
                    allowance,
                },
                Node::Realization(execution),
            ) => {
                let model = space.analyze(&execution)?;
                progress.context.check_model(&model)?;
                if model != *saved {
                    return Err("execution analysis changed during resumption".into());
                }
                let excluded = matches!(outcome, SearchOutcome::Infeasible)
                    || progress.incumbent.as_ref().is_some_and(|i| {
                        schedule_lower(outcome).is_some_and(|lower| lower >= i.upper)
                    });
                if !excluded {
                    *allowance = allowance
                        .checked_add(budget.schedule_assignments)
                        .ok_or("schedule budget overflow")?;
                    *outcome = model.search(*allowance)?;
                }
                if let SearchOutcome::Feasible(solution) = outcome {
                    let upper = solution.schedule().completion;
                    // Reuse the derived analysis, but materialize the owner's
                    // current execution even when its resource costs are equal.
                    if progress
                        .incumbent
                        .as_ref()
                        .is_none_or(|i| i.record == index || upper < i.upper)
                    {
                        progress.incumbent = Some(Incumbent {
                            record: index,
                            execution,
                            upper,
                        });
                    }
                }
            }
            _ => return Err("implementation choices changed during resumption".into()),
        }
    }
    for _ in 0..budget.nodes {
        let Some(path) = domain::pop(&mut progress.pending) else {
            break;
        };
        let analysis = match space.expand(&path)? {
            Node::Choice { alternatives, .. } => {
                alternatives.validate()?;
                progress.pending.push(Region::Children {
                    parent: path.clone(),
                    indices: 0..alternatives.len(),
                });
                Analysis::Choice(alternatives)
            }
            Node::Infeasible(violation) => {
                violation.validate()?;
                Analysis::Infeasible(violation)
            }
            Node::Realization(execution) => {
                let model = space.analyze(&execution)?;
                progress.context.check_model(&model)?;
                let outcome = model.search(budget.schedule_assignments)?;
                if let SearchOutcome::Feasible(solution) = &outcome {
                    let upper = solution.schedule().completion;
                    if progress.incumbent.as_ref().is_none_or(|i| upper < i.upper) {
                        progress.incumbent = Some(Incumbent {
                            record: progress.records.len(),
                            execution,
                            upper,
                        });
                    }
                }
                Analysis::Execution {
                    model,
                    outcome,
                    allowance: budget.schedule_assignments,
                }
            }
        };
        progress.records.push(Record { path, analysis });
    }
    if !progress.pending.is_empty() {
        return Ok(Outcome::Incomplete(progress));
    }
    let Some(best) = &progress.incumbent else {
        let unresolved = progress.records.iter().any(|record| {
            matches!(
                record.analysis,
                Analysis::Execution {
                    outcome: SearchOutcome::Incomplete { .. },
                    ..
                }
            )
        });
        return Ok(if unresolved {
            Outcome::Incomplete(progress)
        } else {
            Outcome::Infeasible
        });
    };
    if progress.lower_bound()? < best.upper {
        return Ok(Outcome::Incomplete(progress));
    }
    let best = progress.incumbent.take().unwrap();
    let record = progress.records.swap_remove(best.record);
    let Analysis::Execution {
        outcome: SearchOutcome::Feasible(solution),
        ..
    } = record.analysis
    else {
        unreachable!("incumbent owns a feasible scheduling result")
    };
    let objective = Objective { solution };
    objective
        .model()
        .check_execution_upper(objective.schedule())?;
    let execution = space.materialize(&best.execution, &objective)?;
    Ok(Outcome::Optimal(Selected {
        execution,
        path: record.path,
        objective,
    }))
}

fn schedule_lower(outcome: &SearchOutcome) -> Option<u64> {
    match outcome {
        SearchOutcome::Feasible(solution) => Some(solution.lower_bound()),
        SearchOutcome::Incomplete { lower_bound } => Some(*lower_bound),
        SearchOutcome::Infeasible => None,
    }
}
