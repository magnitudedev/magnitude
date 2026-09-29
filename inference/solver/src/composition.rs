//! Exact composition of independent models with additive or maximum objectives.
//!
//! Each component is a distinct occurrence with private variables. Any shared
//! resource, state or boundary decision must instead remain in one joint model.
//! For this product of feasible sets, min(sum costs) = sum(min costs) and
//! min(max costs) = max(min costs). The latter can close before every component
//! is solved, provided every component has a validated witness below that bound.
use crate::{
    result::StopReason, Error, FeasibleSolution, Limits, Model, Options, Outcome, Result, Search,
    Stats,
};
use std::{sync::Arc, time::Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Aggregation {
    Sum,
    Maximum,
}
impl Aggregation {
    fn collect(self, costs: impl IntoIterator<Item = u64>) -> Result<u64> {
        costs.into_iter().try_fold(0u64, |total, cost| match self {
            Self::Sum => total
                .checked_add(cost)
                .ok_or_else(|| Error::Overflow("composed objective".into())),
            Self::Maximum => Ok(total.max(cost)),
        })
    }
}

/// Independence is structural: models have disjoint variable namespaces and
/// no factor is permitted to reach into another component.
#[derive(Clone, Debug)]
pub struct Composition {
    aggregation: Aggregation,
    components: Vec<Arc<Model>>,
}
impl Composition {
    pub fn new(aggregation: Aggregation, components: Vec<Arc<Model>>) -> Result<Self> {
        for component in &components {
            component.validate()?;
            if components
                .first()
                .is_some_and(|first| first.units() != component.units())
            {
                return Err(Error::InvalidModel(
                    "component objective units differ".into(),
                ));
            }
        }
        Ok(Self {
            aggregation,
            components,
        })
    }
    pub fn aggregation(&self) -> Aggregation {
        self.aggregation
    }
    pub fn components(&self) -> &[Arc<Model>] {
        &self.components
    }
}

#[derive(Clone, Debug)]
pub struct CompositeWitness {
    model: Arc<Composition>,
    components: Vec<FeasibleSolution>,
    cost: u64,
}
impl CompositeWitness {
    pub fn model(&self) -> &Composition {
        &self.model
    }
    pub fn components(&self) -> &[FeasibleSolution] {
        &self.components
    }
    pub fn cost(&self) -> u64 {
        self.cost
    }
}

/// The aggregate optimum does not assert optimality of each component witness.
#[derive(Clone, Debug)]
pub struct CompositeSolution {
    feasible: CompositeWitness,
}
impl CompositeSolution {
    pub fn feasible(&self) -> &CompositeWitness {
        &self.feasible
    }
    pub fn cost(&self) -> u64 {
        self.feasible.cost()
    }
}

#[derive(Clone, Debug)]
pub struct CompositeProgress {
    pub lower_bound: u64,
    pub incumbent: Option<CompositeWitness>,
    pub reason: StopReason,
    pub work: u64,
    pub retained_bytes: usize,
}
#[derive(Clone, Debug)]
pub enum CompositeOutcome {
    Optimal(CompositeSolution),
    Infeasible,
    Incomplete(CompositeProgress),
}

struct Component {
    search: Search,
    lower: u64,
    witness: Option<FeasibleSolution>,
    solved: bool,
}
pub struct CompositeSearch {
    model: Arc<Composition>,
    components: Vec<Component>,
    cursor: usize,
    infeasible: bool,
    failure: Option<Error>,
}
impl CompositeSearch {
    pub fn new(model: Arc<Composition>, options: Options) -> Result<Self> {
        let components = model
            .components
            .iter()
            .map(|component| {
                Ok(Component {
                    search: Search::new(component.clone(), options.clone())?,
                    lower: 0,
                    witness: None,
                    solved: false,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            model,
            components,
            cursor: 0,
            infeasible: false,
            failure: None,
        })
    }
    pub fn component_stats(&self) -> impl Iterator<Item = &Stats> {
        self.components.iter().map(|c| c.search.stats())
    }
    pub fn work(&self) -> u64 {
        self.component_stats().map(|s| s.work).sum()
    }
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.components.capacity() * std::mem::size_of::<Component>()
            + self
                .component_stats()
                .map(|s| s.retained_bytes)
                .sum::<usize>()
            + self
                .components
                .iter()
                .filter_map(|c| c.witness.as_ref())
                .map(|w| w.values().len() * std::mem::size_of::<i64>())
                .sum::<usize>()
    }
    fn lower_bound(&self) -> Result<u64> {
        self.model
            .aggregation
            .collect(self.components.iter().map(|c| c.lower))
    }
    fn witness(&self) -> Result<Option<CompositeWitness>> {
        let Some(components) = self
            .components
            .iter()
            .map(|c| c.witness.clone())
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let cost = self
            .model
            .aggregation
            .collect(components.iter().map(FeasibleSolution::cost))?;
        Ok(Some(CompositeWitness {
            model: self.model.clone(),
            components,
            cost,
        }))
    }
    fn incomplete(&self, reason: StopReason) -> Result<CompositeOutcome> {
        Ok(CompositeOutcome::Incomplete(CompositeProgress {
            lower_bound: self.lower_bound()?,
            incumbent: self.witness()?,
            reason,
            work: self.work(),
            retained_bytes: self.retained_bytes(),
        }))
    }
    pub fn advance(&mut self, limits: Limits) -> Result<CompositeOutcome> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        let result = self.advance_inner(limits);
        if let Err(error) = &result {
            self.failure = Some(error.clone());
        }
        result
    }
    fn advance_inner(&mut self, limits: Limits) -> Result<CompositeOutcome> {
        let begin = Instant::now();
        let initial_work = self.work();
        let mut blocked = vec![None; self.components.len()];
        loop {
            if self.infeasible {
                return Ok(CompositeOutcome::Infeasible);
            }
            let lower = self.lower_bound()?;
            if let Some(feasible) = self.witness()? {
                if lower > feasible.cost() {
                    return Err(Error::Invariant("composition bound exceeds witness".into()));
                }
                if lower == feasible.cost() {
                    // Each original model, including every conditional factor,
                    // remains the authority for the corresponding occurrence.
                    for (model, witness) in self.model.components.iter().zip(&feasible.components) {
                        let checked = model.validate_assignment(witness.values())?;
                        if checked.infeasible
                            || checked.unresolved.is_some()
                            || checked.exact_cost != Some(witness.cost())
                        {
                            return Err(Error::Invariant("invalid component witness".into()));
                        }
                    }
                    return Ok(CompositeOutcome::Optimal(CompositeSolution { feasible }));
                }
            }
            if self.work() - initial_work >= limits.work {
                return self.incomplete(StopReason::Work);
            }
            if limits.time.is_some_and(|t| begin.elapsed() >= t) {
                return self.incomplete(StopReason::Time);
            }
            let next = (0..self.components.len())
                .map(|offset| (self.cursor + offset) % self.components.len())
                .find(|&i| {
                    let c = &self.components[i];
                    !c.solved
                        && blocked[i].is_none()
                        && (self.model.aggregation == Aggregation::Sum
                            || c.witness.as_ref().is_none_or(|w| w.cost() > lower))
                });
            let Some(next) = next else {
                let mut obligations = Vec::new();
                for reason in blocked.into_iter().flatten() {
                    match reason {
                        StopReason::Coverage(reasons) => obligations.extend(reasons),
                        other => return self.incomplete(other),
                    }
                }
                if obligations.is_empty() {
                    return Err(Error::Invariant(
                        "unfinished composition has no pending component".into(),
                    ));
                }
                obligations.sort();
                obligations.dedup();
                return self.incomplete(StopReason::Coverage(obligations));
            };
            self.cursor = (next + 1) % self.components.len();
            let other_bytes = self
                .retained_bytes()
                .saturating_sub(self.components[next].search.stats().retained_bytes);
            let child_limit = limits
                .memory_bytes
                .map(|cap| cap.saturating_sub(other_bytes));
            let remaining_work = limits.work - (self.work() - initial_work);
            let component = &mut self.components[next];
            let outcome = component.search.advance(Limits {
                work: remaining_work.min(64),
                time: limits.time.map(|t| t.saturating_sub(begin.elapsed())),
                memory_bytes: child_limit,
            })?;
            match outcome {
                Outcome::Infeasible => self.infeasible = true,
                Outcome::Optimal(solution) => {
                    component.lower = solution.cost();
                    component.witness = Some(solution.feasible().clone());
                    component.solved = true;
                }
                Outcome::Incomplete(progress) => {
                    component.lower = progress.lower_bound;
                    component.witness = progress.incumbent;
                    if progress.reason != StopReason::Work {
                        blocked[next] = Some(progress.reason);
                    }
                }
            }
        }
    }
}
