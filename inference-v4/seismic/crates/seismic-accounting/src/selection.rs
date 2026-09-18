//! Selection over the implementation's own choices and derived constraints.
//!
//! Search state is private. Bounds, coverage and schedule feasibility are checked
//! where they are computed, without a second proof graph or certificate language.
use crate::schedule::{self, Model, SearchOutcome, Solution};
use crate::workload::{DerivationError, DerivationLimit};
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
        self.check_timebase(&model.timebase)?;
        model.relationship.require_feasible_upper()
    }
    fn check_timebase(&self, timebase: &schedule::Timebase) -> Result<(), String> {
        if u128::from(timebase.seconds_numerator) * u128::from(self.seconds_denominator)
            != u128::from(self.seconds_numerator) * u128::from(timebase.seconds_denominator)
        {
            return Err("execution analysis has a different objective time unit".into());
        }
        Ok(())
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
    /// Exact immutable inputs owned by this implementation space. Display names
    /// are not cache identity. Include source, legal form, workload, and hardware.
    type Identity: Clone + PartialEq;
    fn identity(&self) -> Self::Identity;
    fn context(&self) -> &Context;
    fn expand(&self, prefix: &[usize]) -> Result<Node<Self::Execution>, String>;
    /// Necessary resource demand of the partial implementation in the domain,
    /// relaxed over every indexed alternative. No arbitrary scalar score.
    fn relax(
        &self,
        _alternatives: &Domain,
        _indices: std::ops::Range<usize>,
    ) -> Result<Option<schedule::Demand>, String> {
        Ok(None)
    }
    /// Derive constraints from this exact execution and bound hardware facts.
    fn analyze(&self, execution: &Self::Execution) -> Result<Model, DerivationError>;
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
    /// Indexed domains are bisected without materializing their alternatives.
    pub nodes: usize,
    /// Additional start-time regions per leaf; resumption retains its frontier.
    pub schedule_assignments: u64,
}
enum Analysis {
    Choice(Domain),
    PendingDerivation {
        exhausted: DerivationLimit,
        lower_bound: u64,
    },
    Execution {
        search: schedule::Search,
        outcome: SearchOutcome,
    },
    Infeasible(CapacityViolation),
}
struct Record<E> {
    path: Vec<usize>,
    analysis: Analysis,
    /// Keep the already constructed execution while its model needs a larger
    /// derivation budget. Completed models remain in their scheduling search.
    deferred_execution: Option<E>,
}
struct Incumbent<E> {
    record: usize,
    execution: E,
    upper: u64,
}
/// Coverage is maintained by the search itself. Cached analyses remain bound to
/// the exact source, workload, implementation and hardware identity on resumption.
pub struct Progress<E, I> {
    context: Context,
    identity: I,
    pending: Vec<Region>,
    excluded: Vec<Region>,
    records: Vec<Record<E>>,
    incumbent: Option<Incumbent<E>>,
}
impl<E, I> Progress<E, I> {
    pub fn nodes_visited(&self) -> usize {
        self.records.len()
    }
    pub fn frontier(&self) -> &[Region] {
        &self.pending
    }
    pub fn excluded_regions(&self) -> &[Region] {
        &self.excluded
    }
    pub fn decision(&self, path: &[usize]) -> Option<&Domain> {
        self.records
            .iter()
            .find_map(|record| match &record.analysis {
                Analysis::Choice(domain) if record.path == path => Some(domain),
                _ => None,
            })
    }
    pub fn capacity_violations(&self) -> impl Iterator<Item = (&[usize], &CapacityViolation)> {
        self.records
            .iter()
            .filter_map(|record| match &record.analysis {
                Analysis::Infeasible(violation) => Some((record.path.as_slice(), violation)),
                _ => None,
            })
    }
    pub fn incumbent(&self) -> Option<&E> {
        self.incumbent.as_ref().map(|i| &i.execution)
    }
    pub fn feasible_upper(&self) -> Option<u64> {
        self.incumbent.as_ref().map(|i| i.upper)
    }
    pub fn unresolved(&self) -> Unresolved {
        let mut unresolved = Unresolved {
            choice_regions: self.pending.len(),
            ..Unresolved::default()
        };
        for record in &self.records {
            if analysis_lower(&record.analysis).is_some_and(|lower| {
                self.incumbent
                    .as_ref()
                    .is_some_and(|best| lower >= best.upper)
            }) {
                continue;
            }
            match &record.analysis {
                Analysis::PendingDerivation { .. } => unresolved.derivations += 1,
                Analysis::Execution { search, outcome } => match outcome {
                    SearchOutcome::Infeasible => {}
                    SearchOutcome::Feasible(solution) if solution.is_optimal() => {}
                    _ if !search.model().unmapped.is_empty() => unresolved.unmapped_models += 1,
                    _ => unresolved.schedules += 1,
                },
                _ => {}
            }
        }
        unresolved
    }
    pub fn exhausted_derivations(&self) -> impl Iterator<Item = (&[usize], DerivationLimit)> {
        self.records
            .iter()
            .filter_map(|record| match record.analysis {
                Analysis::PendingDerivation {
                    exhausted,
                    lower_bound,
                } if self
                    .incumbent
                    .as_ref()
                    .is_none_or(|best| lower_bound < best.upper) =>
                {
                    Some((record.path.as_slice(), exhausted))
                }
                _ => None,
            })
    }
    pub fn lower_bound(&self) -> Result<u64, String> {
        self.records
            .iter()
            .filter_map(|record| analysis_lower(&record.analysis))
            .chain(self.pending.iter().map(Region::lower_bound))
            .min()
            .ok_or_else(|| "execution domain has no feasible member".into())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unresolved {
    pub choice_regions: usize,
    pub derivations: usize,
    pub schedules: usize,
    pub unmapped_models: usize,
}
pub enum Outcome<E, I> {
    Optimal(Selected<E>),
    Incomplete(Progress<E, I>),
    Infeasible,
}

pub fn select<S: Space>(
    space: &S,
    budget: Budget,
) -> Result<Outcome<S::Execution, S::Identity>, String> {
    space.context().validate()?;
    resume(
        space,
        Progress {
            context: space.context().clone(),
            identity: space.identity(),
            pending: vec![Region::root()],
            excluded: Vec::new(),
            records: Vec::new(),
            incumbent: None,
        },
        budget,
    )
}

pub fn resume<S: Space>(
    space: &S,
    mut progress: Progress<S::Execution, S::Identity>,
    budget: Budget,
) -> Result<Outcome<S::Execution, S::Identity>, String> {
    space.context().validate()?;
    if space.context() != &progress.context || space.identity() != progress.identity {
        return Err("selection inputs changed".into());
    }
    // Exact source/implementation/workload/hardware identity was checked above.
    // Reuse the derived constraints; resumption must not rebuild every visited
    // execution or replay every old choice before making forward progress.
    for index in 0..progress.records.len() {
        let record = &mut progress.records[index];
        let excluded = analysis_lower(&record.analysis).is_some_and(|lower| {
            progress
                .incumbent
                .as_ref()
                .is_some_and(|best| lower >= best.upper)
        });
        if !excluded {
            match &mut record.analysis {
                Analysis::PendingDerivation { lower_bound, .. } => {
                    record.analysis = analyze(
                        space,
                        record
                            .deferred_execution
                            .as_ref()
                            .expect("deferred model owns its execution"),
                        *lower_bound,
                        budget.schedule_assignments,
                    )?;
                }
                Analysis::Execution { search, outcome } => {
                    let excluded = matches!(outcome, SearchOutcome::Infeasible);
                    if !excluded {
                        *outcome = search.advance(budget.schedule_assignments)?;
                    }
                }
                _ => {}
            }
        }
        if let Analysis::Execution { outcome, .. } = &record.analysis {
            if let SearchOutcome::Feasible(solution) = outcome {
                let upper = solution.schedule().completion;
                if progress.incumbent.as_ref().is_none_or(|i| upper < i.upper) {
                    let execution = match record.deferred_execution.take() {
                        Some(execution) => execution,
                        None => {
                            let Node::Realization(execution) = space.expand(&record.path)? else {
                                return Err(
                                    "bound implementation changed under identical inputs".into()
                                );
                            };
                            execution
                        }
                    };
                    progress.incumbent = Some(Incumbent {
                        record: index,
                        execution,
                        upper,
                    });
                }
            }
            record.deferred_execution = None;
        }
    }
    for _ in 0..budget.nodes {
        let next = loop {
            // Order only by established objective bounds. Equal bounds follow
            // lexicographic choice identity, with no performance preference.
            let Some((index, _)) = progress
                .pending
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.lower_bound()
                        .cmp(&b.lower_bound())
                        .then_with(|| a.first_path().cmp(&b.first_path()))
                })
            else {
                break None;
            };
            let mut region = progress.pending.remove(index);
            if let Some((alternatives, indices)) = region.alternatives() {
                if let Some(demand) = space.relax(alternatives, indices)? {
                    progress.context.check_timebase(demand.timebase())?;
                    region.strengthen(&demand)?;
                }
            }
            if progress
                .incumbent
                .as_ref()
                .is_some_and(|i| region.lower_bound() >= i.upper)
            {
                progress.excluded.push(region);
                continue;
            }
            match region.split() {
                Ok((left, right)) => {
                    // Each child covers its full interval. Endpoint samples do
                    // not establish bounds on a family of implementations.
                    for mut region in [left, right] {
                        if let Some((domain, indices)) = region.alternatives() {
                            if let Some(demand) = space.relax(domain, indices)? {
                                progress.context.check_timebase(demand.timebase())?;
                                region.strengthen(&demand)?;
                            }
                        }
                        progress.pending.push(region);
                    }
                }
                Err(region) => break Some((region.first_path(), region.lower_bound())),
            }
        };
        let Some((path, lower_bound)) = next else {
            break;
        };
        let mut deferred_execution = None;
        let analysis = match space.expand(&path)? {
            Node::Choice { alternatives, .. } => {
                alternatives.validate()?;
                let mut region = Region::children(path.clone(), alternatives.clone(), lower_bound);
                if let Some(demand) = space.relax(&alternatives, 0..alternatives.len())? {
                    progress.context.check_timebase(demand.timebase())?;
                    region.strengthen(&demand)?;
                }
                progress.pending.push(region);
                Analysis::Choice(alternatives)
            }
            Node::Infeasible(violation) => {
                violation.validate()?;
                Analysis::Infeasible(violation)
            }
            Node::Realization(execution) => {
                let analysis =
                    analyze(space, &execution, lower_bound, budget.schedule_assignments)?;
                if matches!(analysis, Analysis::PendingDerivation { .. }) {
                    deferred_execution = Some(execution);
                } else if let Analysis::Execution {
                    outcome: SearchOutcome::Feasible(solution),
                    ..
                } = &analysis
                {
                    let upper = solution.schedule().completion;
                    if progress.incumbent.as_ref().is_none_or(|i| upper < i.upper) {
                        progress.incumbent = Some(Incumbent {
                            record: progress.records.len(),
                            execution,
                            upper,
                        });
                    }
                }
                analysis
            }
        };
        progress.records.push(Record {
            path,
            analysis,
            deferred_execution,
        });
    }
    if let Some(best) = &progress.incumbent {
        let mut retained = Vec::new();
        for region in std::mem::take(&mut progress.pending) {
            if region.lower_bound() >= best.upper {
                progress.excluded.push(region);
            } else {
                retained.push(region);
            }
        }
        progress.pending = retained;
    }
    if !progress.pending.is_empty() {
        return Ok(Outcome::Incomplete(progress));
    }
    let Some(best) = &progress.incumbent else {
        let unresolved = progress.records.iter().any(|record| {
            matches!(
                record.analysis,
                Analysis::PendingDerivation { .. }
                    | Analysis::Execution {
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

fn analyze<S: Space>(
    space: &S,
    execution: &S::Execution,
    lower_bound: u64,
    budget: u64,
) -> Result<Analysis, String> {
    match space.analyze(execution) {
        Ok(model) => {
            space.context().check_model(&model)?;
            let mut search = model.start_search()?;
            let outcome = search.advance(budget)?;
            Ok(Analysis::Execution { search, outcome })
        }
        Err(DerivationError::Exhausted(exhausted)) => Ok(Analysis::PendingDerivation {
            exhausted,
            lower_bound,
        }),
        Err(DerivationError::Analysis(message)) => Err(message),
    }
}

fn analysis_lower(analysis: &Analysis) -> Option<u64> {
    match analysis {
        Analysis::PendingDerivation { lower_bound, .. } => Some(*lower_bound),
        Analysis::Execution { outcome, .. } => schedule_lower(outcome),
        _ => None,
    }
}

fn schedule_lower(outcome: &SearchOutcome) -> Option<u64> {
    match outcome {
        SearchOutcome::Feasible(solution) => Some(solution.lower_bound()),
        SearchOutcome::Incomplete { lower_bound } => Some(*lower_bound),
        SearchOutcome::Infeasible => None,
    }
}
