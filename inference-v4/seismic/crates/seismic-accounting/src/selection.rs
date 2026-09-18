//! Selection over the implementation's own choices and derived constraints.
//!
//! Search state is private. Bounds, coverage and schedule feasibility are checked
//! where they are computed, without a second proof graph or certificate language.
use crate::schedule::{
    self,
    evaluation::{Model, Outcome as SearchOutcome, Search, Solution},
};
use crate::workload::{DerivationError, DerivationLimit};
use std::collections::{BTreeSet, HashMap};
mod domain;
pub use domain::{Choices, Domain, ExcludedRegion, IntegerRange, Region};

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
        self.check_timebase(model.timebase())?;
        model.relationship().require_feasible_upper()
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
    /// Legal region whose realization needs an unavailable compiler analysis.
    Unresolved(String),
}
pub trait Space {
    type Execution;
    /// Exact immutable inputs owned by this implementation space. Display names
    /// are not cache identity. Include source, legal form, workload, and hardware.
    type Identity: Clone + PartialEq;
    fn identity(&self) -> Self::Identity;
    fn context(&self) -> &Context;
    fn expand(&self, prefix: &[usize]) -> Result<Node<Self::Execution>, String>;
    /// Construct a selected member directly from its retained choice owner.
    /// This changes construction cost only: the result must equal expansion of
    /// the corresponding full path under the same immutable space identity.
    /// Owners without retained construction context return None.
    fn refine(
        &self,
        _alternatives: &Domain,
        _index: usize,
    ) -> Result<Option<Node<Self::Execution>>, String> {
        Ok(None)
    }
    /// Necessary resource demand of the partial implementation in the domain,
    /// relaxed over every indexed alternative. No arbitrary scalar score.
    fn relax(
        &self,
        _alternatives: &Domain,
        _indices: std::ops::Range<usize>,
    ) -> Result<Option<schedule::Demand>, String> {
        Ok(None)
    }
    /// Necessary demand of one completed execution before allocating its
    /// scheduling graph. Derive it from the same implementation operations;
    /// unavailable or budget-limited analysis returns None, never infeasibility.
    fn relax_execution(
        &self,
        _execution: &Self::Execution,
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
    pub fn flat(&self) -> Result<(&schedule::Model, &schedule::Schedule), String> {
        self.solution
            .flat()
            .ok_or_else(|| "objective retains a structured schedule".into())
    }
    pub fn structured(&self) -> Option<&schedule::structured::Witness> {
        self.solution.structured()
    }
    pub fn check_execution_upper(&self) -> Result<(), String> {
        self.solution.check_execution_upper()
    }
    pub fn cost(&self) -> Cost {
        Cost {
            lower: self.solution.lower_bound(),
            upper: self.solution.completion(),
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
    /// Additional start-time regions per distinct derived model. Equal models
    /// share one retained frontier, without identifying their source executions.
    pub schedule_assignments: u64,
}
enum Analysis {
    Choice(domain::WeakDomain),
    Unresolved { reason: String, lower_bound: u64 },
    PendingDerivation {
        exhausted: DerivationLimit,
        lower_bound: u64,
    },
    Execution {
        schedule: usize,
        lower_bound: u64,
    },
    /// This leaf cannot improve the incumbent under the unchanged search
    /// identity. Retain its derived bound, not its entire scheduling frontier.
    Excluded {
        lower_bound: u64,
    },
    Unschedulable,
    Infeasible(CapacityViolation),
}
/// Several legal paths may derive exactly the same scheduling constraints.
/// Their coverage, source execution and region bounds remain separate; only
/// this already-derived subproblem is shared. Unreferenced searches are freed.
struct ScheduleAnalysis {
    search: Search,
    outcome: SearchOutcome,
    expansion_limit: Option<u64>,
    references: usize,
    advanced_at: u64,
    hash: u64,
}
#[derive(Default)]
struct Schedules {
    entries: Vec<Option<ScheduleAnalysis>>,
    free: Vec<usize>,
    by_hash: HashMap<u64, Vec<usize>>,
}
impl Schedules {
    fn get(&self, index: usize) -> &ScheduleAnalysis {
        self.entries[index]
            .as_ref()
            .expect("retained scheduling analysis")
    }
    fn get_mut(&mut self, index: usize) -> &mut ScheduleAnalysis {
        self.entries[index]
            .as_mut()
            .expect("retained scheduling analysis")
    }
    fn release(&mut self, index: usize) {
        let entry = self.get_mut(index);
        entry.references -= 1;
        if entry.references == 0 {
            let hash = entry.hash;
            self.entries[index] = None;
            self.free.push(index);
            let bucket = self
                .by_hash
                .get_mut(&hash)
                .expect("indexed scheduling model");
            bucket.retain(|&i| i != index);
            if bucket.is_empty() {
                self.by_hash.remove(&hash);
            }
        }
    }
    fn derive(&mut self, model: Model, budget: u64, epoch: u64) -> Result<usize, String> {
        let expansion_limit = match &model {
            Model::Flat(_) => None,
            Model::Structured {
                expansion_limit, ..
            } => Some(*expansion_limit),
        };
        let hash = model.sharing_hash();
        for &index in self.by_hash.get(&hash).into_iter().flatten() {
            let entry = self.entries[index]
                .as_mut()
                .expect("indexed scheduling model");
            if entry.expansion_limit == expansion_limit && entry.search.matches_model(&model) {
                entry.references += 1;
                return Ok(index);
            }
        }
        let mut search = model.start_search()?;
        let outcome = search.advance(budget)?;
        let entry = ScheduleAnalysis {
            search,
            outcome,
            expansion_limit,
            references: 1,
            advanced_at: epoch,
            hash,
        };
        let index = if let Some(index) = self.free.pop() {
            self.entries[index] = Some(entry);
            index
        } else {
            self.entries.push(Some(entry));
            self.entries.len() - 1
        };
        self.by_hash.entry(hash).or_default().push(index);
        Ok(index)
    }
}

/// Indexed priority queue over retained, disjoint implementation regions.
/// Before the first witness, depth prevents weak shallow bounds from starving
/// complete construction. Bounds order peers and all regions after that witness.
/// The vector keeps the existing read-only frontier interface, while the index
/// avoids rescanning and reallocating every path at every domain subdivision.
struct Frontier {
    regions: Vec<Region>,
    order: BTreeSet<(u64, Vec<usize>, usize)>,
    depth: BTreeSet<(std::cmp::Reverse<usize>, u64, Vec<usize>, usize)>,
}
impl Frontier {
    fn new() -> Self {
        Self {
            regions: Vec::new(),
            order: BTreeSet::new(),
            depth: BTreeSet::new(),
        }
    }
    fn push(&mut self, region: Region) {
        let path = region.first_path();
        self.depth.insert((std::cmp::Reverse(path.len()), region.lower_bound(), path.clone(), self.regions.len()));
        self.order.insert((
            region.lower_bound(),
            path,
            self.regions.len(),
        ));
        self.regions.push(region);
    }
    fn pop(&mut self, has_incumbent: bool) -> Option<Region> {
        let index = if has_incumbent {
            let (bound, path, index) = self.order.pop_first()?;
            self.depth.remove(&(std::cmp::Reverse(path.len()), bound, path, index));
            index
        } else {
            // A weak bound on every shallow sibling must not prevent reaching
            // any complete implementation. Refine one dependent path first;
            // within a depth, objective bounds still order the legal regions.
            let (_, bound, path, index) = self.depth.pop_first()?;
            self.order.remove(&(bound, path, index));
            index
        };
        let last = self.regions.len() - 1;
        let region = self.regions.swap_remove(index);
        if index != last {
            let moved = &self.regions[index];
            self.order
                .remove(&(moved.lower_bound(), moved.first_path(), last));
            self.order
                .insert((moved.lower_bound(), moved.first_path(), index));
            let path = moved.first_path();
            self.depth.remove(&(std::cmp::Reverse(path.len()), moved.lower_bound(), path.clone(), last));
            self.depth.insert((std::cmp::Reverse(path.len()), moved.lower_bound(), path, index));
        }
        Some(region)
    }
    fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}
struct Record<E> {
    path: Vec<usize>,
    /// Retain the actual owning domain for leaf recovery after schedule resume.
    /// Its ordinal still belongs to the same full decision path.
    refinement: Option<(Domain, usize)>,
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
    pending: Frontier,
    excluded: Vec<ExcludedRegion>,
    records: Vec<Record<E>>,
    incumbent: Option<Incumbent<E>>,
    schedules: Schedules,
    epoch: u64,
}
impl<E, I> Progress<E, I> {
    pub fn retained_schedule_models(&self) -> usize {
        self.schedules
            .entries
            .iter()
            .filter(|entry| entry.is_some())
            .count()
    }

    fn retire(&mut self, index: usize) {
        if self
            .incumbent
            .as_ref()
            .is_some_and(|best| best.record == index)
        {
            return;
        }
        let record = &mut self.records[index];
        if matches!(record.analysis, Analysis::Excluded { .. }) {
            record.refinement = None;
            record.deferred_execution = None;
            return;
        }
        let summary = match &record.analysis {
            Analysis::Execution { schedule, .. }
                if matches!(
                    self.schedules.get(*schedule).outcome,
                    SearchOutcome::Infeasible
                ) =>
            {
                Some(Analysis::Unschedulable)
            }
            Analysis::Execution { .. } | Analysis::PendingDerivation { .. } | Analysis::Unresolved { .. } => {
                analysis_lower(&record.analysis, &self.schedules)
                    .filter(|&lower| {
                        self.incumbent
                            .as_ref()
                            .is_some_and(|best| lower >= best.upper)
                    })
                    .map(|lower_bound| Analysis::Excluded { lower_bound })
            }
            _ => None,
        };
        if let Some(summary) = summary {
            if let Analysis::Execution { schedule, .. } = record.analysis {
                self.schedules.release(schedule);
            }
            record.analysis = summary;
            record.refinement = None;
            record.deferred_execution = None;
        }
    }
    pub fn nodes_visited(&self) -> usize {
        self.records.len()
    }
    pub fn frontier(&self) -> &[Region] {
        &self.pending.regions
    }
    pub fn excluded_regions(&self) -> &[ExcludedRegion] {
        &self.excluded
    }
    /// Inspect a decision while some unresolved member or recoverable execution
    /// still retains its owner. Completed exclusions retain coverage separately.
    pub fn decision(&self, path: &[usize]) -> Option<Domain> {
        self.records
            .iter()
            .find_map(|record| match &record.analysis {
                Analysis::Choice(domain) if record.path == path => domain.upgrade(),
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
            choice_regions: self.pending.regions.len(),
            ..Unresolved::default()
        };
        for record in &self.records {
            if analysis_lower(&record.analysis, &self.schedules).is_some_and(|lower| {
                self.incumbent
                    .as_ref()
                    .is_some_and(|best| lower >= best.upper)
            }) {
                continue;
            }
            match &record.analysis {
                Analysis::Unresolved { .. } => unresolved.unsupported += 1,
                Analysis::PendingDerivation { .. } => unresolved.derivations += 1,
                Analysis::Execution { schedule, .. } => {
                    match &self.schedules.get(*schedule).outcome {
                        SearchOutcome::Infeasible => {}
                        SearchOutcome::Feasible(solution) if solution.is_optimal() => {}
                        _ if !self.schedules.get(*schedule).search.unmapped().is_empty() => {
                            unresolved.unmapped_models += 1
                        }
                        _ => unresolved.schedules += 1,
                    }
                }
                _ => {}
            }
        }
        unresolved
    }
    /// Explicitly unavailable realization/model analyses, with their retained
    /// choice paths. Identical resumption preserves these unresolved regions.
    pub fn unsupported_analyses(&self) -> impl Iterator<Item = (&[usize], &str)> {
        self.records.iter().filter_map(|record| match &record.analysis {
            Analysis::Unresolved { reason, lower_bound }
                if self.incumbent.as_ref().is_none_or(|best| *lower_bound < best.upper) =>
                    Some((record.path.as_slice(), reason.as_str())),
            _ => None,
        })
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
    /// Missing model facts for unresolved executions. This exposes analysis
    /// failures without granting access to an unselected executable.
    pub fn missing_mappings(&self) -> impl Iterator<Item = (&[usize], &[String])> {
        self.records
            .iter()
            .filter_map(|record| match &record.analysis {
                Analysis::Execution {
                    schedule,
                    lower_bound,
                    ..
                } if !self.schedules.get(*schedule).search.unmapped().is_empty()
                    && self
                        .incumbent
                        .as_ref()
                        .is_none_or(|best| *lower_bound < best.upper) =>
                {
                    Some((
                        record.path.as_slice(),
                        self.schedules.get(*schedule).search.unmapped(),
                    ))
                }
                _ => None,
            })
    }
    pub fn lower_bound(&self) -> Result<u64, String> {
        self.records
            .iter()
            .filter_map(|record| analysis_lower(&record.analysis, &self.schedules))
            .chain(self.pending.regions.iter().map(Region::lower_bound))
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
    pub unsupported: usize,
}
pub enum Outcome<E, I> {
    Optimal(Selected<E>),
    Incomplete(Progress<E, I>),
    Infeasible,
}

fn construct<S: Space>(
    space: &S,
    path: &[usize],
    refinement: Option<&(Domain, usize)>,
) -> Result<Node<S::Execution>, String> {
    if let Some((domain, index)) = refinement {
        if *index >= domain.len() {
            return Err("refinement ordinal is outside its retained domain".into());
        }
        if let Some(node) = space.refine(domain, *index)? {
            return Ok(node);
        }
    }
    space.expand(path)
}

pub fn select<S: Space>(
    space: &S,
    budget: Budget,
) -> Result<Outcome<S::Execution, S::Identity>, String> {
    space.context().validate()?;
    let mut pending = Frontier::new();
    pending.push(Region::root());
    resume(
        space,
        Progress {
            context: space.context().clone(),
            identity: space.identity(),
            pending,
            excluded: Vec::new(),
            records: Vec::new(),
            incumbent: None,
            schedules: Schedules::default(),
            epoch: 0,
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
    progress.epoch = progress
        .epoch
        .checked_add(1)
        .ok_or("selection resumption overflow")?;
    let epoch = progress.epoch;
    // A larger analysis budget may strengthen existing bounds. Refresh each
    // retained region once, before ordering it alongside new subdivisions.
    let previous = std::mem::replace(&mut progress.pending, Frontier::new());
    for mut region in previous.regions {
        relax_region(space, &progress.context, &mut region)?;
        progress.pending.push(region);
    }
    // Refine the most promising established bounds first, sharing equal
    // derived scheduling problems even when their implementation paths differ.
    let mut resumable = progress.records.iter().enumerate()
        .filter(|(_, record)| matches!(record.analysis, Analysis::PendingDerivation { .. } | Analysis::Execution { .. }))
        .map(|(index, _)| index).collect::<Vec<_>>();
    resumable.sort_by(|&a, &b| {
        analysis_lower(&progress.records[a].analysis, &progress.schedules)
            .cmp(&analysis_lower(
                &progress.records[b].analysis,
                &progress.schedules,
            ))
            .then_with(|| progress.records[a].path.cmp(&progress.records[b].path))
    });
    for index in resumable {
        let record = &mut progress.records[index];
        let excluded = analysis_lower(&record.analysis, &progress.schedules).is_some_and(|lower| {
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
                        progress.incumbent.as_ref().map(|best| best.upper),
                        &mut progress.schedules,
                        epoch,
                    )?;
                }
                Analysis::Execution { schedule, .. } => {
                    let shared = progress.schedules.get_mut(*schedule);
                    if shared.advanced_at != epoch {
                        shared.advanced_at = epoch;
                        if !matches!(&shared.outcome, SearchOutcome::Infeasible)
                            && !matches!(&shared.outcome, SearchOutcome::Feasible(solution) if solution.is_optimal())
                            && shared.search.unmapped().is_empty()
                        {
                            shared.outcome = shared.search.advance(budget.schedule_assignments)?;
                        }
                    }
                }
                _ => {}
            }
        }
        if let Analysis::Execution {
            schedule,
            lower_bound,
        } = &record.analysis
        {
            let outcome = &progress.schedules.get(*schedule).outcome;
            check_region_bound(outcome, *lower_bound)?;
            if let SearchOutcome::Feasible(solution) = outcome {
                let upper = solution.completion();
                if progress.incumbent.as_ref().is_none_or(|i| upper < i.upper) {
                    let execution = match record.deferred_execution.take() {
                        Some(execution) => execution,
                        None => {
                            let Node::Realization(execution) =
                                construct(space, &record.path, record.refinement.as_ref())?
                            else {
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
        progress.retire(index);
    }
    // Refine each selected region to a leaf before choosing a different region
    // by its global bound. Otherwise every positive-bound child can be starved
    // by shallow siblings even after a first, poor incumbent was discovered.
    // The deferred siblings retain their full coverage and their own bounds.
    let mut descent = None;
    for _ in 0..budget.nodes {
        let next = loop {
            let Some(region) = descent.take().or_else(|| progress.pending.pop(progress.incumbent.is_some())) else {
                break None;
            };
            if progress
                .incumbent
                .as_ref()
                .is_some_and(|i| region.lower_bound() >= i.upper)
            {
                progress.excluded.push(region.exclude());
                continue;
            }
            match region.split() {
                Ok((left, right)) => {
                    // Every child owns a disjoint complete interval. Compute its
                    // family relaxation once; popping it does not repeat analysis.
                    let (mut left, mut right) = (left, right);
                    relax_region(space, &progress.context, &mut left)?;
                    relax_region(space, &progress.context, &mut right)?;
                    if (right.lower_bound(), right.first_path()) < (left.lower_bound(), left.first_path()) {
                        std::mem::swap(&mut left, &mut right);
                    }
                    descent = Some(left);
                    progress.pending.push(right);
                }
                Err(region) => break Some(region),
            }
        };
        let Some(region) = next else { break };
        let path = region.first_path();
        let lower_bound = region.lower_bound();
        let mut refinement = region.alternatives().map(|(domain, indices)| {
            debug_assert_eq!(indices.len(), 1);
            (domain.clone(), indices.start)
        });
        let mut deferred_execution = None;
        let analysis = match construct(space, &path, refinement.as_ref())? {
            Node::Unresolved(reason) => {
                if reason.is_empty() { return Err("unresolved implementation has no analysis reason".into()); }
                refinement = None;
                Analysis::Unresolved { reason, lower_bound }
            }
            Node::Choice { alternatives, .. } => {
                alternatives.validate()?;
                let mut region = Region::children(path.clone(), alternatives.clone(), lower_bound);
                relax_region(space, &progress.context, &mut region)?;
                descent = Some(region);
                refinement = None;
                Analysis::Choice(alternatives.downgrade())
            }
            Node::Infeasible(violation) => {
                violation.validate()?;
                refinement = None;
                Analysis::Infeasible(violation)
            }
            Node::Realization(execution) => {
                let analysis = analyze(
                    space,
                    &execution,
                    lower_bound,
                    budget.schedule_assignments,
                    progress.incumbent.as_ref().map(|best| best.upper),
                    &mut progress.schedules,
                    epoch,
                )?;
                if matches!(analysis, Analysis::PendingDerivation { .. }) {
                    deferred_execution = Some(execution);
                } else if let Analysis::Execution { schedule, .. } = &analysis {
                    if let SearchOutcome::Feasible(solution) =
                        &progress.schedules.get(*schedule).outcome
                    {
                        let upper = solution.completion();
                        if progress.incumbent.as_ref().is_none_or(|i| upper < i.upper) {
                            progress.incumbent = Some(Incumbent {
                                record: progress.records.len(),
                                execution,
                                upper,
                            });
                        }
                    }
                }
                analysis
            }
        };
        progress.records.push(Record {
            path,
            refinement,
            analysis,
            deferred_execution,
        });
        progress.retire(progress.records.len() - 1);
    }
    if let Some(region) = descent { progress.pending.push(region); }
    for index in 0..progress.records.len() {
        progress.retire(index);
    }
    if let Some(best) = &progress.incumbent {
        let previous = std::mem::replace(&mut progress.pending, Frontier::new());
        for region in previous.regions {
            if region.lower_bound() >= best.upper {
                progress.excluded.push(region.exclude());
            } else {
                progress.pending.push(region);
            }
        }
    }
    if !progress.pending.is_empty() {
        return Ok(Outcome::Incomplete(progress));
    }
    let Some(best) = &progress.incumbent else {
        let unresolved = progress
            .records
            .iter()
            .any(|record| match &record.analysis {
                Analysis::PendingDerivation { .. } | Analysis::Unresolved { .. } => true,
                Analysis::Execution { schedule, .. } => matches!(
                    progress.schedules.get(*schedule).outcome,
                    SearchOutcome::Incomplete { .. }
                ),
                _ => false,
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
    let Analysis::Execution { schedule, .. } = record.analysis else {
        unreachable!("incumbent owns a feasible scheduling result")
    };
    let SearchOutcome::Feasible(solution) = &progress.schedules.get(schedule).outcome else {
        unreachable!("incumbent owns a feasible scheduling result")
    };
    let objective = Objective {
        solution: solution.clone(),
    };
    objective.check_execution_upper()?;
    let execution = space.materialize(&best.execution, &objective)?;
    Ok(Outcome::Optimal(Selected {
        execution,
        path: record.path,
        objective,
    }))
}

fn relax_region<S: Space>(space: &S, context: &Context, region: &mut Region) -> Result<(), String> {
    if let Some((alternatives, indices)) = region.alternatives() {
        if let Some(demand) = space.relax(alternatives, indices)? {
            context.check_timebase(demand.timebase())?;
            region.strengthen(&demand)?;
        }
    }
    Ok(())
}

fn analyze<S: Space>(
    space: &S,
    execution: &S::Execution,
    mut lower_bound: u64,
    budget: u64,
    incumbent: Option<u64>,
    schedules: &mut Schedules,
    epoch: u64,
) -> Result<Analysis, String> {
    if let Some(upper) = incumbent {
        if let Some(demand) = space.relax_execution(execution)? {
            space.context().check_timebase(demand.timebase())?;
            lower_bound = lower_bound.max(demand.lower_bound()?);
            if lower_bound >= upper {
                return Ok(Analysis::Excluded { lower_bound });
            }
        }
    }
    match space.analyze(execution) {
        Ok(model) => {
            space.context().check_model(&model)?;
            // The complete model can expose dependencies and resident demand
            // omitted by the cheaper implementation relaxation. Exclude it
            // before allocating or refining any start-time frontier.
            lower_bound = lower_bound.max(model.lower_bound()?);
            if incumbent.is_some_and(|upper| lower_bound >= upper) {
                return Ok(Analysis::Excluded { lower_bound });
            }
            let schedule = schedules.derive(model, budget, epoch)?;
            check_region_bound(&schedules.get(schedule).outcome, lower_bound)?;
            Ok(Analysis::Execution {
                schedule,
                lower_bound,
            })
        }
        Err(DerivationError::Exhausted(exhausted)) => Ok(Analysis::PendingDerivation {
            exhausted,
            lower_bound,
        }),
        Err(DerivationError::Unsupported(reason)) => {
            if reason.is_empty() { return Err("unsupported model has no analysis reason".into()); }
            Ok(Analysis::Unresolved { reason, lower_bound })
        }
        Err(DerivationError::Analysis(message)) => Err(message),
    }
}

fn analysis_lower(analysis: &Analysis, schedules: &Schedules) -> Option<u64> {
    match analysis {
        Analysis::PendingDerivation { lower_bound, .. } | Analysis::Unresolved { lower_bound, .. } | Analysis::Excluded { lower_bound } => {
            Some(*lower_bound)
        }
        Analysis::Execution {
            schedule,
            lower_bound,
        } => schedule_lower(&schedules.get(*schedule).outcome).map(|n| n.max(*lower_bound)),
        _ => None,
    }
}

fn check_region_bound(outcome: &SearchOutcome, lower_bound: u64) -> Result<(), String> {
    if let SearchOutcome::Feasible(solution) = outcome {
        if solution.completion() < lower_bound {
            return Err("feasible execution violates its retained region lower bound".into());
        }
    }
    Ok(())
}

fn schedule_lower(outcome: &SearchOutcome) -> Option<u64> {
    match outcome {
        SearchOutcome::Feasible(solution) => Some(solution.lower_bound()),
        SearchOutcome::Incomplete { lower_bound } => Some(*lower_bound),
        SearchOutcome::Infeasible => None,
    }
}
