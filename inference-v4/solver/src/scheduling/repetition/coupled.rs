//! Exact finite repetition with shared resources.
//!
//! Copies have distinct activity identities. Their only external relationships
//! are the declared shared cumulative capacities and translated precedence
//! edges. Thus permuting whole copies is valid only in the absence of edges
//! crossing copies. Compact algorithms below prove optima for named subclasses;
//! all other templates retain complete integer-time scheduling through `Search`.

use crate::model::{Constraint, Cost, Domain, LinearTerm, ModelBuilder, VarId};
use crate::result::StopReason;
use crate::scheduling::{Activity, ActivityReservation, Demand, Event, SchedulingConstraint};
use crate::{Error, Limits, Options, Outcome, Result, Search, Stats};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

/// One mandatory nonpreemptive activity. Demand is indexed by resource and is
/// held on `[start, start + duration)`. Zero-duration activities use no capacity.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CoupledActivity {
    pub duration: u64,
    pub demand: Vec<u64>,
}

/// `before` in copy `i` finishes at least `lag` before `after` starts in copy
/// `i + distance`, for every pair of copies inside the finite repetition.
/// Distance zero describes an edge inside each body. Edges beyond the finite
/// tail have no instance; they are not wrapped or replaced by periodic edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RepeatedPrecedence {
    pub before: usize,
    pub after: usize,
    pub distance: u64,
    pub lag: u64,
}

/// A finite repeated scheduling family. Every copy may choose its own schedule;
/// no isolated-body optimum or common body profile restricts these choices.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CoupledTemplate {
    pub capacities: Vec<u64>,
    pub activities: Vec<CoupledActivity>,
    pub precedence: Vec<RepeatedPrecedence>,
}

impl CoupledTemplate {
    pub fn validate(&self) -> Result<()> {
        for activity in &self.activities {
            if activity.demand.len() != self.capacities.len() {
                return Err(Error::InvalidModel(
                    "repeated activity resource arity".into(),
                ));
            }
            i64::try_from(activity.duration)
                .map_err(|_| Error::Overflow("repeated activity duration".into()))?;
        }
        for edge in &self.precedence {
            if edge.before >= self.activities.len() || edge.after >= self.activities.len() {
                return Err(Error::InvalidModel(
                    "repeated precedence activity index".into(),
                ));
            }
        }
        Ok(())
    }

    /// Checks an externally supplied expanded schedule directly against this
    /// template, without using solver propagation or objective assessment.
    /// Starts are copy-major; all identities and finite-tail edges are retained.
    pub fn validate_starts(&self, count: u64, horizon: u64, starts: &[u64]) -> Result<u64> {
        self.validate()?;
        let copies =
            usize::try_from(count).map_err(|_| Error::Overflow("repetition copy count".into()))?;
        let width = self.activities.len();
        let length = copies
            .checked_mul(width)
            .ok_or_else(|| Error::Overflow("repeated activity count".into()))?;
        if starts.len() != length {
            return Err(Error::InvalidAssignment("repeated start count".into()));
        }
        let mut completion = 0;
        let mut resource_events = vec![BTreeMap::<u64, i128>::new(); self.capacities.len()];
        for copy in 0..copies {
            for (index, activity) in self.activities.iter().enumerate() {
                let start = starts[copy * width + index];
                let end = start
                    .checked_add(activity.duration)
                    .ok_or_else(|| Error::Overflow("repeated activity end".into()))?;
                if end > horizon {
                    return Err(Error::InvalidAssignment(
                        "repeated activity exceeds horizon".into(),
                    ));
                }
                completion = completion.max(end);
                if activity.duration != 0 {
                    for (resource, &demand) in activity.demand.iter().enumerate() {
                        for (time, change) in [(start, demand as i128), (end, -(demand as i128))] {
                            let value = resource_events[resource].entry(time).or_default();
                            *value = value.checked_add(change).ok_or_else(|| {
                                Error::Overflow("repeated capacity events".into())
                            })?;
                        }
                    }
                }
            }
            for edge in &self.precedence {
                // Subtraction avoids overflow for a distance larger than count.
                if edge.distance >= count - copy as u64 {
                    continue;
                }
                let after_copy = copy + edge.distance as usize;
                let before_end = starts[copy * width + edge.before] as u128
                    + self.activities[edge.before].duration as u128
                    + edge.lag as u128;
                if before_end > starts[after_copy * width + edge.after] as u128 {
                    return Err(Error::InvalidAssignment("repeated precedence".into()));
                }
            }
        }
        for (resource, events) in resource_events.iter().enumerate() {
            let mut usage = 0i128;
            for delta in events.values() {
                usage = usage
                    .checked_add(*delta)
                    .ok_or_else(|| Error::Overflow("repeated resource use".into()))?;
                if usage > self.capacities[resource] as i128 || usage < 0 {
                    return Err(Error::InvalidAssignment("repeated capacity".into()));
                }
            }
        }
        Ok(completion)
    }
}

/// Automatic reductions are exact. Expanded is useful for differential testing
/// and retains precisely the same family, including independently scheduled copies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RepetitionStrategy {
    #[default]
    Automatic,
    Expanded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepetitionMethod {
    Empty,
    IdenticalActivities,
    UnaryPipeline,
    FiniteJoint,
}

#[derive(Clone, Debug, Default)]
pub struct CoupledStats {
    pub construction_work: u64,
    pub constructed_copies: u64,
    pub expanded_variables: usize,
    pub expanded_factors: usize,
    pub retained_bytes: usize,
    pub peak_retained_bytes: usize,
    pub method: Option<RepetitionMethod>,
    pub search: Stats,
}

#[derive(Clone, Debug)]
enum Witness {
    Empty,
    Batches { width: u64 },
    Pipeline { offsets: Vec<u64>, period: u64 },
    Explicit { starts: Vec<u64> },
}

/// An independently validated schedule. Its type carries no optimality claim.
#[derive(Clone, Debug)]
pub struct CoupledIncumbent {
    template: Arc<CoupledTemplate>,
    count: u64,
    horizon: u64,
    cost: u64,
    witness: Witness,
}
impl CoupledIncumbent {
    pub fn template(&self) -> &CoupledTemplate {
        &self.template
    }
    pub fn count(&self) -> u64 {
        self.count
    }
    pub fn horizon(&self) -> u64 {
        self.horizon
    }
    pub fn cost(&self) -> u64 {
        self.cost
    }
    pub fn method(&self) -> RepetitionMethod {
        match self.witness {
            Witness::Empty => RepetitionMethod::Empty,
            Witness::Batches { .. } => RepetitionMethod::IdenticalActivities,
            Witness::Pipeline { .. } => RepetitionMethod::UnaryPipeline,
            Witness::Explicit { .. } => RepetitionMethod::FiniteJoint,
        }
    }
    /// Constant-space access even when the schedule has trillions of copies.
    pub fn start(&self, copy: u64, activity: usize) -> Option<u64> {
        if copy >= self.count || activity >= self.template.activities.len() {
            return None;
        }
        Some(match &self.witness {
            Witness::Empty => return None,
            Witness::Batches { width } => (copy / width) * self.template.activities[0].duration,
            Witness::Pipeline { offsets, period } => offsets[activity] + copy * period,
            Witness::Explicit { starts } => {
                starts[copy as usize * self.template.activities.len() + activity]
            }
        })
    }
    pub fn witness_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + match &self.witness {
                Witness::Pipeline { offsets, .. } => {
                    offsets.capacity() * std::mem::size_of::<u64>()
                }
                Witness::Explicit { starts } => starts.capacity() * std::mem::size_of::<u64>(),
                _ => 0,
            }
    }
    /// Validate original semantics. Compact witnesses are checked by their
    /// resource separation and arithmetic repetition invariant, not by expansion.
    pub fn validate(&self) -> Result<()> {
        self.template.validate()?;
        let invalid = || Error::Invariant("invalid repeated witness".into());
        let actual_cost = match &self.witness {
            Witness::Empty => {
                if self.count != 0 && !self.template.activities.is_empty() {
                    return Err(invalid());
                }
                0
            }
            Witness::Explicit { starts } => {
                self.template
                    .validate_starts(self.count, self.horizon, starts)?
            }
            Witness::Batches { width } => {
                if *width == 0
                    || self.count == 0
                    || self.template.activities.len() != 1
                    || self
                        .template
                        .precedence
                        .iter()
                        .any(|edge| edge.distance < self.count)
                {
                    return Err(invalid());
                }
                let activity = &self.template.activities[0];
                if activity.duration > 0
                    && activity.demand.iter().zip(&self.template.capacities).any(
                        |(&demand, &capacity)| *width as u128 * demand as u128 > capacity as u128,
                    )
                {
                    return Err(invalid());
                }
                self.count
                    .div_ceil(*width)
                    .checked_mul(activity.duration)
                    .ok_or_else(|| Error::Overflow("repeated batch completion".into()))?
            }
            Witness::Pipeline { offsets, period } => {
                if self.count == 0 || offsets.len() != self.template.activities.len() {
                    return Err(invalid());
                }
                // Each resource belongs to at most one stage, and occurrences
                // of that stage never overlap because duration <= period.
                let mut owners = vec![None; self.template.capacities.len()];
                let mut completion = 0;
                for (stage, activity) in self.template.activities.iter().enumerate() {
                    if activity.duration > *period {
                        return Err(invalid());
                    }
                    for (resource, &demand) in activity.demand.iter().enumerate() {
                        if demand == 0 || activity.duration == 0 {
                            continue;
                        }
                        if demand > self.template.capacities[resource] || owners[resource].is_some()
                        {
                            return Err(invalid());
                        }
                        owners[resource] = Some(stage);
                    }
                    completion =
                        completion.max(
                            offsets[stage]
                                .checked_add((self.count - 1).checked_mul(*period).ok_or_else(
                                    || Error::Overflow("pipeline repeated start".into()),
                                )?)
                                .and_then(|start| start.checked_add(activity.duration))
                                .ok_or_else(|| Error::Overflow("pipeline completion".into()))?,
                        );
                }
                for edge in &self.template.precedence {
                    if edge.distance >= self.count {
                        continue;
                    }
                    let required = offsets[edge.before] as u128
                        + self.template.activities[edge.before].duration as u128
                        + edge.lag as u128;
                    let available =
                        offsets[edge.after] as u128 + edge.distance as u128 * *period as u128;
                    if required > available {
                        return Err(invalid());
                    }
                }
                completion
            }
        };
        if actual_cost != self.cost || self.cost > self.horizon {
            return Err(invalid());
        }
        Ok(())
    }
}

/// Constructed only after a complete finite proof or a proved exact reduction.
#[derive(Clone, Debug)]
pub struct CoupledSolution {
    feasible: CoupledIncumbent,
}
impl CoupledSolution {
    pub fn feasible(&self) -> &CoupledIncumbent {
        &self.feasible
    }
    pub fn cost(&self) -> u64 {
        self.feasible.cost()
    }
    pub fn count(&self) -> u64 {
        self.feasible.count()
    }
    pub fn start(&self, copy: u64, activity: usize) -> Option<u64> {
        self.feasible.start(copy, activity)
    }
    pub fn method(&self) -> RepetitionMethod {
        self.feasible.method()
    }
    pub fn validate(&self) -> Result<()> {
        self.feasible.validate()
    }
}

#[derive(Clone, Debug)]
pub struct CoupledProgress {
    pub lower_bound: u64,
    pub incumbent: Option<CoupledIncumbent>,
    pub reason: StopReason,
    pub stats: CoupledStats,
}
#[derive(Clone, Debug)]
pub enum CoupledOutcome {
    Optimal(CoupledSolution),
    Infeasible,
    Incomplete(CoupledProgress),
}

struct Expansion {
    builder: ModelBuilder,
    completion: VarId,
    durations: Vec<VarId>,
    activities: Vec<Activity>,
    copies: u64,
}
impl Expansion {
    fn new(template: &CoupledTemplate, horizon: u64) -> Result<Self> {
        let mut builder = ModelBuilder::new();
        let completion = builder.variable("completion", Domain::interval(0, horizon as i64)?);
        builder.cost(Cost::Linear {
            constant: 0,
            terms: vec![LinearTerm::new(completion, 1)],
        });
        let durations = template
            .activities
            .iter()
            .enumerate()
            .map(|(i, activity)| {
                builder.variable(
                    format!("duration_{i}"),
                    Domain::singleton(activity.duration as i64),
                )
            })
            .collect();
        Ok(Self {
            builder,
            completion,
            durations,
            activities: Vec::new(),
            copies: 0,
        })
    }
    fn append(&mut self, template: &CoupledTemplate, horizon: u64) -> Result<()> {
        let width = template.activities.len();
        let copy = usize::try_from(self.copies)
            .map_err(|_| Error::Overflow("expanded copy index".into()))?;
        copy.checked_add(1)
            .and_then(|copies| copies.checked_mul(width))
            .ok_or_else(|| Error::Overflow("expanded activity count".into()))?;
        for (stage, &duration) in self.durations.iter().enumerate() {
            let start = self.builder.variable(
                format!("start_{copy}_{stage}"),
                Domain::interval(0, horizon as i64)?,
            );
            let end = self.builder.variable(
                format!("end_{copy}_{stage}"),
                Domain::interval(0, horizon as i64)?,
            );
            let activity = Activity {
                start,
                duration,
                end,
                presence: None,
            };
            self.builder
                .constraint(Constraint::Schedule(SchedulingConstraint::Activity(
                    activity.clone(),
                )));
            self.builder.constraint(Constraint::LinearLe {
                terms: vec![
                    LinearTerm::new(end, 1),
                    LinearTerm::new(self.completion, -1),
                ],
                rhs: 0,
            });
            self.activities.push(activity);
        }
        for edge in &template.precedence {
            if edge.distance > self.copies {
                continue;
            }
            let before =
                self.activities[(copy - edge.distance as usize) * width + edge.before].end_event();
            let after = self.activities[copy * width + edge.after].start_event();
            self.builder
                .constraint(Constraint::Schedule(SchedulingConstraint::Precedence {
                    before,
                    after,
                    lag: edge.lag,
                }));
        }
        // Whole-copy label permutations preserve all constraints exactly when
        // there is no cross-copy edge. Sorting by the first start retains a
        // representative of every feasible schedule, including equal starts.
        if copy > 0 && !template.precedence.iter().any(|edge| edge.distance != 0) {
            self.builder
                .constraint(Constraint::Schedule(SchedulingConstraint::Precedence {
                    before: Event::mandatory(self.activities[(copy - 1) * width].start),
                    after: Event::mandatory(self.activities[copy * width].start),
                    lag: 0,
                }));
        }
        self.copies += 1;
        Ok(())
    }
    fn finish(mut self, template: &CoupledTemplate, options: Options) -> Result<Expanded> {
        let width = template.activities.len();
        for (resource, &capacity) in template.capacities.iter().enumerate() {
            let activities = self
                .activities
                .iter()
                .enumerate()
                .filter_map(|(index, activity)| {
                    let demand = template.activities[index % width].demand[resource];
                    (demand != 0).then(|| ActivityReservation {
                        activity: activity.clone(),
                        demand: Demand::Constant(demand),
                    })
                })
                .collect::<Vec<_>>();
            if !activities.is_empty() {
                self.builder.constraint(Constraint::Schedule(
                    SchedulingConstraint::ActivityCumulative {
                        completion: self.completion,
                        capacity,
                        activities,
                    },
                ));
            }
        }
        let starts = self
            .activities
            .iter()
            .map(|activity| activity.start)
            .collect();
        let model = Arc::new(self.builder.build()?);
        let model_bytes = model.retained_bytes();
        let variables = model.variables().len();
        let factors = model.factors().len();
        Ok(Expanded {
            search: Search::new(model, options)?,
            starts,
            model_bytes,
            variables,
            factors,
        })
    }
    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.durations.capacity() * std::mem::size_of::<VarId>()
            + self.activities.capacity() * std::mem::size_of::<Activity>()
            + self.builder.variables.capacity() * std::mem::size_of::<crate::model::Variable>()
            + self
                .builder
                .variables
                .iter()
                .map(|v| {
                    v.name.capacity() + v.domain.retained_bytes() - std::mem::size_of::<Domain>()
                })
                .sum::<usize>()
            + self.builder.factors.capacity() * std::mem::size_of::<crate::model::Factor>()
            + self
                .builder
                .factors
                .iter()
                .map(|f| f.retained_bytes() - std::mem::size_of::<crate::model::Factor>())
                .sum::<usize>()
    }
}
struct Expanded {
    search: Search,
    starts: Vec<VarId>,
    model_bytes: usize,
    variables: usize,
    factors: usize,
}
enum State {
    Pending,
    Expanding(Expansion),
    Searching(Expanded),
    Done(CoupledOutcome),
}

/// Solves minimum makespan within the supplied finite horizon. Horizon is an
/// input assumption, never a guessed cutoff. One construction unit expands one
/// body copy; subsequent refinement uses the core's work/time/memory budgets.
/// Generated model storage counts toward this wrapper's retained-memory target.
/// The immutable input template and allocator overhead are excluded.
pub struct CoupledRepetition {
    template: Arc<CoupledTemplate>,
    count: u64,
    horizon: u64,
    strategy: RepetitionStrategy,
    options: Options,
    state: State,
    stats: CoupledStats,
    lower_bound: u64,
    incumbent: Option<CoupledIncumbent>,
    error: Option<Error>,
}
impl CoupledRepetition {
    pub fn new(
        template: Arc<CoupledTemplate>,
        count: u64,
        horizon: u64,
        strategy: RepetitionStrategy,
        options: Options,
    ) -> Result<Self> {
        template.validate()?;
        i64::try_from(horizon).map_err(|_| Error::Overflow("repetition horizon".into()))?;
        Ok(Self {
            template,
            count,
            horizon,
            strategy,
            options,
            state: State::Pending,
            stats: CoupledStats::default(),
            lower_bound: 0,
            incumbent: None,
            error: None,
        })
    }
    pub fn stats(&self) -> &CoupledStats {
        &self.stats
    }
    pub fn advance(&mut self, limits: Limits) -> Result<CoupledOutcome> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        let result = self.advance_inner(limits);
        if let Err(error) = &result {
            self.error = Some(error.clone());
        }
        result
    }
    fn advance_inner(&mut self, limits: Limits) -> Result<CoupledOutcome> {
        let began = Instant::now();
        let mut remaining = limits.work;
        loop {
            self.update_stats();
            if let State::Done(outcome) = &self.state {
                return Ok(outcome.clone());
            }
            if !matches!(self.state, State::Searching(_))
                && limits
                    .memory_bytes
                    .is_some_and(|cap| self.stats.retained_bytes >= cap)
            {
                return Ok(self.incomplete(StopReason::Memory, None));
            }
            if remaining == 0 {
                return Ok(self.incomplete(StopReason::Work, None));
            }
            if limits.time.is_some_and(|limit| began.elapsed() >= limit) {
                return Ok(self.incomplete(StopReason::Time, None));
            }
            match &mut self.state {
                State::Pending => {
                    remaining -= 1;
                    self.stats.construction_work += 1;
                    if self.count == 0 || self.template.activities.is_empty() {
                        self.complete(Witness::Empty, 0)?;
                        continue;
                    }
                    self.lower_bound = energy_bound(&self.template, self.count)?;
                    if self.lower_bound > self.horizon
                        || self.template.activities.iter().any(|activity| {
                            activity.duration > 0
                                && activity
                                    .demand
                                    .iter()
                                    .zip(&self.template.capacities)
                                    .any(|(d, c)| d > c)
                        })
                    {
                        self.state = State::Done(CoupledOutcome::Infeasible);
                        continue;
                    }
                    if self.strategy == RepetitionStrategy::Automatic {
                        if let Some((witness, optimum)) = reduce(&self.template, self.count)? {
                            self.lower_bound = optimum;
                            if optimum > self.horizon {
                                self.state = State::Done(CoupledOutcome::Infeasible);
                            } else {
                                self.complete(witness, optimum)?;
                            }
                            continue;
                        }
                    }
                    self.stats.method = Some(RepetitionMethod::FiniteJoint);
                    self.state = State::Expanding(Expansion::new(&self.template, self.horizon)?);
                }
                State::Expanding(expansion) if expansion.copies < self.count => {
                    expansion.append(&self.template, self.horizon)?;
                    self.stats.construction_work += 1;
                    remaining -= 1;
                }
                State::Expanding(_) => {
                    let State::Expanding(expansion) =
                        std::mem::replace(&mut self.state, State::Pending)
                    else {
                        unreachable!()
                    };
                    self.state =
                        State::Searching(expansion.finish(&self.template, self.options.clone())?);
                    self.stats.construction_work += 1;
                    remaining -= 1;
                }
                State::Searching(expanded) => {
                    let overhead = expanded.model_bytes
                        + expanded.starts.capacity() * std::mem::size_of::<VarId>()
                        + std::mem::size_of::<Self>()
                        + self
                            .incumbent
                            .as_ref()
                            .map_or(0, CoupledIncumbent::witness_bytes);
                    let outcome = expanded.search.advance(Limits {
                        work: remaining,
                        time: limits
                            .time
                            .map(|limit| limit.saturating_sub(began.elapsed())),
                        memory_bytes: limits.memory_bytes.map(|cap| cap.saturating_sub(overhead)),
                    })?;
                    self.stats.search = expanded.search.stats().clone();
                    // A terminal result drops the expanded search below. Record
                    // its peak before that state disappears, including the
                    // generated model retained throughout this advance.
                    self.stats.peak_retained_bytes = self
                        .stats
                        .peak_retained_bytes
                        .max(overhead + self.stats.search.peak_retained_bytes);
                    match outcome {
                        Outcome::Optimal(solution) => {
                            let starts = expanded
                                .starts
                                .iter()
                                .map(|v| solution.values()[v.0] as u64)
                                .collect();
                            let cost = solution.cost();
                            self.complete(Witness::Explicit { starts }, cost)?;
                        }
                        Outcome::Infeasible => self.state = State::Done(CoupledOutcome::Infeasible),
                        Outcome::Incomplete(progress) => {
                            self.lower_bound = self.lower_bound.max(progress.lower_bound);
                            let incumbent = progress
                                .incumbent
                                .map(|solution| {
                                    let starts = expanded
                                        .starts
                                        .iter()
                                        .map(|v| solution.values()[v.0] as u64)
                                        .collect();
                                    make_incumbent(
                                        self.template.clone(),
                                        self.count,
                                        self.horizon,
                                        Witness::Explicit { starts },
                                        solution.cost(),
                                    )
                                })
                                .transpose()?;
                            self.incumbent = incumbent.clone();
                            self.update_stats();
                            return Ok(self.incomplete(progress.reason, incumbent));
                        }
                    }
                }
                State::Done(_) => unreachable!(),
            }
        }
    }
    fn complete(&mut self, witness: Witness, cost: u64) -> Result<()> {
        let feasible = make_incumbent(
            self.template.clone(),
            self.count,
            self.horizon,
            witness,
            cost,
        )?;
        self.stats.method = Some(feasible.method());
        self.lower_bound = cost;
        self.incumbent = None;
        self.state = State::Done(CoupledOutcome::Optimal(CoupledSolution { feasible }));
        Ok(())
    }
    fn incomplete(
        &self,
        reason: StopReason,
        incumbent: Option<CoupledIncumbent>,
    ) -> CoupledOutcome {
        CoupledOutcome::Incomplete(CoupledProgress {
            lower_bound: self.lower_bound,
            incumbent: incumbent.or_else(|| self.incumbent.clone()),
            reason,
            stats: self.stats.clone(),
        })
    }
    fn update_stats(&mut self) {
        let bytes = match &self.state {
            State::Pending => 0,
            State::Expanding(expansion) => {
                self.stats.constructed_copies = expansion.copies;
                self.stats.expanded_variables = expansion.builder.variables.len();
                self.stats.expanded_factors = expansion.builder.factors.len();
                expansion.retained_bytes()
            }
            State::Searching(expanded) => {
                self.stats.constructed_copies = self.count;
                self.stats.expanded_variables = expanded.variables;
                self.stats.expanded_factors = expanded.factors;
                expanded.model_bytes
                    + expanded.starts.capacity() * std::mem::size_of::<VarId>()
                    + expanded.search.stats().retained_bytes
            }
            State::Done(CoupledOutcome::Optimal(solution)) => solution.feasible.witness_bytes(),
            State::Done(_) => 0,
        };
        self.stats.retained_bytes = std::mem::size_of::<Self>()
            + bytes
            + self
                .incumbent
                .as_ref()
                .map_or(0, CoupledIncumbent::witness_bytes);
        self.stats.peak_retained_bytes = self
            .stats
            .peak_retained_bytes
            .max(self.stats.retained_bytes);
    }
}

fn make_incumbent(
    template: Arc<CoupledTemplate>,
    count: u64,
    horizon: u64,
    witness: Witness,
    cost: u64,
) -> Result<CoupledIncumbent> {
    let result = CoupledIncumbent {
        template,
        count,
        horizon,
        cost,
        witness,
    };
    result.validate()?;
    Ok(result)
}

/// Independent copies still contribute all their resource work. This bound is
/// valid before any copy is expanded; it does not presume uniform scheduling.
fn energy_bound(template: &CoupledTemplate, count: u64) -> Result<u64> {
    let mut lower = template
        .activities
        .iter()
        .map(|a| a.duration)
        .max()
        .unwrap_or(0) as u128;
    for (resource, &capacity) in template.capacities.iter().enumerate() {
        let mut energy = 0u128;
        for activity in &template.activities {
            energy = energy
                .checked_add(activity.duration as u128 * activity.demand[resource] as u128)
                .ok_or_else(|| Error::Overflow("repeated body energy".into()))?;
        }
        if energy == 0 {
            continue;
        }
        if capacity == 0 {
            return Ok(u64::MAX);
        }
        let total = energy
            .checked_mul(count as u128)
            .ok_or_else(|| Error::Overflow("repeated total energy".into()))?;
        lower = lower.max(total.div_ceil(capacity as u128));
    }
    u64::try_from(lower).map_err(|_| Error::Overflow("repeated lower bound".into()))
}

fn reduce(template: &CoupledTemplate, count: u64) -> Result<Option<(Witness, u64)>> {
    let edges: Vec<_> = template
        .precedence
        .iter()
        .filter(|e| e.distance < count)
        .collect();
    if template.activities.len() == 1 && edges.is_empty() {
        let activity = &template.activities[0];
        let mut width = count;
        if activity.duration != 0 {
            for (&demand, &capacity) in activity.demand.iter().zip(&template.capacities) {
                if demand > 0 {
                    width = width.min(capacity / demand);
                }
            }
        }
        if width == 0 {
            return Err(Error::Invariant(
                "infeasible capacity reached batch reduction".into(),
            ));
        }
        let optimum = count
            .div_ceil(width)
            .checked_mul(activity.duration)
            .ok_or_else(|| Error::Overflow("identical activity optimum".into()))?;
        return Ok(Some((Witness::Batches { width }, optimum)));
    }
    // Identical jobs on distinct unary stages. Require the exact chain, not a
    // guessed common schedule for a richer body. An exclusive resource on each
    // stage proves its processing is serial; no resource may join two stages.
    // At a longest stage, the first copy needs at least its prefix duration
    // before starting; processing all copies consumes count * period; the last
    // copy still needs its suffix. This proves sum(durations) + (count - 1) *
    // period as a lower bound. Offsetting every next copy by period attains it.
    let size = template.activities.len();
    if size < 2 || edges.len() != size - 1 {
        return Ok(None);
    }
    if edges
        .iter()
        .any(|edge| edge.distance != 0 || edge.lag != 0 || edge.before + 1 != edge.after)
        || (0..size - 1).any(|stage| edges.iter().filter(|edge| edge.before == stage).count() != 1)
    {
        return Ok(None);
    }
    let mut used = vec![false; template.capacities.len()];
    for activity in &template.activities {
        let mut exclusive = activity.duration == 0;
        for (resource, &demand) in activity.demand.iter().enumerate() {
            if demand == 0 || activity.duration == 0 {
                continue;
            }
            if used[resource] {
                return Ok(None);
            }
            used[resource] = true;
            exclusive |= demand as u128 * 2 > template.capacities[resource] as u128;
        }
        if !exclusive {
            return Ok(None);
        }
    }
    let mut offsets = Vec::with_capacity(size);
    let mut serial = 0u64;
    let mut period = 0;
    for activity in &template.activities {
        offsets.push(serial);
        serial = serial
            .checked_add(activity.duration)
            .ok_or_else(|| Error::Overflow("pipeline startup/drain".into()))?;
        period = period.max(activity.duration);
    }
    let optimum = serial
        .checked_add(
            (count - 1)
                .checked_mul(period)
                .ok_or_else(|| Error::Overflow("pipeline repeated period".into()))?,
        )
        .ok_or_else(|| Error::Overflow("pipeline optimum".into()))?;
    Ok(Some((Witness::Pipeline { offsets, period }, optimum)))
}
