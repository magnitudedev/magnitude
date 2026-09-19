//! Exact bridge from one complete Seismic scheduling model to the independent
//! finite solver. This does not select an implementation family or produce Tuned
//! IR. Source/workload/mapping identity and execution authority remain in `Model`.
//!
//! A checked feasible schedule supplies a sufficient finite horizon: no better
//! makespan needs an operation outside that horizon. Reconstruction is checked
//! against the original model, including common static instruction orders.

use super::{Model, Point, Schedule, Solution};
use magnitude_solver::model::{Constraint, Cost, Domain, LinearTerm, Literal, ModelBuilder, VarId};
use magnitude_solver::scheduling::{Activity, Demand, Event, Lifetime, SchedulingConstraint};
use magnitude_solver::{Limits, Options, Outcome, Search, Stats};
use std::sync::Arc;

#[derive(Debug)]
pub enum Error {
    Model(String),
    Unsupported(String),
    Solver(magnitude_solver::Error),
    Reconstruction(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(message) => write!(formatter, "invalid schedule model: {message}"),
            Self::Unsupported(message) => {
                write!(formatter, "unsupported schedule translation: {message}")
            }
            Self::Solver(error) => write!(formatter, "independent solver: {error}"),
            Self::Reconstruction(message) => {
                write!(formatter, "schedule reconstruction: {message}")
            }
        }
    }
}
impl std::error::Error for Error {}
impl From<magnitude_solver::Error> for Error {
    fn from(error: magnitude_solver::Error) -> Self {
        Self::Solver(error)
    }
}

pub enum IndependentOutcome {
    Optimal(Solution),
    Infeasible,
    Incomplete {
        lower_bound: u64,
        incumbent: Option<Solution>,
        reason: magnitude_solver::result::StopReason,
        stats: Stats,
    },
}

/// Retains one fixed execution model, its translation and original-model witness.
pub struct IndependentSearch {
    model: Arc<Model>,
    search: Search,
    starts: Vec<VarId>,
    completion: VarId,
    incumbent: Option<Schedule>,
}

impl IndependentSearch {
    pub fn new(model: Arc<Model>, upper: Schedule, options: Options) -> Result<Self, Error> {
        model.require_complete().map_err(Error::Unsupported)?;
        model.check_schedule(&upper).map_err(Error::Model)?;
        let horizon = tick(upper.completion)?;
        let (translated, starts, completion) = translate(&model, horizon)?;
        let search = Search::new(Arc::new(translated), options)?;
        Ok(Self {
            model,
            search,
            starts,
            completion,
            incumbent: Some(upper),
        })
    }

    /// Complete finite scheduling domain without assuming that a serial order
    /// is feasible. Removing idle gaps contracts lifetimes without altering
    /// operation reservations; some feasible schedule therefore completes within
    /// the sum of operation latencies whenever the model is feasible.
    pub fn from_model(model: Arc<Model>, options: Options) -> Result<Self, Error> {
        model.require_complete().map_err(Error::Unsupported)?;
        let order = model.validate().map_err(Error::Model)?;
        let mut upper = Schedule {
            starts: vec![0; model.operations.len()],
            completion: 0,
        };
        for &index in &order {
            upper.starts[index] = upper.completion;
            upper.completion = upper
                .completion
                .checked_add(model.operations[index].latency)
                .ok_or_else(|| {
                    Error::Unsupported("finite scheduling horizon exceeds u64".into())
                })?;
        }
        let fallback_horizon = upper.completion;
        let mut incumbent = model.check_schedule(&upper).is_ok().then_some(upper);
        // Dependency-earliest times are only a seed: shared reservations,
        // lifetimes and static orders are checked by the original model. When
        // feasible, this bounds the complete improvement region much more
        // tightly than serializing every operation. It performs no selection.
        let mut earliest = Schedule {
            starts: vec![0; model.operations.len()],
            completion: 0,
        };
        for index in order {
            let operation = &model.operations[index];
            let mut start = 0;
            for &before in &operation.predecessors {
                start = start.max(
                    earliest.starts[before]
                        .checked_add(model.operations[before].latency)
                        .ok_or_else(|| Error::Unsupported("dependency time exceeds u64".into()))?,
                );
            }
            for &before in &operation.start_predecessors {
                start = start.max(earliest.starts[before]);
            }
            earliest.starts[index] = start;
            earliest.completion =
                earliest
                    .completion
                    .max(start.checked_add(operation.latency).ok_or_else(|| {
                        Error::Unsupported("dependency completion exceeds u64".into())
                    })?);
        }
        if incumbent
            .as_ref()
            .is_none_or(|upper| earliest.completion < upper.completion)
            && model.check_schedule(&earliest).is_ok()
        {
            incumbent = Some(earliest);
        }
        let horizon = tick(
            incumbent
                .as_ref()
                .map_or(fallback_horizon, |upper| upper.completion),
        )?;
        let (translated, starts, completion) = translate(&model, horizon)?;
        let search = Search::new(Arc::new(translated), options)?;
        Ok(Self {
            model,
            search,
            starts,
            completion,
            incumbent,
        })
    }

    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn translated_model(&self) -> &magnitude_solver::Model {
        self.search.model()
    }
    pub fn stats(&self) -> &Stats {
        self.search.stats()
    }

    pub fn advance(&mut self, limits: Limits) -> Result<IndependentOutcome, Error> {
        match self.search.advance(limits)? {
            Outcome::Optimal(solution) => {
                let schedule = self.reconstruct(solution.values(), solution.cost())?;
                self.incumbent = Some(schedule);
                Ok(IndependentOutcome::Optimal(
                    self.solution(solution.cost())?
                        .expect("validated optimum has a witness"),
                ))
            }
            Outcome::Incomplete(progress) => {
                if let Some(candidate) = progress.incumbent {
                    let schedule = self.reconstruct(candidate.values(), candidate.cost())?;
                    if self
                        .incumbent
                        .as_ref()
                        .is_none_or(|old| schedule.completion < old.completion)
                    {
                        self.incumbent = Some(schedule);
                    }
                }
                let lower_bound = progress.lower_bound;
                Ok(IndependentOutcome::Incomplete {
                    lower_bound,
                    incumbent: self.solution(lower_bound)?,
                    reason: progress.reason,
                    stats: progress.stats,
                })
            }
            // A feasible original schedule was checked before translation. This
            // must be a translation/solver defect, never execution infeasibility.
            Outcome::Infeasible if self.incumbent.is_some() => Err(Error::Reconstruction(
                "translation excluded the validated upper witness".into(),
            )),
            Outcome::Infeasible => Ok(IndependentOutcome::Infeasible),
        }
    }

    fn solution(&self, lower_bound: u64) -> Result<Option<Solution>, Error> {
        let Some(incumbent) = &self.incumbent else {
            return Ok(None);
        };
        if lower_bound > incumbent.completion {
            return Err(Error::Reconstruction(
                "lower bound exceeds original-model witness".into(),
            ));
        }
        Ok(Some(Solution {
            model: self.model.clone(),
            schedule: incumbent.clone(),
            lower_bound,
            search_work: self.search.stats().work,
        }))
    }

    fn reconstruct(&self, values: &[i64], objective: u64) -> Result<Schedule, Error> {
        let read = |variable: VarId| -> Result<u64, Error> {
            let value = values
                .get(variable.0)
                .ok_or_else(|| Error::Reconstruction("assignment arity mismatch".into()))?;
            u64::try_from(*value).map_err(|_| Error::Reconstruction("negative time".into()))
        };
        let schedule = Schedule {
            starts: self
                .starts
                .iter()
                .copied()
                .map(read)
                .collect::<Result<_, _>>()?,
            completion: read(self.completion)?,
        };
        if schedule.completion != objective {
            return Err(Error::Reconstruction(
                "makespan differs from objective".into(),
            ));
        }
        self.model
            .check_schedule(&schedule)
            .map_err(Error::Reconstruction)?;
        Ok(schedule)
    }
}

fn tick(value: u64) -> Result<i64, Error> {
    i64::try_from(value)
        .map_err(|_| Error::Unsupported("time exceeds the solver's exact i64 domain".into()))
}

fn equal_offset(builder: &mut ModelBuilder, output: VarId, input: VarId, offset: u64) {
    builder.constraint(Constraint::LinearLe {
        terms: vec![LinearTerm::new(output, 1), LinearTerm::new(input, -1)],
        rhs: offset as i128,
    });
    builder.constraint(Constraint::LinearLe {
        terms: vec![LinearTerm::new(input, 1), LinearTerm::new(output, -1)],
        rhs: -(offset as i128),
    });
}

fn precedence(builder: &mut ModelBuilder, before: VarId, after: VarId) {
    builder.constraint(Constraint::Schedule(SchedulingConstraint::Precedence {
        before: Event::mandatory(before),
        after: Event::mandatory(after),
        lag: 0,
    }));
}

/// Original scheduling semantics embedded in a shared family resource boundary.
/// This owns reconstruction correspondence, but adds no isolated objective or
/// search. The enclosing family owns coverage, model identity and qualification.
pub struct Fragment {
    model: Arc<Model>,
    starts: Vec<VarId>,
    ends: Vec<VarId>,
    presence: Option<VarId>,
}
impl Fragment {
    /// `presence` must represent the complete activation condition of this
    /// fragment, including enclosing topology choices: resources are collected
    /// now and their joint constraints are emitted after lexical guards close.
    /// All fragments must use the enclosing family's workload and timebase.
    pub fn append(
        builder: &mut ModelBuilder,
        encoding: &mut super::symbolic::Encoding,
        model: Arc<Model>,
        presence: Option<VarId>,
    ) -> Result<Self, Error> {
        model.require_complete().map_err(Error::Unsupported)?;
        model.validate().map_err(Error::Model)?;
        if model.resources != encoding.resources() {
            return Err(Error::Model(
                "fragment resource identities differ from the shared boundary".into(),
            ));
        }
        encoding.bind_timebase(&model.timebase)?;
        let mut build =
            |builder: &mut ModelBuilder| append_operations(builder, encoding, &model, presence);
        let (starts, ends) = match presence {
            Some(presence) => builder.when(Literal::new(presence, 1), build),
            None => build(builder),
        }?;
        Ok(Self {
            model,
            starts,
            ends,
            presence,
        })
    }
    pub fn starts(&self) -> &[VarId] {
        &self.starts
    }
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
    /// Original event identity in the enclosing resource scope. Inter-fragment
    /// dependencies and crossing storage lifetimes must use these same events.
    pub fn event(&self, event: super::Event) -> Result<Event, Error> {
        let times = match event.point {
            Point::Start => &self.starts,
            Point::Completion => &self.ends,
        };
        Ok(Event {
            time: *times.get(event.operation).ok_or_else(|| {
                Error::Model("fragment event references an absent operation".into())
            })?,
            presence: self.presence,
        })
    }
    /// Call after the enclosing model has validated its complete assignment.
    /// Rechecks the selected fragment against its original dependencies,
    /// reservations, lifetimes and shared static order. Inactive fragments do
    /// not denote an execution or contribute to the family objective.
    pub fn reconstruct(&self, values: &[i64]) -> Result<Option<Schedule>, Error> {
        let read = |id: VarId| -> Result<u64, Error> {
            values
                .get(id.0)
                .copied()
                .and_then(|v| u64::try_from(v).ok())
                .ok_or_else(|| {
                    Error::Reconstruction("missing or negative fragment assignment".into())
                })
        };
        if let Some(presence) = self.presence {
            match read(presence)? {
                0 => return Ok(None),
                1 => {}
                _ => {
                    return Err(Error::Reconstruction(
                        "fragment presence is not boolean".into(),
                    ));
                }
            }
        }
        let starts = self
            .starts
            .iter()
            .copied()
            .map(read)
            .collect::<Result<Vec<_>, _>>()?;
        let completion =
            starts
                .iter()
                .zip(&self.model.operations)
                .try_fold(0, |end, (start, op)| {
                    start
                        .checked_add(op.latency)
                        .map(|next| end.max(next))
                        .ok_or_else(|| Error::Reconstruction("fragment completion overflow".into()))
                })?;
        let schedule = Schedule { starts, completion };
        self.model
            .check_schedule(&schedule)
            .map_err(Error::Reconstruction)?;
        Ok(Some(schedule))
    }
}

fn translate(
    model: &Model,
    horizon: i64,
) -> Result<(magnitude_solver::Model, Vec<VarId>, VarId), Error> {
    let mut builder = ModelBuilder::new();
    let mut encoding =
        super::symbolic::Encoding::new(&mut builder, "schedule", &model.resources, horizon)?;
    let (starts, _) = append_operations(&mut builder, &mut encoding, model, None)?;
    let completion = encoding.finish(&mut builder)?;
    builder.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    Ok((builder.build()?, starts, completion))
}

fn append_operations(
    builder: &mut ModelBuilder,
    encoding: &mut super::symbolic::Encoding,
    model: &Model,
    presence: Option<VarId>,
) -> Result<(Vec<VarId>, Vec<VarId>), Error> {
    let horizon = encoding.horizon();
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    for (index, operation) in model.operations.iter().enumerate() {
        let latency = tick(operation.latency)?;
        let start = builder.local_variable(
            format!("operation_{index}_start"),
            Domain::interval(0, (horizon - latency).max(0))?,
        )?;
        let duration = builder.local_variable(
            format!("operation_{index}_duration"),
            Domain::singleton(latency),
        )?;
        let end = builder.local_variable(
            format!("operation_{index}_end"),
            // An alternative longer than the enclosing improvement horizon is
            // infeasible only when active. Keep its private domain nonempty so
            // the Activity equation, under presence, establishes that fact.
            Domain::interval(latency.min(horizon), horizon)?,
        )?;
        let activity = Activity {
            start,
            duration,
            end,
            presence,
        };
        encoding.activity(builder, activity.clone());
        for reservation in &operation.reservations {
            if reservation.offset == 0 && reservation.duration == operation.latency {
                encoding.whole_activity(
                    reservation.resource,
                    activity.clone(),
                    Demand::Constant(reservation.units),
                )?;
            }
        }
        starts.push(start);
        ends.push(end);
        for (reservation_index, reservation) in operation.reservations.iter().enumerate() {
            if reservation.offset == 0 && reservation.duration == operation.latency {
                continue;
            }
            let begin = if reservation.offset == 0 {
                start
            } else {
                let begin = builder.local_variable(
                    format!("reservation_{index}_{reservation_index}_start"),
                    Domain::interval(0, horizon)?,
                )?;
                equal_offset(builder, begin, start, reservation.offset);
                begin
            };
            let reservation_end = if reservation.offset + reservation.duration == operation.latency
            {
                end
            } else if reservation.duration == 0 {
                begin
            } else {
                let end = builder.local_variable(
                    format!("reservation_{index}_{reservation_index}_end"),
                    Domain::interval(0, horizon)?,
                )?;
                equal_offset(builder, end, begin, reservation.duration);
                end
            };
            // A partial reservation is itself an activity with its own
            // duration. Exposing that duration lets the solver derive energy
            // and incompatibility bounds for issue service as well as residency.
            let duration = builder.local_variable(
                format!("reservation_{index}_{reservation_index}_duration"),
                Domain::singleton(tick(reservation.duration)?),
            )?;
            encoding.whole_activity(
                reservation.resource,
                Activity {
                    start: begin,
                    duration,
                    end: reservation_end,
                    presence,
                },
                Demand::Constant(reservation.units),
            )?;
        }
    }
    for (index, operation) in model.operations.iter().enumerate() {
        for &before in &operation.predecessors {
            precedence(builder, ends[before], starts[index]);
        }
        for &before in &operation.start_predecessors {
            precedence(builder, starts[before], starts[index]);
        }
    }
    let event = |event: super::Event| Event {
        time: match event.point {
            Point::Start => starts[event.operation],
            Point::Completion => ends[event.operation],
        },
        presence,
    };
    for lifetime in &model.lifetimes {
        encoding.reservation(
            lifetime.resource,
            Lifetime {
                begin: event(lifetime.begin),
                end: event(lifetime.end),
                demand: Demand::Constant(lifetime.units),
            }
            .reservation(),
        )?;
    }
    add_static_orders(builder, model, &starts)?;
    Ok((starts, ends))
}

fn add_static_orders(
    builder: &mut ModelBuilder,
    model: &Model,
    starts: &[VarId],
) -> Result<(), Error> {
    for (block, constraint) in model.static_orders.iter().enumerate() {
        let count = constraint.instructions.len();
        let maximum = i64::try_from(count - 1)
            .map_err(|_| Error::Unsupported("static instruction count exceeds i64".into()))?;
        let positions: Vec<_> = (0..count)
            .map(|index| {
                builder.local_variable(
                    format!("block_{block}_instruction_{index}_position"),
                    Domain::interval(0, maximum).expect("validated position range"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut successors = vec![Vec::new(); count];
        for &(before, after) in &constraint.predecessors {
            successors[before].push(after);
        }
        let mut reaches = vec![vec![false; count]; count];
        for origin in 0..count {
            let mut pending = successors[origin].clone();
            while let Some(after) = pending.pop() {
                if !reaches[origin][after] {
                    reaches[origin][after] = true;
                    pending.extend(successors[after].iter().copied());
                }
            }
        }
        for &(before, after) in &constraint.predecessors {
            builder.constraint(Constraint::LinearLe {
                terms: vec![
                    LinearTerm::new(positions[before], 1),
                    LinearTerm::new(positions[after], -1),
                ],
                rhs: -1,
            });
        }
        // Mandatory static edges imply the same order in every visit. Roots
        // are nonempty by model validation, so transitive edges need not be
        // expanded into pair decisions or repeated dynamic inequalities.
        for &(before, after) in &constraint.predecessors {
            for visit in &constraint.visits {
                for &first in &visit.roots[before] {
                    for &second in &visit.roots[after] {
                        precedence(builder, starts[first], starts[second]);
                    }
                }
            }
        }
        for left in 0..count {
            for right in left + 1..count {
                if reaches[left][right] || reaches[right][left] {
                    continue;
                }
                // Only incomparable instructions introduce a decision. Each
                // guarded direction imposes strict rank order, already implying
                // disequality. Together with mandatory edges these form the full
                // permutation, shared by every visit including equal-time ties.
                let direction = builder.local_variable(
                    format!("block_{block}_pair_{left}_{right}"),
                    Domain::boolean(),
                )?;
                for (value, before, after) in [(1, left, right), (0, right, left)] {
                    let guards = vec![Literal::new(direction, value)];
                    builder.guarded_constraint(
                        guards.clone(),
                        Constraint::LinearLe {
                            terms: vec![
                                LinearTerm::new(positions[before], 1),
                                LinearTerm::new(positions[after], -1),
                            ],
                            rhs: -1,
                        },
                    );
                    for visit in &constraint.visits {
                        for &first in &visit.roots[before] {
                            for &second in &visit.roots[after] {
                                builder.guarded_constraint(
                                    guards.clone(),
                                    Constraint::Schedule(SchedulingConstraint::Precedence {
                                        before: Event::mandatory(starts[first]),
                                        after: Event::mandatory(starts[second]),
                                        lag: 0,
                                    }),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
