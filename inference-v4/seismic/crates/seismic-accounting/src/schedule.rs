//! Exact scheduling under an explicitly bound discrete-time resource model.
//! This proves a modeled schedule optimum, not fidelity of an unqualified hardware
//! model. Native mappings must supply dependencies, latency and every reservation;
//! missing mappings cannot be represented by an empty reservation list.
use std::collections::BTreeSet;
use std::sync::Arc;
mod demand;
pub use demand::Demand;
pub mod evaluation;
pub mod export;
pub mod independent;
pub mod symbolic;
pub mod static_order;
pub mod structured;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CapacityUnit {
    Slots,
    Bytes,
    /// Renewable service capacity per model tick, not resident storage.
    ServicePerTick(crate::resource::Unit),
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Timebase {
    pub seconds_numerator: u64,
    pub seconds_denominator: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Resource {
    pub name: String,
    pub capacity: u64,
    pub unit: CapacityUnit,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub resource: usize,
    /// Offset from issue, duration, and simultaneous occupied units. A pipeline
    /// may reserve an issue port for one tick while the result latency is longer.
    pub offset: u64,
    pub duration: u64,
    pub units: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Point {
    Start,
    Completion,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub operation: usize,
    pub point: Point,
}

/// Resident capacity is held between actual schedule events, rather than for a
/// guessed duration. This covers block admission, shared arrays and live values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Lifetime {
    pub resource: usize,
    pub units: u64,
    pub begin: Event,
    pub end: Event,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Operation {
    pub name: String,
    /// Result dependencies: this operation starts after each predecessor completes.
    pub predecessors: Vec<usize>,
    /// Issue-order constraints: this operation cannot start before these
    /// predecessors start. Their execution may overlap. Both relations together
    /// form a DAG; repeating an edge across the two relations is permitted.
    pub start_predecessors: Vec<usize>,
    pub latency: u64,
    pub reservations: Vec<Reservation>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    pub relationship: crate::authority::ModelRelationship,
    /// Identifies the realization, workload, hardware profile and tick definition.
    pub identity: String,
    pub timebase: Timebase,
    pub resources: Vec<Resource>,
    pub operations: Vec<Operation>,
    pub lifetimes: Vec<Lifetime>,
    /// A single emitted block order must explain all its dynamic visits.
    pub static_orders: Vec<static_order::Constraint>,
    pub unmapped: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub starts: Vec<u64>,
    pub completion: u64,
}

fn validate_reservations(operation: &Operation, resources: usize) -> Result<(), String> {
    for reservation in &operation.reservations {
        if reservation.resource >= resources || reservation.units == 0 || reservation.duration == 0
        {
            return Err(format!("invalid reservation of {}", operation.name));
        }
        if reservation
            .offset
            .checked_add(reservation.duration)
            .is_none_or(|end| end > operation.latency)
        {
            return Err("reservation extends beyond operation completion".into());
        }
    }
    Ok(())
}
/// Result of this analysis's own bounded search. Private construction ties the
/// bounds and feasible schedule to the exact constraints used to derive them.
/// There is no separately supplied proof marker or certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solution {
    model: Arc<Model>,
    schedule: Schedule,
    lower_bound: u64,
    search_work: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SearchOutcome {
    Feasible(Solution),
    /// Unresolved resource facts or exhausted search cannot establish infeasibility.
    Incomplete {
        lower_bound: u64,
    },
    Infeasible,
}
impl Solution {
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }
    pub fn lower_bound(&self) -> u64 {
        self.lower_bound
    }
    pub fn search_work(&self) -> u64 {
        self.search_work
    }
    pub fn is_optimal(&self) -> bool {
        self.lower_bound == self.schedule.completion
    }
}

impl Model {
    /// Necessary duration in this discrete scheduling model. This is deliberately
    /// not a physical bound: the resource and mapping premises belong to the
    /// supplied model. Validation precedes every use, including pruning.
    pub fn lower_bound(&self) -> Result<u64, String> {
        self.floor(&self.validate()?)
    }

    fn require_complete(&self) -> Result<(), String> {
        if !self.unmapped.is_empty() {
            return Err(format!(
                "cannot optimize incomplete execution model: {:?}",
                self.unmapped
            ));
        }
        Ok(())
    }

    fn validate(&self) -> Result<Vec<usize>, String> {
        if self.identity.is_empty() {
            return Err("schedule model needs an identity".into());
        }
        if self.timebase.seconds_numerator == 0 || self.timebase.seconds_denominator == 0 {
            return Err("model tick duration must be a positive rational number of seconds".into());
        }
        let mut names = BTreeSet::new();
        for resource in &self.resources {
            if resource.name.is_empty() || !names.insert(&resource.name) || resource.capacity == 0 {
                return Err("resources require unique names and positive capacities".into());
            }
        }
        names.clear();
        let mut followers = vec![Vec::new(); self.operations.len()];
        let mut incoming = vec![0; self.operations.len()];
        for (i, operation) in self.operations.iter().enumerate() {
            if operation.name.is_empty() || !names.insert(&operation.name) {
                return Err("operations require unique names".into());
            }
            let mut dependencies = BTreeSet::new();
            for relation in [&operation.predecessors, &operation.start_predecessors] {
                let mut seen = BTreeSet::new();
                for &predecessor in relation {
                    if predecessor >= self.operations.len()
                        || predecessor == i
                        || !seen.insert(predecessor)
                    {
                        return Err(format!("invalid dependency of {}", operation.name));
                    }
                    dependencies.insert(predecessor);
                }
            }
            for predecessor in dependencies {
                followers[predecessor].push(i);
                incoming[i] += 1;
            }
            validate_reservations(operation, self.resources.len())?;
        }
        let mut ready: BTreeSet<_> = incoming
            .iter()
            .enumerate()
            .filter_map(|(i, n)| (*n == 0).then_some(i))
            .collect();
        let mut order = Vec::new();
        while let Some(i) = ready.pop_first() {
            order.push(i);
            for &next in &followers[i] {
                incoming[next] -= 1;
                if incoming[next] == 0 {
                    ready.insert(next);
                }
            }
        }
        if order.len() != self.operations.len() {
            return Err("cyclic operation dependencies".into());
        }
        for lifetime in &self.lifetimes {
            if lifetime.resource >= self.resources.len()
                || lifetime.units == 0
                || lifetime.begin.operation >= self.operations.len()
                || lifetime.end.operation >= self.operations.len()
            {
                return Err("invalid resource lifetime".into());
            }
            if matches!(
                self.resources[lifetime.resource].unit,
                CapacityUnit::ServicePerTick(_)
            ) {
                return Err(
                    "a resident lifetime requires storage or slot capacity, not a service rate"
                        .into(),
                );
            }
        }
        static_order::validate(self)?;
        Ok(order)
    }

    fn floor(&self, order: &[usize]) -> Result<u64, String> {
        let mut starts = vec![0u64; self.operations.len()];
        let mut ends = vec![0u64; self.operations.len()];
        let mut result = 0;
        for &i in order {
            let start = self.operations[i]
                .predecessors
                .iter()
                .map(|&p| ends[p])
                .chain(
                    self.operations[i]
                        .start_predecessors
                        .iter()
                        .map(|&p| starts[p]),
                )
                .max()
                .unwrap_or(0);
            starts[i] = start;
            ends[i] = start
                .checked_add(self.operations[i].latency)
                .ok_or("dependency latency overflow")?;
            result = result.max(ends[i]);
        }
        let mut demand = Demand::new(self.timebase.clone(), self.resources.clone())?;
        for op in &self.operations {
            demand.include(op, 1)?;
        }
        // Along a dependency path, the shortest possible lifetime is a
        // necessary occupancy integral. Ignore paths that provide no floor;
        // their capacity is still enforced on every feasible witness.
        for lifetime in &self.lifetimes {
            let mut distance = vec![None; self.operations.len()];
            distance[lifetime.begin.operation] = Some(0u64);
            for &i in order {
                if i == lifetime.begin.operation {
                    continue;
                }
                let completion_edges = self.operations[i].predecessors.iter().filter_map(|&p| {
                    distance[p].map(|d| {
                        d.checked_add(self.operations[p].latency)
                            .ok_or("lifetime dependency overflow")
                    })
                });
                let start_edges = self.operations[i]
                    .start_predecessors
                    .iter()
                    .filter_map(|&p| distance[p].map(Ok));
                distance[i] = completion_edges
                    .chain(start_edges)
                    .collect::<Result<Vec<_>, &str>>()?
                    .into_iter()
                    .max();
            }
            if let Some(distance) = distance[lifetime.end.operation] {
                let offset = |event: Event| {
                    if event.point == Point::Completion {
                        self.operations[event.operation].latency
                    } else {
                        0
                    }
                };
                let duration = distance
                    .checked_add(offset(lifetime.end))
                    .ok_or("lifetime duration overflow")?
                    .saturating_sub(offset(lifetime.begin));
                demand.add_occupancy(lifetime.resource, lifetime.units, duration, 1)?;
            }
        }
        Ok(result.max(demand.lower_bound()?))
    }

    /// Verify an execution upper witness, including the model's permitted claim.
    pub fn check_execution_upper(&self, schedule: &Schedule) -> Result<(), String> {
        self.require_complete()?;
        self.relationship.require_feasible_upper()?;
        self.check_schedule(schedule)
    }

    /// Verify feasibility inside this model independently of search order. For
    /// an optimistic relaxation this is only feasibility of the relaxation;
    /// use check_execution_upper before treating it as an execution witness.
    pub fn check_schedule(&self, schedule: &Schedule) -> Result<(), String> {
        self.validate()?;
        if schedule.starts.len() != self.operations.len() {
            return Err("schedule length mismatch".into());
        }
        let mut completion = 0;
        for (i, operation) in self.operations.iter().enumerate() {
            let end = schedule.starts[i]
                .checked_add(operation.latency)
                .ok_or("schedule time overflow")?;
            completion = completion.max(end);
            for &p in &operation.predecessors {
                let ready = schedule.starts[p]
                    .checked_add(self.operations[p].latency)
                    .ok_or("dependency time overflow")?;
                if schedule.starts[i] < ready {
                    return Err("schedule violates a dependency".into());
                }
            }
            if operation
                .start_predecessors
                .iter()
                .any(|&p| schedule.starts[i] < schedule.starts[p])
            {
                return Err("schedule violates selected issue order".into());
            }
        }
        if completion != schedule.completion {
            return Err("incorrect schedule completion".into());
        }
        let assignments: Vec<_> = schedule.starts.iter().copied().map(Some).collect();
        if !self.fits(&assignments)? {
            return Err(
                "schedule violates resource capacity, lifetime, or static-order constraints".into(),
            );
        }
        Ok(())
    }

    fn fits(&self, starts: &[Option<u64>]) -> Result<bool, String> {
        let time = |event: Event| -> Result<Option<u64>, String> {
            starts[event.operation]
                .map(|start| {
                    start
                        .checked_add(if event.point == Point::Completion {
                            self.operations[event.operation].latency
                        } else {
                            0
                        })
                        .ok_or_else(|| "lifetime event time overflow".into())
                })
                .transpose()
        };
        for (resource_id, resource) in self.resources.iter().enumerate() {
            let mut events = Vec::new();
            for (operation, start) in self.operations.iter().zip(starts) {
                let Some(start) = start else { continue };
                for r in &operation.reservations {
                    if r.resource != resource_id {
                        continue;
                    }
                    let begin = start
                        .checked_add(r.offset)
                        .ok_or("reservation time overflow")?;
                    let end = begin
                        .checked_add(r.duration)
                        .ok_or("reservation time overflow")?;
                    // Ends sort before starts at an equal timestamp: intervals are half-open.
                    events.push((begin, true, r.units));
                    events.push((end, false, r.units));
                }
            }
            for lifetime in &self.lifetimes {
                if lifetime.resource != resource_id {
                    continue;
                }
                let (Some(begin), Some(end)) = (time(lifetime.begin)?, time(lifetime.end)?) else {
                    // Unassigned endpoints can only add restrictions later.
                    continue;
                };
                if end < begin {
                    return Ok(false);
                }
                if end != begin {
                    events.push((begin, true, lifetime.units));
                    events.push((end, false, lifetime.units));
                }
            }
            events.sort_unstable();
            let mut used = 0u128;
            for (_, start, units) in events {
                if start {
                    used += u128::from(units);
                } else {
                    used -= u128::from(units);
                }
                if used > u128::from(resource.capacity) {
                    return Ok(false);
                }
            }
        }
        static_order::fits(self, starts)
    }

    /// The budget counts independent solver work, including propagation and
    /// branch analysis. An incomplete feasible solution is diagnostic only.
    pub fn solve(&self, work_budget: u64) -> Result<Solution, String> {
        match self.search(work_budget)? {
            SearchOutcome::Feasible(solution) => Ok(solution),
            SearchOutcome::Incomplete { .. } => {
                self.require_complete()?;
                Err("schedule search has not found a feasible witness within its budget".into())
            }
            SearchOutcome::Infeasible => {
                Err("execution constraints admit no feasible schedule".into())
            }
        }
    }

    pub fn search(&self, work_budget: u64) -> Result<SearchOutcome, String> {
        self.start_search()?.advance(work_budget)
    }

    /// Retain exact constraints and the independent solver frontier across
    /// budgeted calls. Neither derivation nor explored assignments are replayed.
    pub fn start_search(&self) -> Result<Search, String> {
        let order = self.validate()?;
        let lower_bound = self.floor(&order)?;
        if !self.unmapped.is_empty() {
            return Ok(Search {
                model: Arc::new(self.clone()),
                state: None,
                obligations: self.unmapped.iter().map(|reason| magnitude_solver::model::Obligation { kind: magnitude_solver::model::ObligationKind::Analysis, reason: reason.clone() }).collect(),
                lower_bound,
            });
        }
        let model = Arc::new(self.clone());
        let (state, obligations) = match independent::IndependentSearch::from_model(
            model.clone(), magnitude_solver::Options::default(),
        ) {
            Ok(state) => (Some(state), Vec::new()),
            Err(independent::Error::Unsupported(reason)) => (None, vec![magnitude_solver::model::Obligation {
                kind: magnitude_solver::model::ObligationKind::Analysis, reason,
            }]),
            Err(error) => return Err(error.to_string()),
        };
        Ok(Search { model, state, obligations, lower_bound })
    }
}

/// An analysis in progress bound privately to its immutable execution constraints.
pub struct Search {
    model: Arc<Model>,
    state: Option<independent::IndependentSearch>,
    obligations: Vec<magnitude_solver::model::Obligation>,
    lower_bound: u64,
}
impl Search {
    pub fn model(&self) -> &Model {
        &self.model
    }
    pub fn obligations(&self) -> &[magnitude_solver::model::Obligation] { &self.obligations }
    pub fn advance(&mut self, region_budget: u64) -> Result<SearchOutcome, String> {
        let Some(state) = self.state.as_mut() else {
            return Ok(SearchOutcome::Incomplete {
                lower_bound: self.lower_bound,
            });
        };
        match state
            .advance(magnitude_solver::Limits {
                work: region_budget,
                time: None,
                memory_bytes: None,
            })
            .map_err(|error| error.to_string())?
        {
            independent::IndependentOutcome::Optimal(solution) => {
                Ok(SearchOutcome::Feasible(solution))
            }
            independent::IndependentOutcome::Infeasible => Ok(SearchOutcome::Infeasible),
            independent::IndependentOutcome::Incomplete {
                incumbent,
                lower_bound,
                reason,
                ..
            } => {
                if let magnitude_solver::result::StopReason::Coverage(obligations) = reason { self.obligations = obligations; }
                self.lower_bound = self.lower_bound.max(lower_bound);
                match incumbent {
                    Some(mut solution) => {
                        solution.lower_bound = solution.lower_bound.max(self.lower_bound);
                        if solution.lower_bound > solution.schedule.completion {
                            return Err(
                                "scheduling lower bound exceeds original-model witness".into()
                            );
                        }
                        Ok(SearchOutcome::Feasible(solution))
                    }
                    None => Ok(SearchOutcome::Incomplete {
                        lower_bound: self.lower_bound,
                    }),
                }
            }
        }
    }
}

/// Joint operation-local conditional activity export.
pub mod guarded;
