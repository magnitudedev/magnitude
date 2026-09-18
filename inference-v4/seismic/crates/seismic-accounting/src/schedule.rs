//! Exact scheduling under an explicitly bound discrete-time resource model.
//! This proves a modeled schedule optimum, not fidelity of an unqualified hardware
//! model. Native mappings must supply dependencies, latency and every reservation;
//! missing mappings cannot be represented by an empty reservation list.
use std::collections::BTreeSet;
pub mod static_order;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CapacityUnit {
    Slots,
    Bytes,
    /// Renewable service capacity per model tick, not resident storage.
    ServicePerTick(crate::resource::Unit),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Timebase {
    pub seconds_numerator: u64,
    pub seconds_denominator: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
/// Result of this analysis's own bounded search. Private construction ties the
/// bounds and feasible schedule to the exact constraints used to derive them.
/// There is no separately supplied proof marker or certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solution {
    model: Model,
    schedule: Schedule,
    lower_bound: u64,
    assignments_examined: u64,
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
    pub fn assignments_examined(&self) -> u64 {
        self.assignments_examined
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
            for reservation in &operation.reservations {
                if reservation.resource >= self.resources.len()
                    || reservation.units == 0
                    || reservation.duration == 0
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
        let mut work = vec![0u128; self.resources.len()];
        for op in &self.operations {
            for r in &op.reservations {
                work[r.resource] = work[r.resource]
                    .checked_add(u128::from(r.units) * u128::from(r.duration))
                    .ok_or("resource work overflow")?;
            }
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
                work[lifetime.resource] = work[lifetime.resource]
                    .checked_add(u128::from(duration) * u128::from(lifetime.units))
                    .ok_or("lifetime occupancy overflow")?;
            }
        }
        for (work, resource) in work.into_iter().zip(&self.resources) {
            let floor = work.div_ceil(u128::from(resource.capacity));
            result = result.max(
                floor
                    .try_into()
                    .map_err(|_| "resource lower bound overflow")?,
            );
        }
        Ok(result)
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

    /// The budget counts proposed start-time assignments, including infeasible
    /// proposals. No score, candidate cap, or floating-point comparison is hidden.
    pub fn solve(&self, assignment_budget: u64) -> Result<Solution, String> {
        match self.search(assignment_budget)? {
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

    pub fn search(&self, assignment_budget: u64) -> Result<SearchOutcome, String> {
        let order = self.validate()?;
        let lower_bound = self.floor(&order)?;
        if !self.unmapped.is_empty() {
            return Ok(SearchOutcome::Incomplete { lower_bound });
        }
        let mut serial = Schedule {
            starts: vec![0; self.operations.len()],
            completion: 0,
        };
        for &i in &order {
            serial.starts[i] = serial.completion;
            serial.completion = serial
                .completion
                .checked_add(self.operations[i].latency)
                .ok_or("serial schedule overflow")?;
        }
        // A serial schedule is only an initial witness, not an admission rule.
        // Remove every idle interval in which no positive-latency operation is
        // active. No operation straddles a removed gap, so its reservations are
        // unchanged; event order is preserved and resident lifetimes contract.
        // The resulting integer schedule completes within the total operation
        // latency. This also covers zero-latency events at the final endpoint.
        let horizon = serial.completion;
        let best = self.check_schedule(&serial).is_ok().then_some(serial);
        let mut search = Search {
            model: self,
            order: &order,
            starts: vec![None; self.operations.len()],
            best,
            horizon,
            floor: lower_bound,
            examined: 0,
            budget: assignment_budget,
            interrupted: false,
        };
        if search
            .best
            .as_ref()
            .is_none_or(|best| best.completion > lower_bound)
        {
            search.run()?;
        }
        let Some(best) = search.best else {
            return Ok(if search.interrupted {
                SearchOutcome::Incomplete { lower_bound }
            } else {
                SearchOutcome::Infeasible
            });
        };
        self.check_schedule(&best)?;
        let lower_bound = if !search.interrupted || best.completion == lower_bound {
            best.completion
        } else {
            lower_bound
        };
        Ok(SearchOutcome::Feasible(Solution {
            model: self.clone(),
            schedule: best,
            lower_bound,
            assignments_examined: search.examined,
        }))
    }
}

struct Search<'a> {
    model: &'a Model,
    order: &'a [usize],
    starts: Vec<Option<u64>>,
    best: Option<Schedule>,
    horizon: u64,
    floor: u64,
    examined: u64,
    budget: u64,
    interrupted: bool,
}
impl Search<'_> {
    fn run(&mut self) -> Result<(), String> {
        if self.order.is_empty() {
            return Ok(());
        }
        let mut depth = 0;
        let mut next_start = vec![Some(0u64); self.order.len()];
        loop {
            if self
                .best
                .as_ref()
                .is_some_and(|best| best.completion == self.floor)
            {
                break;
            }
            if depth == self.order.len() {
                let starts: Vec<_> = self.starts.iter().map(|s| s.unwrap()).collect();
                let completion = starts
                    .iter()
                    .zip(&self.model.operations)
                    .map(|(s, o)| s + o.latency)
                    .max()
                    .unwrap_or(0);
                if self
                    .best
                    .as_ref()
                    .is_none_or(|best| completion < best.completion)
                {
                    self.best = Some(Schedule { starts, completion });
                }
                depth -= 1;
                self.starts[self.order[depth]] = None;
                continue;
            }
            let i = self.order[depth];
            let operation = &self.model.operations[i];
            let start = next_start[depth];
            if start
                .and_then(|s| s.checked_add(operation.latency))
                .is_none_or(|end| {
                    self.best
                        .as_ref()
                        .map_or(end > self.horizon, |best| end >= best.completion)
                })
            {
                self.starts[i] = None;
                if depth == 0 {
                    break;
                }
                depth -= 1;
                self.starts[self.order[depth]] = None;
                continue;
            }
            let start = start.expect("candidate start lies within the checked horizon");
            if self.examined == self.budget {
                self.interrupted = true;
                break;
            }
            self.examined += 1;
            next_start[depth] = start.checked_add(1);
            self.starts[i] = Some(start);
            if self.model.fits(&self.starts)? {
                depth += 1;
                if depth < self.order.len() {
                    let op = &self.model.operations[self.order[depth]];
                    next_start[depth] = Some(
                        op.predecessors
                            .iter()
                            .map(|&p| self.starts[p].unwrap() + self.model.operations[p].latency)
                            .chain(
                                op.start_predecessors
                                    .iter()
                                    .map(|&p| self.starts[p].unwrap()),
                            )
                            .max()
                            .unwrap_or(0),
                    );
                }
            } else {
                self.starts[i] = None;
            }
        }
        Ok(())
    }
}
