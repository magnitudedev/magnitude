//! Exact scheduling under an explicitly bound discrete-time resource model.
//! This proves a modeled schedule optimum, not fidelity of an unqualified hardware
//! model. Native mappings must supply dependencies, latency and every reservation;
//! missing mappings cannot be represented by an empty reservation list.
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CapacityUnit {
    Slots,
    Bytes,
    /// Renewable service capacity per model tick, not resident storage.
    ServicePerTick(crate::resource::Unit),
}

#[derive(Clone, Debug)]
pub struct Timebase {
    pub seconds_numerator: u64,
    pub seconds_denominator: u64,
}

#[derive(Clone, Debug)]
pub struct Resource {
    pub name: String,
    pub capacity: u64,
    pub unit: CapacityUnit,
}
#[derive(Clone, Debug)]
pub struct Reservation {
    pub resource: usize,
    /// Offset from issue, duration, and simultaneous occupied units. A pipeline
    /// may reserve an issue port for one tick while the result latency is longer.
    pub offset: u64,
    pub duration: u64,
    pub units: u64,
}
#[derive(Clone, Debug)]
pub struct Operation {
    pub name: String,
    pub predecessors: Vec<usize>,
    pub latency: u64,
    pub reservations: Vec<Reservation>,
}
#[derive(Clone, Debug)]
pub struct Model {
    /// Identifies the realization, workload, hardware profile and tick definition.
    pub identity: String,
    pub timebase: Timebase,
    pub resources: Vec<Resource>,
    pub operations: Vec<Operation>,
    pub unmapped: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    pub starts: Vec<u64>,
    pub completion: u64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Proof {
    /// A feasible witness attains the dependency/resource lower bound.
    LowerBound,
    /// Every strictly better integer schedule has been excluded. Verification
    /// independently enumerates that finite domain; this marker alone is no proof.
    Exhaustive,
}
#[derive(Clone, Debug)]
pub struct Solution {
    pub model_identity: String,
    pub schedule: Schedule,
    pub lower_bound: u64,
    pub assignments_examined: u64,
    pub proof: Option<Proof>,
}

impl Model {
    fn validate(&self) -> Result<Vec<usize>, String> {
        if self.identity.is_empty() {
            return Err("schedule model needs an identity".into());
        }
        if self.timebase.seconds_numerator == 0 || self.timebase.seconds_denominator == 0 {
            return Err("model tick duration must be a positive rational number of seconds".into());
        }
        if !self.unmapped.is_empty() {
            return Err(format!(
                "cannot optimize incomplete execution model: {:?}",
                self.unmapped
            ));
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
            let mut seen = BTreeSet::new();
            for &predecessor in &operation.predecessors {
                if predecessor >= self.operations.len()
                    || predecessor == i
                    || !seen.insert(predecessor)
                {
                    return Err(format!("invalid dependency of {}", operation.name));
                }
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
        Ok(order)
    }

    fn floor(&self, order: &[usize]) -> Result<u64, String> {
        let mut ends = vec![0u64; self.operations.len()];
        let mut result = 0;
        for &i in order {
            let start = self.operations[i]
                .predecessors
                .iter()
                .map(|&p| ends[p])
                .max()
                .unwrap_or(0);
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

    /// Verify the entire witness independently of the search order.
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
        }
        if completion != schedule.completion {
            return Err("incorrect schedule completion".into());
        }
        let assignments: Vec<_> = schedule.starts.iter().copied().map(Some).collect();
        if !self.fits(&assignments)? {
            return Err("schedule exceeds resource capacity".into());
        }
        Ok(())
    }

    fn fits(&self, starts: &[Option<u64>]) -> Result<bool, String> {
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
        Ok(true)
    }

    /// The budget counts proposed start-time assignments, including infeasible
    /// proposals. No score, candidate cap, or floating-point comparison is hidden.
    pub fn solve(&self, assignment_budget: u64) -> Result<Solution, String> {
        let order = self.validate()?;
        let lower_bound = self.floor(&order)?;
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
        self.check_schedule(&serial)?;
        let mut search = Search {
            model: self,
            order: &order,
            starts: vec![None; self.operations.len()],
            best: serial,
            floor: lower_bound,
            examined: 0,
            budget: assignment_budget,
            interrupted: false,
        };
        if search.best.completion > lower_bound {
            search.run()?;
        }
        let proof = if search.best.completion == lower_bound {
            Some(Proof::LowerBound)
        } else if !search.interrupted {
            Some(Proof::Exhaustive)
        } else {
            None
        };
        Ok(Solution {
            model_identity: self.identity.clone(),
            schedule: search.best,
            lower_bound,
            assignments_examined: search.examined,
            proof,
        })
    }
}

struct Search<'a> {
    model: &'a Model,
    order: &'a [usize],
    starts: Vec<Option<u64>>,
    best: Schedule,
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
        let mut next_start = vec![0u64; self.order.len()];
        loop {
            if self.best.completion == self.floor {
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
                if completion < self.best.completion {
                    self.best = Schedule { starts, completion };
                }
                depth -= 1;
                self.starts[self.order[depth]] = None;
                continue;
            }
            let i = self.order[depth];
            let operation = &self.model.operations[i];
            let start = next_start[depth];
            if start
                .checked_add(operation.latency)
                .is_none_or(|end| end >= self.best.completion)
            {
                self.starts[i] = None;
                if depth == 0 {
                    break;
                }
                depth -= 1;
                self.starts[self.order[depth]] = None;
                continue;
            }
            if self.examined == self.budget {
                self.interrupted = true;
                break;
            }
            self.examined += 1;
            next_start[depth] = start
                .checked_add(1)
                .ok_or("start-time enumeration overflow")?;
            self.starts[i] = Some(start);
            if self.model.fits(&self.starts)? {
                depth += 1;
                if depth < self.order.len() {
                    let op = &self.model.operations[self.order[depth]];
                    next_start[depth] = op
                        .predecessors
                        .iter()
                        .map(|&p| self.starts[p].unwrap() + self.model.operations[p].latency)
                        .max()
                        .unwrap_or(0);
                }
            } else {
                self.starts[i] = None;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verification {
    Verified { assignments_examined: u64 },
    BudgetExhausted { assignments_examined: u64 },
}

impl Model {
    /// Check the witness and the claimed optimum without running the optimizer.
    /// Exhaustive proofs are checked by a separate Cartesian enumeration of every
    /// strictly better schedule, with no dependency-order or resource pruning.
    pub fn verify_solution(
        &self,
        solution: &Solution,
        budget: u64,
    ) -> Result<Verification, String> {
        let order = self.validate()?;
        if solution.model_identity != self.identity {
            return Err("solution model identity mismatch".into());
        }
        self.check_schedule(&solution.schedule)?;
        let floor = self.floor(&order)?;
        if solution.lower_bound != floor {
            return Err("incorrect lower bound".into());
        }
        match solution.proof {
            None => Err("solution carries no optimality proof".into()),
            Some(Proof::LowerBound) => {
                if solution.schedule.completion != floor {
                    return Err("witness does not attain the lower bound".into());
                }
                Ok(Verification::Verified {
                    assignments_examined: 0,
                })
            }
            Some(Proof::Exhaustive) => {
                let mut verifier = Verifier {
                    model: self,
                    better_than: solution.schedule.completion,
                    starts: vec![0; self.operations.len()],
                    examined: 0,
                    budget,
                    interrupted: false,
                };
                verifier.run()?;
                Ok(if verifier.interrupted {
                    Verification::BudgetExhausted {
                        assignments_examined: verifier.examined,
                    }
                } else {
                    Verification::Verified {
                        assignments_examined: verifier.examined,
                    }
                })
            }
        }
    }
}

struct Verifier<'a> {
    model: &'a Model,
    better_than: u64,
    starts: Vec<u64>,
    examined: u64,
    budget: u64,
    interrupted: bool,
}
impl Verifier<'_> {
    fn run(&mut self) -> Result<(), String> {
        let limits: Vec<_> = self
            .model
            .operations
            .iter()
            .map(|op| self.better_than.saturating_sub(op.latency))
            .collect();
        if limits.contains(&0) || limits.is_empty() {
            return Ok(());
        }
        let mut next = vec![0u64; limits.len()];
        let mut depth = 0;
        loop {
            if depth == limits.len() {
                let completion = self
                    .starts
                    .iter()
                    .zip(&self.model.operations)
                    .map(|(s, o)| s + o.latency)
                    .max()
                    .unwrap_or(0);
                let witness = Schedule {
                    starts: self.starts.clone(),
                    completion,
                };
                if completion < self.better_than && self.model.check_schedule(&witness).is_ok() {
                    return Err(format!(
                        "claimed optimum has a better feasible schedule: {witness:?}"
                    ));
                }
                depth -= 1;
                continue;
            }
            if next[depth] == limits[depth] {
                if depth == 0 {
                    break;
                }
                depth -= 1;
                continue;
            }
            if self.examined == self.budget {
                self.interrupted = true;
                break;
            }
            self.examined += 1;
            self.starts[depth] = next[depth];
            next[depth] += 1;
            depth += 1;
            if depth < limits.len() {
                next[depth] = 0;
            }
        }
        Ok(())
    }
}
