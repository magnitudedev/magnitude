//! Repeated execution structure under the same reservation/lifetime semantics
//! as the flat scheduling oracle. Repetition never requires one stored node or
//! start-time variable per instance. A compact witness is not an optimum unless
//! its completion meets the derived lower bound.
use super::*;
mod periodic;
mod profile;
mod overlap;
mod windows;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order { Serial, Parallel }

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Node {
    /// Dependencies are expressed by enclosing structure; leaf edge lists must
    /// be empty. The primitive's latency and reservations are unchanged.
    Operation(Operation),
    Compose { order: Order, children: Vec<Arc<Node>> },
    Repeat { order: Order, count: u64, body: Arc<Node> },
    /// Resident capacity held from entry through completion of the entire body.
    Scope { reservations: Vec<(usize, u64)>, body: Arc<Node> },
}
impl Node {
    fn uses_resource(&self, resource: usize) -> bool {
        match self {
            Self::Operation(operation) => operation.reservations.iter().any(|r| r.resource == resource),
            Self::Compose { children, .. } => children.iter().any(|child| child.uses_resource(resource)),
            Self::Repeat { count, body, .. } => *count > 0 && body.uses_resource(resource),
            Self::Scope { reservations, body } => reservations.iter().any(|(r, _)| *r == resource) || body.uses_resource(resource),
        }
    }
    /// Factor equal parallel bodies after deriving their actual constraints.
    /// This shares accounting structure; it does not merge native invocations.
    pub fn append_parallel(children: &mut Vec<Arc<Node>>, node: Arc<Node>) -> Result<(), String> {
        if let Some(previous) = children.last() {
            let (body, count) = match previous.as_ref() {
                Self::Repeat { order: Order::Parallel, body, count } => (body, *count),
                _ => (previous, 1),
            };
            if **body == *node {
                let repeated = Arc::new(Self::Repeat { order: Order::Parallel,
                    count: count.checked_add(1).ok_or("parallel repetition overflow")?, body: body.clone() });
                *children.last_mut().expect("existing parallel member") = repeated;
                return Ok(());
            }
        }
        children.push(node);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Structured {
    pub relationship: crate::authority::ModelRelationship,
    pub identity: String,
    pub timebase: Timebase,
    pub resources: Vec<Resource>,
    pub root: Arc<Node>,
    pub unmapped: Vec<String>,
}

/// A compact feasible traversal, including bounded-concurrency repeated bodies.
/// Private construction binds the plan to validated immutable constraints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Witness {
    model: Arc<Structured>,
    completion: u64,
    lower_bound: u64,
    plan: Arc<Plan>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Plan {
    Flat(super::Solution),
    Selected(Arc<Witness>),
    Operation { duration: u64 },
    Sequence { children: Vec<Arc<Plan>>, duration: u64 },
    Parallel { children: Vec<Arc<Plan>>, duration: u64 },
    Offset { children: Vec<Arc<Plan>>, starts: Vec<u64>, duration: u64, profile_limit: u64 },
    Repeat { body: Arc<Plan>, count: u64, concurrent: u64, duration: u64 },
    Scope { body: Arc<Plan>, duration: u64 },
    Periodic { body: Arc<Witness>, count: u64, period: u64, duration: u64, profile_limit: u64 },
}
impl Plan {
    fn has_events(&self) -> bool {
        match self {
            Self::Flat(solution) => !solution.schedule().starts.is_empty(),
            Self::Selected(witness) => witness.plan.has_events(),
            Self::Periodic { body, count, .. } => *count > 0 && body.plan.has_events(),
            Self::Operation { .. } | Self::Scope { .. } => true,
            Self::Sequence { children, .. } | Self::Parallel { children, .. } | Self::Offset { children, .. } => children.iter().any(|child| child.has_events()),
            Self::Repeat { body, count, .. } => *count > 0 && body.has_events(),
        }
    }
    fn duration(&self) -> u64 { match self { Self::Flat(solution) => solution.schedule().completion, Self::Selected(witness) => witness.completion, Self::Operation { duration } | Self::Sequence { duration, .. } | Self::Parallel { duration, .. } | Self::Offset { duration, .. } | Self::Repeat { duration, .. } | Self::Scope { duration, .. } | Self::Periodic { duration, .. } => *duration } }
    fn starts(&self, origin: u64, output: &mut Vec<u64>) {
        match self {
            Self::Flat(solution) => output.extend(solution.schedule().starts.iter().map(|start| origin + start)),
            Self::Selected(witness) => witness.plan.starts(origin, output),
            Self::Operation { .. } => output.push(origin),
            Self::Sequence { children, .. } => {
                let mut at = origin;
                for child in children { child.starts(at, output); at += child.duration(); }
            }
            Self::Parallel { children, .. } => {
                for child in children { child.starts(origin, output); }
            }
            Self::Offset { children, starts, .. } => {
                for (child, offset) in children.iter().zip(starts) { child.starts(origin + offset, output); }
            }
            Self::Repeat { body, count, concurrent, .. } => {
                if !body.has_events() { return; }
                for index in 0..*count { body.starts(origin + index / concurrent * body.duration(), output); }
            }
            Self::Periodic { body, count, period, .. } => {
                if body.plan.has_events() { for index in 0..*count { body.plan.starts(origin + index * period, output); } }
            }
            Self::Scope { body, duration } => { output.push(origin); body.starts(origin, output); output.push(origin + duration); }
        }
    }
}
impl Witness {
    /// Reconstruct a joint shared-model assignment in original occurrence order.
    /// No child optimum is inferred: the witness retains only an independently
    /// derived lower bound and the checked feasible completion.
    pub fn from_schedule(model: Arc<Structured>, schedule: Schedule) -> Result<Self, String> {
        let flat = Arc::new(model.expand(schedule.starts.len() as u64).map_err(|e| e.to_string())?);
        flat.check_execution_upper(&schedule)?;
        let lower_bound = flat.lower_bound()?;
        model.witness(Arc::new(Plan::Flat(super::Solution {
            model: flat, schedule, lower_bound, search_work: 0,
        })))
    }
    fn peak(&self) -> Result<Vec<u64>, String> {
        let summary = self.model.summary()?;
        if summary.plan == self.plan { return Ok(summary.peak); }
        let mut peak = vec![0u64; self.model.resources.len()];
        match self.plan.as_ref() {
            Plan::Flat(solution) => {
                let model = solution.model();
                let schedule = solution.schedule();
                let event_time = |event: Event| schedule.starts[event.operation] +
                    if event.point == Point::Completion { model.operations[event.operation].latency } else { 0 };
                for resource in 0..peak.len() {
                    let mut events = Vec::new();
                    for (operation, start) in model.operations.iter().zip(&schedule.starts) {
                        for r in operation.reservations.iter().filter(|r| r.resource == resource) {
                            events.push((start + r.offset, true, r.units));
                            events.push((start + r.offset + r.duration, false, r.units));
                        }
                    }
                    for r in model.lifetimes.iter().filter(|r| r.resource == resource) {
                        let (begin, end) = (event_time(r.begin), event_time(r.end));
                        if begin != end { events.push((begin, true, r.units)); events.push((end, false, r.units)); }
                    }
                    events.sort_unstable();
                    let mut used = 0u64;
                    for (_, start, units) in events {
                        used = if start { used.checked_add(units).ok_or("refined peak overflow")? }
                            else { used.checked_sub(units).ok_or("invalid refined interval")? };
                        peak[resource] = peak[resource].max(used);
                    }
                }
            }
            Plan::Sequence { children, .. } => {
                for child in children {
                    let Plan::Selected(child) = child.as_ref() else { return Err("refined sequence child missing".into()); };
                    for (peak, next) in peak.iter_mut().zip(child.peak()?) { *peak = (*peak).max(next); }
                }
            }
            Plan::Periodic { body, period, profile_limit, .. } => { peak = body.periodic_peak(*period, *profile_limit)?.ok_or("periodic witness exceeds capacity")?; }
            Plan::Parallel { children, .. } => {
                peak = self.model.parallel_peak(children)?;
            }
            Plan::Offset { children, starts, duration, profile_limit } => {
                peak = overlap::checked_peak(&self.model, children, starts, *duration, *profile_limit)?;
            }
            Plan::Repeat { body, concurrent, count, .. } => {
                let Plan::Selected(body) = body.as_ref() else { return Err("refined repeat body missing".into()); };
                for (peak, next) in peak.iter_mut().zip(body.peak()?) { *peak = next.checked_mul((*concurrent).min(*count)).ok_or("refined peak overflow")?; }
            }
            Plan::Scope { body, duration } => {
                let Plan::Selected(body) = body.as_ref() else { return Err("refined scope body missing".into()); };
                peak = body.peak()?;
                let Node::Scope { reservations, .. } = self.model.root.as_ref() else { return Err("refined scope model missing".into()); };
                if *duration > 0 {
                    for &(resource, units) in reservations { peak[resource] = peak[resource].checked_add(units).ok_or("refined resident peak overflow")?; }
                }
            }
            _ => return Err("unsupported refined peak plan".into()),
        }
        Ok(peak)
    }
    pub(crate) fn check_execution_upper(&self) -> Result<(), String> {
        self.model.relationship.require_feasible_upper()?;
        if !self.model.unmapped.is_empty() { return Err("structured witness has missing mappings".into()); }
        let (lower, completion) = self.model.plan_bounds(&self.plan)?;
        if (lower, completion) != (self.lower_bound, self.completion) || lower > completion {
            return Err("structured witness no longer matches its constraints".into());
        }
        Ok(())
    }
    pub fn model(&self) -> &Structured { &self.model }
    pub fn completion(&self) -> u64 { self.completion }
    pub fn lower_bound(&self) -> u64 { self.lower_bound }
    pub fn is_optimal(&self) -> bool { self.completion == self.lower_bound }
    pub fn expand(&self, maximum_operations: u64) -> Result<(Model, Schedule), crate::workload::DerivationError> {
        let model = self.model.expand(maximum_operations)?;
        let mut starts = Vec::with_capacity(model.operations.len());
        self.plan.starts(0, &mut starts);
        let schedule = Schedule { starts, completion: self.completion };
        model.check_execution_upper(&schedule)?;
        Ok((model, schedule))
    }
}

struct Summary {
    demand: Demand,
    windows: Vec<Option<windows::Window>>,
    completion: u64,
    feasible: bool,
    expanded_nodes: u64,
    peak: Vec<u64>,
    plan: Arc<Plan>,
}
impl Structured {
    fn with_root(&self, root: Arc<Node>) -> Self { Self { root, ..self.clone() } }
    /// A whole-body resident reservation larger than half the available pool
    /// forces every pair of copies to be disjoint. Because copies are identical
    /// and have no cross-occurrence edges, their ordering is immaterial: the
    /// complete optimum is the sum of private body optima. Instruction demand
    /// or the peak of a particular witness cannot establish this property.
    fn resident_copies_are_serial(&self, body: &Node) -> Result<bool, String> {
        let mut held = vec![0u64; self.resources.len()];
        let mut node = body;
        while let Node::Scope { reservations, body } = node {
            for &(resource, units) in reservations {
                let amount = held.get_mut(resource).ok_or("invalid resident resource")?;
                *amount = amount.checked_add(units).ok_or("resident capacity overflow")?;
            }
            node = body;
        }
        Ok(self.resources.iter().zip(held).any(|(resource, units)| {
            units <= resource.capacity && units > resource.capacity / 2
        }))
    }
    fn parallel_concurrency(&self, witness: &Witness, count: u64) -> Result<u64, String> {
        let mut concurrent = count.max(1);
        for (resource, peak) in self.resources.iter().zip(witness.peak()?) {
            if peak > 0 { concurrent = concurrent.min(resource.capacity / peak); }
        }
        if concurrent == 0 { return Err("refined body exceeds available capacity".into()); }
        Ok(concurrent)
    }
    fn parallel_lower(&self, body: &Node, count: u64, lower: u64) -> Result<u64, String> {
        self.resident_repetition_lower(body, count, lower, &vec![0; self.resources.len()])
    }
    fn resident_repetition_lower(&self, body: &Node, count: u64, lower: u64, ancestor: &[u64]) -> Result<u64, String> {
        if count == 0 { return Ok(0); }
        // Whole-body resident intervals admit at most K simultaneous copies.
        // Interval coloring partitions N copies into K serial chains, one of
        // which contains at least ceil(N/K) bodies of at least `lower` duration.
        let mut held = vec![0u64; self.resources.len()];
        let mut node = body;
        while let Node::Scope { reservations, body } = node {
            for &(resource, units) in reservations { held[resource] = held[resource].checked_add(units).ok_or("refined residency overflow")?; }
            node = body;
        }
        let mut bound = lower;
        if let Node::Operation(operation) = node {
            // Each identical reservation colors operation starts into at most
            // K lanes separated by its duration. The final result still needs
            // the full operation latency, including its unreserved tail.
            for reservation in &operation.reservations {
                if reservation.duration == 0 || reservation.units == 0 { continue; }
                let capacity = self.resources[reservation.resource].capacity
                    .saturating_sub(ancestor[reservation.resource]).saturating_sub(held[reservation.resource]);
                let copies = capacity / reservation.units;
                if copies > 0 {
                    let start = (count.div_ceil(copies) - 1).checked_mul(reservation.duration).ok_or("pipeline lower bound overflow")?;
                    bound = bound.max(start.checked_add(operation.latency).ok_or("pipeline tail overflow")?);
                }
            }
        }
        for ((resource, ancestor), units) in self.resources.iter().zip(ancestor).zip(held) {
            if units == 0 { continue; }
            let copies = resource.capacity.saturating_sub(*ancestor) / units;
            if copies > 0 { bound = bound.max(count.div_ceil(copies).checked_mul(lower).ok_or("refined resident bound overflow")?); }
        }
        Ok(bound)
    }
    fn parallel_plan(&self, witness: Witness) -> Result<Arc<Plan>, String> {
        let Node::Repeat { order: Order::Parallel, count, .. } = self.root.as_ref() else { return Err("parallel refinement requires repetition".into()); };
        let concurrent = self.parallel_concurrency(&witness, *count)?;
        let duration = witness.completion.checked_mul(count.div_ceil(concurrent)).ok_or("refined parallel completion overflow")?;
        Ok(Arc::new(Plan::Repeat { body: Arc::new(Plan::Selected(Arc::new(witness))), count: *count, concurrent, duration }))
    }
    /// During the body's entire execution, enclosing residency is unavailable
    /// to its instructions and nested scopes. Derive that residual capacity,
    /// retaining the same reservations, ordering and physical resource names.
    fn scope_body(&self) -> Result<Option<Self>, String> {
        let Node::Scope { reservations, body } = self.root.as_ref() else { return Ok(None); };
        let mut held = vec![0u64; self.resources.len()];
        for &(resource, units) in reservations {
            let amount = held.get_mut(resource).ok_or("invalid scoped resource")?;
            *amount = amount.checked_add(units).ok_or("scoped capacity overflow")?;
        }
        let mut child = self.with_root(body.clone());
        for (resource, amount) in held.into_iter().enumerate() {
            let Some(remaining) = self.resources[resource].capacity.checked_sub(amount) else { return Ok(None); };
            if remaining == 0 {
                // Flat models require positive capacities. An unused resource
                // has no constraints to reduce; used zero-capacity resources
                // keep the original scope and its exact feasibility analysis.
                if body.uses_resource(resource) { return Ok(None); }
            } else { child.resources[resource].capacity = remaining; }
        }
        Ok(Some(child))
    }
    /// Conservative simultaneous resource envelope of already checked children.
    /// It may reject a feasible staggered overlap, but cannot admit contention.
    fn parallel_peak(&self, plans: &[Arc<Plan>]) -> Result<Vec<u64>, String> {
        let mut peak = vec![0u64; self.resources.len()];
        for plan in plans {
            let Plan::Selected(child) = plan.as_ref() else { return Err("parallel child witness missing".into()); };
            for (total, next) in peak.iter_mut().zip(child.peak()?) {
                *total = total.checked_add(next).ok_or("parallel refined peak overflow")?;
            }
        }
        Ok(peak)
    }
    fn parallel_fits(&self, plans: &[Arc<Plan>]) -> Result<bool, String> {
        let mut remaining: Vec<_> = self.resources.iter().map(|r| r.capacity).collect();
        for plan in plans {
            let Plan::Selected(child) = plan.as_ref() else { return Err("parallel child witness missing".into()); };
            for (capacity, peak) in remaining.iter_mut().zip(child.peak()?) {
                let Some(residual) = capacity.checked_sub(peak) else { return Ok(false); };
                *capacity = residual;
            }
        }
        Ok(true)
    }
    fn plan_bounds(&self, plan: &Plan) -> Result<(u64, u64), String> {
        let summary = self.summary()?;
        let floor = summary.demand.lower_bound()?;
        if summary.feasible && summary.plan.as_ref() == plan { return Ok((floor, summary.completion)); }
        let child_bounds = |node: &Arc<Node>, plan: &Arc<Plan>| -> Result<(u64, u64), String> {
            let Plan::Selected(witness) = plan.as_ref() else { return Err("refined child lacks its owning model".into()); };
            if *witness.model != self.with_root(node.clone()) { return Err("refined child model differs from its serial region".into()); }
            witness.check_execution_upper()?;
            Ok((witness.lower_bound, witness.completion))
        };
        let (lower, completion) = match (self.root.as_ref(), plan) {
            (_, Plan::Flat(solution)) => {
                let expanded = self.expand(solution.model().operations.len() as u64).map_err(|e| e.to_string())?;
                if &expanded != solution.model() { return Err("refined flat model differs from its structured region".into()); }
                solution.model().check_execution_upper(solution.schedule())?;
                (solution.lower_bound(), solution.schedule().completion)
            }
            (Node::Compose { order, children }, Plan::Sequence { children: plans, duration }) if children.len() == plans.len() => {
                let mut lower = 0u64;
                let mut upper = 0u64;
                for (child, plan) in children.iter().zip(plans) {
                    let (lo, hi) = child_bounds(child, plan)?;
                    lower = if *order == Order::Serial { lower.checked_add(lo).ok_or("serial refined bound overflow")? } else { lower.max(lo) };
                    upper = upper.checked_add(hi).ok_or("serial refined completion overflow")?;
                }
                if upper != *duration { return Err("serial refined duration mismatch".into()); }
                (lower, upper)
            }
            (Node::Compose { order: Order::Parallel, children }, Plan::Parallel { children: plans, duration }) if children.len() == plans.len() => {
                let (mut lower, mut upper) = (0, 0);
                for (child, plan) in children.iter().zip(plans) {
                    let (lo, hi) = child_bounds(child, plan)?;
                    lower = lower.max(lo);
                    upper = upper.max(hi);
                }
                if upper != *duration || !self.parallel_fits(plans)? { return Err("parallel refined capacity or duration mismatch".into()); }
                (lower, upper)
            }
            (Node::Compose { order: Order::Parallel, children }, Plan::Offset { children: plans, starts, duration, profile_limit }) if children.len() == plans.len() => {
                let mut lower = 0;
                for (child, plan) in children.iter().zip(plans) { lower = lower.max(child_bounds(child, plan)?.0); }
                overlap::checked_peak(self, plans, starts, *duration, *profile_limit)?;
                (lower, *duration)
            }
            (Node::Repeat { order: Order::Serial, count, body }, Plan::Repeat { body: plan, count: visits, concurrent: 1, duration }) if count == visits => {
                let (lo, hi) = child_bounds(body, plan)?;
                let upper = hi.checked_mul(*count).ok_or("repeated refined completion overflow")?;
                if upper != *duration { return Err("repeated refined duration mismatch".into()); }
                (lo.checked_mul(*count).ok_or("repeated refined bound overflow")?, upper)
            }
            (Node::Repeat { order: Order::Parallel, count, body }, Plan::Repeat { body: plan, count: visits, concurrent, duration }) if count == visits => {
                let (lower, upper) = child_bounds(body, plan)?;
                let Plan::Selected(witness) = plan.as_ref() else { unreachable!() };
                if *concurrent != self.parallel_concurrency(witness, *count)? { return Err("refined parallel concurrency mismatch".into()); }
                let completion = upper.checked_mul(count.div_ceil(*concurrent)).ok_or("refined parallel completion overflow")?;
                if completion != *duration { return Err("refined parallel duration mismatch".into()); }
                (self.parallel_lower(body, *count, lower)?, completion)
            }
            (Node::Repeat { order: Order::Parallel, count, body: node }, Plan::Periodic { body, count: visits, period, duration, profile_limit }) if count == visits && *count > 0 => {
                if *body.model != self.with_root(node.clone()) { return Err("periodic body model mismatch".into()); }
                body.check_execution_upper()?;
                if body.periodic_peak(*period, *profile_limit)?.is_none() { return Err("periodic capacity exceeded".into()); }
                let completion = period.checked_mul(count - 1).and_then(|t| t.checked_add(body.completion)).ok_or("periodic duration overflow")?;
                if completion != *duration { return Err("periodic duration mismatch".into()); }
                (self.parallel_lower(node, *count, body.lower_bound)?, completion)
            }
            (Node::Scope { .. }, Plan::Scope { body, duration }) => {
                let Plan::Selected(witness) = body.as_ref() else { return Err("refined scope lacks its body model".into()); };
                let expected = self.scope_body()?.ok_or("scoped capacity cannot be decomposed")?;
                if *witness.model != expected { return Err("refined scope has different residual capacities".into()); }
                witness.check_execution_upper()?;
                if witness.completion != *duration { return Err("refined scope duration mismatch".into()); }
                (witness.lower_bound, witness.completion)
            }
            _ => return Err("refined plan does not preserve structured dependencies".into()),
        };
        Ok((floor.max(lower), completion))
    }
    fn witness(&self, plan: Arc<Plan>) -> Result<Witness, String> {
        let (lower_bound, completion) = self.plan_bounds(&plan)?;
        if lower_bound > completion { return Err("refined lower bound exceeds feasible completion".into()); }
        Ok(Witness { model: Arc::new(self.clone()), lower_bound, completion, plan })
    }
    fn empty_model(&self) -> Model {
        Model { relationship: self.relationship.clone(), identity: self.identity.clone(),
            timebase: self.timebase.clone(), resources: self.resources.clone(),
            operations: vec![], lifetimes: vec![], static_orders: vec![], unmapped: self.unmapped.clone() }
    }
    fn summary(&self) -> Result<Summary, String> {
        self.empty_model().validate()?;
        self.summarize(&self.root, &vec![0; self.resources.len()])
    }
    fn summarize(&self, node: &Node, held: &[u64]) -> Result<Summary, String> {
        let mut result = Summary { demand: Demand::new(self.timebase.clone(), self.resources.clone())?,
            windows: vec![None; self.resources.len()], completion: 0, feasible: true, expanded_nodes: 0, peak: vec![0; self.resources.len()],
            plan: Arc::new(Plan::Sequence { children: vec![], duration: 0 }) };
        match node {
            Node::Operation(operation) => {
                if operation.name.is_empty() || !operation.predecessors.is_empty() || !operation.start_predecessors.is_empty() {
                    return Err("structured leaf dependencies must belong to its enclosing graph".into());
                }
                let mut local = self.empty_model();
                local.operations = vec![event("scope begin", vec![]), operation.clone(), event("scope end", vec![1])];
                local.operations[1].name = "structured leaf".into();
                local.operations[1].predecessors = vec![0];
                local.lifetimes = held.iter().enumerate().filter(|(_, units)| **units > 0).map(|(resource, &units)| Lifetime {
                    resource, units, begin: Event { operation: 0, point: Point::Start },
                    end: Event { operation: 2, point: Point::Completion },
                }).collect();
                local.validate()?;
                result.feasible = local.check_schedule(&Schedule { starts: vec![0, 0, operation.latency], completion: operation.latency }).is_ok();
                result.demand.include(operation, 1)?;
                for reservation in &operation.reservations {
                    windows::Window::include(&mut result.windows[reservation.resource],
                        u128::from(reservation.units) * u128::from(reservation.duration), reservation.offset,
                        operation.latency - reservation.offset - reservation.duration)?;
                }
                result.completion = operation.latency;
                result.expanded_nodes = 1;
                result.plan = Arc::new(Plan::Operation { duration: operation.latency });
                for resource in 0..self.resources.len() {
                    let mut events = Vec::new();
                    for reservation in operation.reservations.iter().filter(|r| r.resource == resource) {
                        events.push((reservation.offset, true, reservation.units));
                        events.push((reservation.offset + reservation.duration, false, reservation.units));
                    }
                    events.sort_unstable();
                    let mut used = 0u64;
                    for (_, start, units) in events {
                        used = if start { used.checked_add(units).ok_or("structured peak overflow")? } else { used - units };
                        result.peak[resource] = result.peak[resource].max(used);
                    }
                }
            }
            Node::Compose { order, children } => {
                let mut plans = Vec::new();
                let mut child_windows = Vec::new();
                let mut serial_lower = 0u64;
                for child in children {
                    let next = self.summarize(child, held)?;
                    let lower = next.demand.lower_bound()?;
                    serial_lower = serial_lower.checked_add(lower).ok_or("structured window dependency overflow")?;
                    child_windows.push((next.windows, lower));
                    result.demand.append(&next.demand, *order == Order::Serial)?;
                    result.completion = result.completion.checked_add(next.completion).ok_or("compact witness overflow")?;
                    result.feasible &= next.feasible;
                    result.expanded_nodes = result.expanded_nodes.checked_add(next.expanded_nodes).ok_or("structured node count overflow")?;
                    for (peak, next) in result.peak.iter_mut().zip(&next.peak) { *peak = (*peak).max(*next); }
                    plans.push(next.plan);
                }
                let mut before = 0u64;
                for (windows, lower) in child_windows {
                    let after = serial_lower - before - lower;
                    for (resource, window) in windows.into_iter().enumerate() {
                        if let Some(window) = window {
                            let (head, tail) = if *order == Order::Serial {
                                (window.head.checked_add(before).ok_or("structured window head overflow")?,
                                 window.tail.checked_add(after).ok_or("structured window tail overflow")?)
                            } else { (window.head, window.tail) };
                            windows::Window::include(&mut result.windows[resource], window.work, head, tail)?;
                        }
                    }
                    before += lower;
                }
                result.plan = Arc::new(Plan::Sequence { children: plans, duration: result.completion });
            }
            Node::Repeat { order, count, body } => {
                // Validate the template even when it has no dynamic visits.
                result = self.summarize(body, held)?;
                let child_lower = result.demand.lower_bound()?;
                let child_duration = result.completion;
                let mut concurrent = 1;
                if *order == Order::Parallel && *count > 0 && result.feasible {
                    concurrent = *count;
                    for ((resource, held), peak) in self.resources.iter().zip(held).zip(&result.peak) {
                        if *peak > 0 { concurrent = concurrent.min(resource.capacity.saturating_sub(*held) / peak); }
                    }
                    if concurrent == 0 { result.feasible = false; concurrent = 1; }
                }
                result.demand.repeat(*count, *order == Order::Serial)?;
                for window in &mut result.windows {
                    if *count == 0 { *window = None; }
                    else if let Some(window) = window { window.work = window.work.checked_mul(u128::from(*count)).ok_or("structured repeated window overflow")?; }
                }
                if *order == Order::Parallel && *count > 0 {
                    result.demand.require_duration(self.resident_repetition_lower(body, *count, child_lower, held)?);
                }
                result.completion = child_duration.checked_mul(count.div_ceil(concurrent)).ok_or("repeated witness overflow")?;
                for peak in &mut result.peak { *peak = peak.checked_mul(concurrent.min(*count)).ok_or("structured peak overflow")?; }
                result.expanded_nodes = result.expanded_nodes.checked_mul(*count).ok_or("structured node count overflow")?;
                if *count == 0 { result.feasible = true; }
                result.plan = Arc::new(Plan::Repeat { body: result.plan, count: *count, concurrent, duration: result.completion });
            }
            Node::Scope { reservations, body } => {
                let mut nested = held.to_vec();
                for &(resource, units) in reservations {
                    if resource >= self.resources.len() || units == 0 || matches!(self.resources[resource].unit, CapacityUnit::ServicePerTick(_)) {
                        return Err("invalid structured resident lifetime".into());
                    }
                    nested[resource] = nested[resource].checked_add(units).ok_or("resident capacity overflow")?;
                }
                result = self.summarize(body, &nested)?;
                let duration = result.demand.lower_bound()?;
                for &(resource, units) in reservations {
                    result.demand.add_occupancy(resource, units, duration, 1)?;
                    windows::Window::include(&mut result.windows[resource], u128::from(units) * u128::from(duration), 0, 0)?;
                }
                if result.completion > 0 { for &(resource, units) in reservations { result.peak[resource] = result.peak[resource].checked_add(units).ok_or("resident peak overflow")?; } }
                result.expanded_nodes = result.expanded_nodes.checked_add(2).ok_or("structured node count overflow")?;
                result.plan = Arc::new(Plan::Scope { body: result.plan, duration: result.completion });
            }
        }
        for (resource, window) in self.resources.iter().zip(&result.windows) {
            if let Some(window) = window { result.demand.require_duration(window.lower_bound(resource.capacity)?); }
        }
        Ok(result)
    }
    pub fn lower_bound(&self) -> Result<u64, String> { self.summary()?.demand.lower_bound() }
    pub fn compact_witness(&self) -> Result<Option<Witness>, String> {
        if !self.unmapped.is_empty() { return Err("cannot construct a witness with missing structured mappings".into()); }
        self.relationship.require_feasible_upper()?;
        let summary = self.summary()?;
        let lower_bound = summary.demand.lower_bound()?;
        if summary.feasible && lower_bound > summary.completion { return Err("structured lower bound exceeds its feasible plan".into()); }
        Ok(summary.feasible.then(|| Witness { model: Arc::new(self.clone()),
            completion: summary.completion, lower_bound, plan: summary.plan }))
    }

    /// Bounded expansion into the existing exact oracle. Exceeding the caller's
    /// limit is unfinished derivation, never an infeasibility result.
    pub fn expand(&self, maximum_operations: u64) -> Result<Model, crate::workload::DerivationError> {
        let summary = self.summary()?;
        if summary.expanded_nodes > maximum_operations {
            return Err(crate::workload::DerivationError::Exhausted(crate::workload::DerivationLimit::Operations(
                usize::try_from(maximum_operations).unwrap_or(usize::MAX))));
        }
        let mut model = self.empty_model();
        self.expand_node(&self.root, vec![], &mut model)?;
        model.validate()?;
        Ok(model)
    }
    fn expand_node(&self, node: &Node, before: Vec<usize>, model: &mut Model) -> Result<Vec<usize>, String> {
        match node {
            Node::Operation(operation) => {
                let index = model.operations.len();
                let mut operation = operation.clone();
                operation.name = format!("{} @{index}", operation.name);
                operation.predecessors = before;
                model.operations.push(operation);
                Ok(vec![index])
            }
            Node::Compose { order, children } => {
                let mut after = before.clone();
                let mut parallel = Vec::new();
                for child in children {
                    let next = self.expand_node(child, if *order == Order::Serial { after.clone() } else { before.clone() }, model)?;
                    if *order == Order::Serial { after = next; } else { parallel.extend(next); }
                }
                if *order == Order::Serial || children.is_empty() { Ok(after) }
                else { parallel.sort_unstable(); parallel.dedup(); Ok(parallel) }
            }
            Node::Repeat { order, count, body } => {
                if self.summarize(body, &vec![0; self.resources.len()])?.expanded_nodes == 0 { return Ok(before); }
                let mut after = before.clone();
                let mut parallel = Vec::new();
                for _ in 0..*count {
                    let next = self.expand_node(body, if *order == Order::Serial { after.clone() } else { before.clone() }, model)?;
                    if *order == Order::Serial { after = next; } else { parallel.extend(next); }
                }
                if *order == Order::Serial || *count == 0 { Ok(after) }
                else { parallel.sort_unstable(); parallel.dedup(); Ok(parallel) }
            }
            Node::Scope { reservations, body } => {
                let begin = model.operations.len();
                model.operations.push(event(&format!("scope begin @{begin}"), before));
                let after = self.expand_node(body, vec![begin], model)?;
                let end = model.operations.len();
                model.operations.push(event(&format!("scope end @{end}"), after));
                for &(resource, units) in reservations {
                    model.lifetimes.push(Lifetime { resource, units, begin: Event { operation: begin, point: Point::Start }, end: Event { operation: end, point: Point::Completion } });
                }
                Ok(vec![end])
            }
        }
    }
}
fn event(name: &str, predecessors: Vec<usize>) -> Operation {
    Operation { name: name.into(), predecessors, start_predecessors: vec![], latency: 0, reservations: vec![] }
}
