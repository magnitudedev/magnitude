//! Search over intervals of start assignments in the owning execution model.
//! Subdivision is disjoint and exhaustive. Dependency propagation, compulsory
//! resource occupancy, and forced resource/static orders relax these same
//! constraints; they do not introduce another execution or cost interpretation.
use super::{Event, Model, Point, Schedule};

#[derive(Clone, Copy, Debug)]
struct Window {
    first: u64,
    last: u64,
}
#[derive(Clone)]
struct Region {
    starts: Vec<Window>,
    floor: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Edge {
    before: usize,
    after: usize,
    /// start(after) >= start(before) + lag; event offsets can make lag negative.
    lag: i128,
}
#[derive(Clone, Copy)]
struct Endpoint {
    operation: usize,
    offset: u64,
}
impl Endpoint {
    fn first(self, starts: &[Window]) -> u64 {
        starts[self.operation].first + self.offset
    }
    fn last(self, starts: &[Window]) -> u64 {
        starts[self.operation].last + self.offset
    }
    fn before(self, other: Self) -> Edge {
        Edge {
            before: self.operation,
            after: other.operation,
            lag: i128::from(self.offset) - i128::from(other.offset),
        }
    }
    fn can_precede(self, other: Self, starts: &[Window]) -> bool {
        if self.operation == other.operation {
            self.offset <= other.offset
        } else {
            self.first(starts) <= other.last(starts)
        }
    }
}
struct Occupancy {
    resource: usize,
    units: u64,
    begin: Endpoint,
    end: Endpoint,
    /// A fixed reservation is always positive; a lifetime may be empty.
    positive: bool,
}

pub(super) struct State {
    best: Option<Schedule>,
    frontier: Vec<Region>,
    edges: Vec<Edge>,
    occupancy: Vec<Occupancy>,
    order: Vec<usize>,
    floor: u64,
    horizon: u64,
    examined: u64,
}
impl State {
    pub fn new(
        model: &Model,
        order: Vec<usize>,
        floor: u64,
        horizon: u64,
        best: Option<Schedule>,
    ) -> Self {
        let frontier = if best.as_ref().is_some_and(|s| s.completion == floor) {
            Vec::new()
        } else {
            vec![Region {
                starts: model
                    .operations
                    .iter()
                    .map(|op| Window {
                        first: 0,
                        last: horizon - op.latency,
                    })
                    .collect(),
                floor,
            }]
        };
        let (edges, occupancy) = constraints(model);
        Self {
            best,
            frontier,
            edges,
            occupancy,
            order,
            floor,
            horizon,
            examined: 0,
        }
    }
    pub fn best(&self) -> Option<&Schedule> {
        self.best.as_ref()
    }
    pub fn examined(&self) -> u64 {
        self.examined
    }
    pub fn incomplete(&self) -> bool {
        !self.frontier.is_empty()
    }
    pub fn lower_bound(&self) -> u64 {
        self.frontier
            .iter()
            .map(|r| r.floor)
            .chain(self.best.as_ref().map(|s| s.completion))
            .min()
            .unwrap_or(self.floor)
    }
    pub fn advance(&mut self, model: &Model, budget: u64) -> Result<(), String> {
        let Self {
            best,
            frontier,
            edges,
            occupancy,
            order,
            horizon,
            examined,
            ..
        } = self;
        let stop = examined
            .checked_add(budget)
            .ok_or("schedule search budget overflow")?;
        while *examined < stop && !frontier.is_empty() {
            // Bounds are mathematical floors, not a heuristic score. Equal floors
            // retain deterministic depth-first traversal without ranking hardware.
            let index = frontier
                .iter()
                .enumerate()
                .rev()
                .min_by_key(|(_, r)| r.floor)
                .unwrap()
                .0;
            let mut region = frontier.swap_remove(index);
            *examined += 1;
            if best.as_ref().is_some_and(|s| region.floor >= s.completion) {
                continue;
            }
            let deadline = best.as_ref().map_or(*horizon, |s| s.completion - 1);
            if !propagate(model, edges, occupancy, &mut region, deadline)? {
                continue;
            }
            if best.as_ref().is_some_and(|s| region.floor >= s.completion) {
                continue;
            }
            // If the componentwise earliest assignment is feasible, it attains this
            // region's dependency floor and no member of this region can improve it.
            let candidate = Schedule {
                starts: region.starts.iter().map(|w| w.first).collect(),
                completion: region
                    .starts
                    .iter()
                    .zip(&model.operations)
                    .map(|(w, op)| w.first + op.latency)
                    .max()
                    .unwrap_or(0),
            };
            if model.check_schedule(&candidate).is_ok() {
                *best = Some(candidate);
                frontier.retain(|r| r.floor < best.as_ref().unwrap().completion);
                continue;
            }
            // Source identity is used only for deterministic partitioning. Both
            // halves remain in the frontier, including delayed earlier operations.
            let Some(&operation) = order
                .iter()
                .find(|&&i| region.starts[i].first != region.starts[i].last)
            else {
                continue;
            };
            let window = region.starts[operation];
            let middle = window.first + (window.last - window.first) / 2;
            let mut later = region.clone();
            later.starts[operation].first = middle + 1;
            region.starts[operation].last = middle;
            frontier.push(later);
            frontier.push(region);
        }
        Ok(())
    }
}

fn constraints(model: &Model) -> (Vec<Edge>, Vec<Occupancy>) {
    let mut edges = Vec::new();
    let mut occupancy = Vec::new();
    for (i, op) in model.operations.iter().enumerate() {
        edges.extend(op.predecessors.iter().map(|&p| Edge {
            before: p,
            after: i,
            lag: i128::from(model.operations[p].latency),
        }));
        edges.extend(op.start_predecessors.iter().map(|&p| Edge {
            before: p,
            after: i,
            lag: 0,
        }));
        occupancy.extend(op.reservations.iter().map(|r| Occupancy {
            resource: r.resource,
            units: r.units,
            begin: Endpoint {
                operation: i,
                offset: r.offset,
            },
            end: Endpoint {
                operation: i,
                offset: r.offset + r.duration,
            },
            positive: true,
        }));
    }
    let endpoint = |event: Event| Endpoint {
        operation: event.operation,
        offset: if event.point == Point::Completion {
            model.operations[event.operation].latency
        } else {
            0
        },
    };
    for lifetime in &model.lifetimes {
        let begin = endpoint(lifetime.begin);
        let end = endpoint(lifetime.end);
        edges.push(begin.before(end));
        // Oversized resident storage can only have an empty lifetime. This is a
        // constraint, not a reason to misclassify all zero-length events.
        if lifetime.units > model.resources[lifetime.resource].capacity {
            edges.push(end.before(begin));
        }
        occupancy.push(Occupancy {
            resource: lifetime.resource,
            units: lifetime.units,
            begin,
            end,
            positive: false,
        });
    }
    (edges, occupancy)
}

fn propagate(
    model: &Model,
    base: &[Edge],
    occupancy: &[Occupancy],
    region: &mut Region,
    deadline: u64,
) -> Result<bool, String> {
    for (w, op) in region.starts.iter_mut().zip(&model.operations) {
        let Some(last) = deadline.checked_sub(op.latency) else {
            return Ok(false);
        };
        w.last = w.last.min(last);
        if w.first > w.last {
            return Ok(false);
        }
    }
    let mut edges: std::collections::BTreeSet<_> = base.iter().copied().collect();
    loop {
        if !difference_closure(&mut region.starts, &edges) {
            return Ok(false);
        }
        if !compulsory_fits(model, occupancy, &region.starts)? {
            return Ok(false);
        }
        let old_len = edges.len();
        for (index, a) in occupancy.iter().enumerate() {
            if a.positive && a.units > model.resources[a.resource].capacity {
                return Ok(false);
            }
            if !(a.positive || a.end.first(&region.starts) > a.begin.last(&region.starts)) {
                continue;
            }
            for b in &occupancy[index + 1..] {
                if a.resource != b.resource
                    || u128::from(a.units) + u128::from(b.units)
                        <= u128::from(model.resources[a.resource].capacity)
                    || !(b.positive || b.end.first(&region.starts) > b.begin.last(&region.starts))
                {
                    continue;
                }
                let a_first = a.end.can_precede(b.begin, &region.starts);
                let b_first = b.end.can_precede(a.begin, &region.starts);
                match (a_first, b_first) {
                    (false, false) => return Ok(false),
                    (true, false) => {
                        edges.insert(a.end.before(b.begin));
                    }
                    (false, true) => {
                        edges.insert(b.end.before(a.begin));
                    }
                    (true, true) => {}
                }
            }
        }
        let ranges: Vec<_> = region.starts.iter().map(|w| (w.first, w.last)).collect();
        let Some(static_edges) = super::static_order::interval_edges(model, &ranges)? else {
            return Ok(false);
        };
        edges.extend(static_edges.into_iter().map(|(before, after)| Edge {
            before,
            after,
            lag: 0,
        }));
        if edges.len() == old_len {
            break;
        }
    }
    region.floor = region.floor.max(
        region
            .starts
            .iter()
            .zip(&model.operations)
            .map(|(w, op)| w.first + op.latency)
            .max()
            .unwrap_or(0),
    );
    Ok(region.floor <= deadline)
}

/// Bellman-Ford closure of existing timing inequalities in both directions.
/// A change after n passes establishes a positive cycle, even for large ticks;
/// infeasibility never requires waiting for a bound to creep across the horizon.
fn difference_closure(starts: &mut [Window], edges: &std::collections::BTreeSet<Edge>) -> bool {
    for _ in 0..starts.len() {
        let mut changed = false;
        for &Edge { before, after, lag } in edges {
            let first = i128::from(starts[before].first) + lag;
            let last = i128::from(starts[after].last) - lag;
            if first > i128::from(starts[after].last) || last < i128::from(starts[before].first) {
                return false;
            }
            if first > i128::from(starts[after].first) {
                starts[after].first = first as u64;
                changed = true;
            }
            if last < i128::from(starts[before].last) {
                starts[before].last = last as u64;
                changed = true;
            }
        }
        if !changed {
            return true;
        }
    }
    // The nth pass may finish a path of n-1 edges in reverse iteration order.
    // Only a still-unsatisfied inequality establishes a remaining cycle.
    edges.iter().all(|e| {
        i128::from(starts[e.after].first) >= i128::from(starts[e.before].first) + e.lag
            && i128::from(starts[e.before].last) <= i128::from(starts[e.after].last) - e.lag
    })
}

fn compulsory_fits(
    model: &Model,
    occupancy: &[Occupancy],
    starts: &[Window],
) -> Result<bool, String> {
    for (resource, capacity) in model.resources.iter().enumerate() {
        let mut events = Vec::new();
        for claim in occupancy.iter().filter(|r| r.resource == resource) {
            // Every possible placement covers [latest begin, earliest end).
            let begin = claim.begin.last(starts);
            let end = claim.end.first(starts);
            if begin < end {
                events.push((begin, true, claim.units));
                events.push((end, false, claim.units));
            }
        }
        events.sort_unstable();
        let mut occupied = 0u128;
        for (_, acquire, units) in events {
            if acquire {
                occupied = occupied
                    .checked_add(u128::from(units))
                    .ok_or("compulsory demand overflow")?;
            } else {
                occupied = occupied
                    .checked_sub(u128::from(units))
                    .ok_or("compulsory demand underflow")?;
            }
            if occupied > u128::from(capacity.capacity) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}
