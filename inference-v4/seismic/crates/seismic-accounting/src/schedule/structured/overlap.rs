//! Retained start-offset search for different child schedules. Fixed child
//! profiles define feasible upper schedules only: their restricted optimum
//! never raises the lower bound of the original parallel region.
use super::*;

pub(super) struct Search {
    children: Vec<Arc<Plan>>,
    profile_limit: u64,
    frontier: Option<super::super::Search>,
    best: Option<Arc<Plan>>,
}
impl Search {
    pub(super) fn settled(children: Vec<Arc<Plan>>, profile_limit: u64) -> Self {
        Self { children, profile_limit, frontier: None, best: None }
    }
    pub(super) fn new(model: &Structured, children: Vec<Arc<Plan>>, profile_limit: u64) -> Result<Self, String> {
        let frontier = offset_model(model, &children, profile_limit)?.map(|model| model.start_search()).transpose()?;
        Ok(Self { children, profile_limit, frontier, best: None })
    }
    pub(super) fn matches(&self, children: &[Arc<Plan>]) -> bool { self.children == children }
    pub(super) fn can_advance(&self) -> bool { self.frontier.is_some() }
    pub(super) fn advance(&mut self, budget: u64) -> Result<Option<Arc<Plan>>, String> {
        if let Some(frontier) = &mut self.frontier {
            match frontier.advance(budget)? {
                super::super::SearchOutcome::Feasible(solution) => {
                    self.best = Some(Arc::new(Plan::Offset { children: self.children.clone(), starts: solution.schedule().starts.clone(),
                        duration: solution.schedule().completion, profile_limit: self.profile_limit }));
                    if solution.is_optimal() { self.frontier = None; }
                }
                // Serial placement of checked child profiles is always feasible.
                super::super::SearchOutcome::Infeasible => return Err("feasible child profiles became infeasible".into()),
                super::super::SearchOutcome::Incomplete { .. } => {},
            }
        }
        Ok(self.best.clone())
    }
}
fn offset_model(model: &Structured, children: &[Arc<Plan>], profile_limit: u64) -> Result<Option<Model>, String> {
    let mut result = model.empty_model();
    for (index, child) in children.iter().enumerate() {
        let Plan::Selected(witness) = child.as_ref() else { return Err("offset child witness missing".into()); };
        let Some(profile) = witness.reservation_profile(profile_limit)? else { return Ok(None); };
        result.operations.push(Operation { name: format!("parallel child @{index}"), predecessors: vec![], start_predecessors: vec![],
            latency: witness.completion, reservations: profile.intervals.into_iter().map(|(resource, begin, end, units)| Reservation {
                resource, offset: begin, duration: end - begin, units,
            }).collect() });
    }
    result.validate()?;
    Ok(Some(result))
}
pub(super) fn checked_peak(model: &Structured, children: &[Arc<Plan>], starts: &[u64], duration: u64, profile_limit: u64) -> Result<Vec<u64>, String> {
    let model = offset_model(model, children, profile_limit)?.ok_or("offset profile no longer fits its derivation limit")?;
    let schedule = Schedule { starts: starts.to_vec(), completion: duration };
    model.check_execution_upper(&schedule)?;
    let mut events = Vec::new();
    for (operation, start) in model.operations.iter().zip(starts) {
        for r in &operation.reservations {
            events.push((r.resource, start + r.offset, true, r.units));
            events.push((r.resource, start + r.offset + r.duration, false, r.units));
        }
    }
    events.sort_unstable();
    let mut used = vec![0u64; model.resources.len()];
    let mut peak = used.clone();
    for (resource, _, begin, units) in events {
        used[resource] = if begin { used[resource].checked_add(units).ok_or("offset occupancy overflow")? }
            else { used[resource].checked_sub(units).ok_or("offset occupancy mismatch")? };
        peak[resource] = peak[resource].max(used[resource]);
    }
    Ok(peak)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn operation(name: &str, latency: u64, offset: u64, duration: u64, units: u64) -> Arc<Node> {
        Arc::new(Node::Operation(Operation { name: name.into(), predecessors: vec![], start_predecessors: vec![], latency,
            reservations: vec![Reservation { resource: 0, offset, duration, units }] }))
    }
    fn model(children: Vec<Arc<Node>>, capacity: u64) -> Structured {
        Structured { relationship: crate::authority::ModelRelationship::hypothetical_execution(),
            identity: "heterogeneous shifted profiles".into(), timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
            resources: vec![Resource { name: "service".into(), capacity, unit: CapacityUnit::Slots }],
            root: Arc::new(Node::Compose { order: Order::Parallel, children }), unmapped: vec![] }
    }
    #[test]
    fn heterogeneous_offsets_close_the_dependency_window_bound() {
        // Both children reserve the same service late in their lifetime. A
        // staggered start fills the gap without shortening either dependency.
        let model = model(vec![operation("left", 5, 1, 2, 1), operation("right", 4, 1, 1, 1)], 1);
        let reference = model.expand(2).unwrap().solve(100_000).unwrap();
        assert!(reference.is_optimal());
        let mut search = Refinement::new(model, 1).unwrap();
        let mut selected = None;
        for _ in 0..100 {
            if let RefinementOutcome::Feasible(witness) = search.advance(1).unwrap() {
                if witness.is_optimal() { selected = Some(witness); break; }
            }
        }
        let selected = selected.expect("retained offsets complete");
        assert_eq!(selected.completion(), reference.schedule().completion);
        assert_eq!(selected.completion(), 6);
        selected.expand(2).unwrap();
        let Plan::Offset { children, duration, profile_limit, .. } = selected.plan.as_ref() else { panic!("offset plan") };
        assert!(selected.model.witness(Arc::new(Plan::Offset { children: children.clone(), starts: vec![0, 0],
            duration: *duration, profile_limit: *profile_limit })).is_err());
    }
    #[test]
    fn restricted_offset_optimum_never_promotes_a_lower_bound() {
        let model = model(vec![operation("two", 2, 0, 2, 2), operation("three", 3, 0, 3, 2), operation("four", 4, 0, 4, 2)], 3);
        let reference = model.expand(3).unwrap().solve(100_000).unwrap();
        assert!(reference.is_optimal());
        assert_eq!(reference.schedule().completion, 9);
        let mut search = Refinement::new(model, 1).unwrap();
        let RefinementOutcome::Feasible(selected) = search.advance(100_000).unwrap() else { panic!("checked upper") };
        assert_eq!(selected.completion(), 9);
        assert_eq!(selected.lower_bound(), 6);
        assert!(!selected.is_optimal());
        selected.expand(3).unwrap();
        // Exhaustion of this restricted frontier is still an unresolved
        // original problem, even when its best upper happens to be optimal.
        let RefinementOutcome::Feasible(again) = search.advance(100_000).unwrap() else { panic!("retained upper") };
        assert_eq!(again, selected);
    }
    #[test]
    fn offset_profiles_keep_nested_resident_capacity_unavailable() {
        let mut model = model(vec![operation("left", 5, 1, 2, 1), operation("right", 4, 1, 1, 1)], 3);
        for _ in 0..2 { model.root = Arc::new(Node::Scope { reservations: vec![(0, 1)], body: model.root }); }
        let reference = model.expand(6).unwrap().solve(100_000).unwrap();
        assert!(reference.is_optimal());
        let mut search = Refinement::new(model, 1).unwrap();
        let RefinementOutcome::Feasible(selected) = search.advance(100_000).unwrap() else { panic!("scoped offsets") };
        assert!(selected.is_optimal());
        assert_eq!(selected.completion(), reference.schedule().completion);
        assert_eq!(selected.peak().unwrap(), vec![3]);
        selected.expand(6).unwrap();
    }
}
