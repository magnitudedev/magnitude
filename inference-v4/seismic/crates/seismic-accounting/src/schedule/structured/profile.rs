//! Exact occupied intervals of a retained feasible plan. Profiles coalesce
//! adjacent equal occupancy; the bound applies to retained intervals, not to
//! expanded operation visits. Unsupported large profiles remain unresolved.
use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Profile {
    pub(super) capacities: Vec<u64>,
    pub(super) intervals: Vec<(usize, u64, u64, u64)>,
}
impl Profile {
    fn empty(resources: &[Resource]) -> Self {
        Self { capacities: resources.iter().map(|r| r.capacity).collect(), intervals: vec![] }
    }
    fn normalize(&mut self, limit: u64) -> Result<bool, String> {
        let mut events = Vec::with_capacity(self.intervals.len().saturating_mul(2));
        for &(resource, begin, end, units) in &self.intervals {
            if begin > end || resource >= self.capacities.len() { return Err("invalid profile interval".into()); }
            if begin == end || units == 0 { continue; }
            events.push((resource, begin, true, units));
            events.push((resource, end, false, units));
        }
        events.sort_unstable();
        self.intervals.clear();
        let (mut resource, mut at, mut used) = (0, 0, 0u64);
        for (next_resource, next_at, begin, units) in events {
            if next_resource != resource {
                if used != 0 { return Err("unclosed profile interval".into()); }
                resource = next_resource;
                at = next_at;
            }
            if next_at > at && used > 0 {
                if let Some(last) = self.intervals.last_mut().filter(|last| last.0 == resource && last.2 == at && last.3 == used) {
                    last.2 = next_at;
                } else {
                    self.intervals.push((resource, at, next_at, used));
                    if self.intervals.len() as u64 > limit { return Ok(false); }
                }
            }
            at = next_at;
            used = if begin { used.checked_add(units).ok_or("profile occupancy overflow")? }
                else { used.checked_sub(units).ok_or("unbalanced profile interval")? };
        }
        if used != 0 { return Err("unclosed profile interval".into()); }
        Ok(true)
    }
    pub(super) fn append(&mut self, other: &Self, origin: u64, units: u64, limit: u64) -> Result<bool, String> {
        if self.capacities.len() != other.capacities.len() { return Err("profile resource mismatch".into()); }
        for &(resource, begin, end, occupied) in &other.intervals {
            self.intervals.push((resource, origin.checked_add(begin).ok_or("profile start overflow")?,
                origin.checked_add(end).ok_or("profile end overflow")?, occupied.checked_mul(units).ok_or("profile occupancy overflow")?));
        }
        self.normalize(limit)
    }
    fn repeated(&self, count: u64, period: u64, limit: u64) -> Result<Option<Self>, String> {
        let mut result = Self { capacities: self.capacities.clone(), intervals: vec![] };
        if count == 0 || self.intervals.is_empty() { return Ok(Some(result)); }
        if period == 0 {
            return Ok(result.append(self, 0, count, limit)?.then_some(result));
        }
        for &(resource, begin, end, units) in &self.intervals {
            if end - begin == period {
                // Consecutive occupied intervals tile one uninterrupted span.
                let end = period.checked_mul(count - 1).and_then(|t| t.checked_add(end)).ok_or("repeated profile overflow")?;
                result.intervals.push((resource, begin, end, units));
            } else {
                // These intervals require distinct retained boundaries. This
                // guard never turns a failed representation into infeasibility.
                if count > limit { return Ok(None); }
                for index in 0..count {
                    let origin = index.checked_mul(period).ok_or("repeated profile overflow")?;
                    result.intervals.push((resource, origin.checked_add(begin).ok_or("repeated profile overflow")?, origin.checked_add(end).ok_or("repeated profile overflow")?, units));
                }
            }
            if !result.normalize(limit)? { return Ok(None); }
        }
        Ok(Some(result))
    }
    #[cfg(test)]
    pub(super) fn finite_peak(&self) -> Vec<u64> {
        let mut peak = vec![0; self.capacities.len()];
        for &(resource, _, _, units) in &self.intervals { peak[resource] = peak[resource].max(units); }
        peak
    }
    fn derive(model: &Structured, node: &Node, plan: &Plan, limit: u64) -> Result<Option<Self>, String> {
        let mut result = Self::empty(&model.resources);
        match plan {
            Plan::Selected(witness) => return witness.reservation_profile(limit),
            Plan::Flat(solution) => {
                let model = solution.model();
                let schedule = solution.schedule();
                for (operation, start) in model.operations.iter().zip(&schedule.starts) {
                    for r in &operation.reservations {
                        let begin = start.checked_add(r.offset).ok_or("profile interval overflow")?;
                        result.intervals.push((r.resource, begin, begin.checked_add(r.duration).ok_or("profile interval overflow")?, r.units));
                    }
                    if !result.normalize(limit)? { return Ok(None); }
                }
                let time = |event: Event| schedule.starts[event.operation].checked_add(
                    if event.point == Point::Completion { model.operations[event.operation].latency } else { 0 }).ok_or("profile lifetime overflow");
                for r in &model.lifetimes {
                    result.intervals.push((r.resource, time(r.begin)?, time(r.end)?, r.units));
                    if !result.normalize(limit)? { return Ok(None); }
                }
            }
            Plan::Operation { .. } => {
                let Node::Operation(operation) = node else { return Err("profile operation mismatch".into()); };
                for r in &operation.reservations { result.intervals.push((r.resource, r.offset, r.offset.checked_add(r.duration).ok_or("profile reservation overflow")?, r.units)); }
                if !result.normalize(limit)? { return Ok(None); }
            }
            Plan::Sequence { children, .. } | Plan::Parallel { children, .. } | Plan::Offset { children, .. } => {
                let Node::Compose { children: nodes, .. } = node else { return Err("profile composition mismatch".into()); };
                if nodes.len() != children.len() { return Err("profile composition length mismatch".into()); }
                let mut next = 0u64;
                for (index, (node, child)) in nodes.iter().zip(children).enumerate() {
                    let origin = match plan { Plan::Sequence { .. } => next, Plan::Offset { starts, .. } => *starts.get(index).ok_or("profile start missing")?, _ => 0 };
                    let Some(profile) = Self::derive(model, node, child, limit)? else { return Ok(None); };
                    if !result.append(&profile, origin, 1, limit)? { return Ok(None); }
                    next = next.checked_add(child.duration()).ok_or("profile sequence overflow")?;
                }
            }
            Plan::Repeat { body, count, concurrent, .. } => {
                if *count == 0 { return Ok(Some(result)); }
                let Node::Repeat { body: node, .. } = node else { return Err("profile repetition mismatch".into()); };
                let Some(profile) = Self::derive(model, node, body, limit)? else { return Ok(None); };
                let waves = count / concurrent;
                let Some(full) = profile.repeated(waves, body.duration(), limit)? else { return Ok(None); };
                if !result.append(&full, 0, *concurrent, limit)? { return Ok(None); }
                let remainder = count % concurrent;
                if remainder > 0 && !result.append(&profile, waves.checked_mul(body.duration()).ok_or("profile wave overflow")?, remainder, limit)? { return Ok(None); }
            }
            Plan::Scope { body, duration } => {
                let Node::Scope { reservations, body: node } = node else { return Err("profile scope mismatch".into()); };
                let Some(profile) = Self::derive(model, node, body, limit)? else { return Ok(None); };
                result = profile;
                for &(resource, units) in reservations { result.intervals.push((resource, 0, *duration, units)); }
                if !result.normalize(limit)? { return Ok(None); }
            }
            Plan::Periodic { body, count, period, .. } => {
                let Some(profile) = body.reservation_profile(limit)? else { return Ok(None); };
                return profile.repeated(*count, *period, limit);
            }
        }
        result.capacities = model.resources.iter().map(|resource| resource.capacity).collect();
        Ok(Some(result))
    }
}
impl Witness {
    pub(super) fn reservation_profile(&self, maximum_intervals: u64) -> Result<Option<Profile>, String> {
        Profile::derive(&self.model, &self.model.root, &self.plan, maximum_intervals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn repeated(count: u64, duration: u64) -> Structured {
        Structured { relationship: crate::authority::ModelRelationship::hypothetical_execution(),
            identity: "compact occupied intervals".into(), timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
            resources: vec![Resource { name: "service".into(), capacity: 2, unit: CapacityUnit::Slots }],
            root: Arc::new(Node::Repeat { order: Order::Serial, count, body: Arc::new(Node::Operation(Operation {
                name: "body".into(), predecessors: vec![], start_predecessors: vec![], latency: 4,
                reservations: vec![Reservation { resource: 0, offset: 0, duration, units: 1 }],
            })) }), unmapped: vec![] }
    }
    #[test]
    fn trillion_adjacent_reservations_retain_one_interval() {
        let model = repeated(1_000_000_000_000, 4);
        let witness = model.compact_witness().unwrap().unwrap();
        assert!(witness.expand(1).is_err());
        let profile = witness.reservation_profile(1).unwrap().unwrap();
        assert_eq!(profile.intervals, vec![(0, 0, 4_000_000_000_000, 1)]);
        assert_eq!(profile.finite_peak(), vec![1]);
        assert_eq!(profile.peak(2_000_000_000_000).unwrap(), Some(vec![2]));
    }
    #[test]
    fn fragmented_profiles_exhaust_their_bound_without_losing_intervals() {
        let witness = repeated(3, 2).compact_witness().unwrap().unwrap();
        assert!(witness.reservation_profile(2).unwrap().is_none());
        assert_eq!(witness.reservation_profile(3).unwrap().unwrap().intervals,
            vec![(0, 0, 2, 1), (0, 4, 6, 1), (0, 8, 10, 1)]);
    }
    #[test]
    fn nested_scope_profiles_restore_parent_capacity_and_hold_full_lifetimes() {
        let mut model = repeated(2, 2);
        model.root = Arc::new(Node::Scope { reservations: vec![(0, 1)], body: model.root });
        let witness = model.compact_witness().unwrap().unwrap();
        let profile = witness.reservation_profile(4).unwrap().unwrap();
        assert_eq!(profile.capacities, vec![2]);
        assert_eq!(profile.intervals, vec![(0, 0, 2, 2), (0, 2, 4, 1), (0, 4, 6, 2), (0, 6, 8, 1)]);
        assert_eq!(profile.finite_peak(), vec![2]);
    }
}
