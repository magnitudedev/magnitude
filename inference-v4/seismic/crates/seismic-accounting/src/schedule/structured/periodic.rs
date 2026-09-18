//! Checked periodic upper schedules. A bounded body supplies exact occupied
//! intervals. Complete periods plus a residue sweep bound an infinite stream,
//! hence every finite prefix, without enumerating its repetition count.
use super::*;

pub(super) use super::profile::Profile;
impl Profile {
    pub(super) fn peak(&self, period: u64) -> Result<Option<Vec<u64>>, String> {
        if period == 0 { return Err("zero initiation period".into()); }
        let mut base = vec![0u64; self.capacities.len()];
        let mut events = vec![Vec::new(); self.capacities.len()];
        for &(resource, start, end, units) in &self.intervals {
            let duration = end.checked_sub(start).ok_or("reversed periodic interval")?;
            let full = (duration / period).checked_mul(units).ok_or("periodic occupancy overflow")?;
            base[resource] = base[resource].checked_add(full).ok_or("periodic occupancy overflow")?;
            let tail = duration % period;
            if tail == 0 { continue; }
            let start = start % period;
            let first = tail.min(period - start);
            events[resource].push((start, true, units));
            events[resource].push((start + first, false, units));
            if first < tail {
                events[resource].push((0, true, units));
                events[resource].push((tail - first, false, units));
            }
        }
        let mut peak = base.clone();
        for (resource, events) in events.iter_mut().enumerate() {
            events.sort_unstable();
            let mut used = base[resource];
            for &(_, start, units) in events.iter() {
                used = if start { used.checked_add(units).ok_or("periodic peak overflow")? }
                    else { used.checked_sub(units).ok_or("invalid periodic interval")? };
                peak[resource] = peak[resource].max(used);
            }
            if peak[resource] > self.capacities[resource] { return Ok(None); }
        }
        Ok(Some(peak))
    }
}
impl Witness {
    pub(super) fn periodic_profile(&self, maximum_intervals: u64) -> Result<Option<Profile>, String> {
        self.reservation_profile(maximum_intervals)
    }
    pub(super) fn periodic_peak(&self, period: u64, maximum_intervals: u64) -> Result<Option<Vec<u64>>, String> {
        match self.periodic_profile(maximum_intervals)? { Some(profile) => profile.peak(period), None => Ok(None) }
    }
    pub(super) fn periodic_plan(&self, count: u64, period: u64, maximum_intervals: u64, profile: &Profile) -> Result<Option<Arc<Plan>>, String> {
        if count == 0 || profile.peak(period)?.is_none() { return Ok(None); }
        let duration = period.checked_mul(count - 1).and_then(|t| t.checked_add(self.completion)).ok_or("periodic completion overflow")?;
        Ok(Some(Arc::new(Plan::Periodic { body: Arc::new(self.clone()), count, period, duration, profile_limit: maximum_intervals })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model(count: u64) -> Structured {
        Structured { relationship: crate::authority::ModelRelationship::hypothetical_execution(), identity: "overlapping pipelines".into(),
            timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
            resources: vec![Resource { name: "issue".into(), capacity: 1, unit: CapacityUnit::Slots }],
            root: Arc::new(Node::Repeat { order: Order::Parallel, count, body: Arc::new(Node::Operation(Operation {
                name: "long latency".into(), predecessors: vec![], start_predecessors: vec![], latency: 10,
                reservations: vec![Reservation { resource: 0, offset: 2, duration: 2, units: 1 }],
            })) }), unmapped: vec![] }
    }
    #[test]
    fn compact_pipeline_closes_trillion_copy_bound_and_matches_small_oracle() {
        for count in [3, 1_000_000_000_000] {
            let model = model(count);
            let mut search = Refinement::new(model.clone(), 1).unwrap();
            let mut selected = None;
            for _ in 0..10 {
                if let RefinementOutcome::Feasible(witness) = search.advance(4).unwrap() {
                    if witness.is_optimal() { selected = Some(witness); break; }
                }
            }
            let selected = selected.expect("periodic frontier completes");
            selected.check_execution_upper().unwrap();
            assert_eq!(selected.completion(), 10 + 2 * (count - 1));
            if count == 3 {
                let flat = model.expand(3).unwrap().solve(100_000).unwrap();
                assert!(flat.is_optimal());
                assert_eq!(selected.expand(3).unwrap().1.completion, flat.schedule().completion);
            }
        }
    }
    #[test]
    fn periodic_reservations_keep_offsets_wrap_and_resident_lifetimes() {
        let mut model = model(4);
        let Node::Repeat { body, .. } = model.root.as_ref() else { unreachable!() };
        model.root = body.clone();
        model.resources[0].capacity = 2;
        model.root = Arc::new(Node::Scope { reservations: vec![(0, 1)], body: model.root });
        let witness = model.compact_witness().unwrap().unwrap();
        assert!(witness.periodic_peak(2, 3).unwrap().is_none());
        assert_eq!(witness.periodic_peak(9, 3).unwrap(), Some(vec![2]));
        assert_eq!(witness.periodic_peak(10, 3).unwrap(), Some(vec![2]));
    }
}
