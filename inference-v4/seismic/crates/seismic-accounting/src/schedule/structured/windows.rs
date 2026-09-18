//! Necessary occupied work inside a dependency-constrained time window. Every
//! counted interval starts at least `head` ticks after region entry and ends at
//! least `tail` ticks before region completion. Capacity therefore requires
//! completion >= head + ceil(work / capacity) + tail. The window is derived
//! from mandatory dependencies, never from a chosen feasible schedule.
#[derive(Clone, Debug)]
pub(super) struct Window {
    pub(super) work: u128,
    pub(super) head: u64,
    pub(super) tail: u64,
}
impl Window {
    pub(super) fn include(destination: &mut Option<Self>, work: u128, head: u64, tail: u64) -> Result<(), String> {
        if work == 0 { return Ok(()); }
        if let Some(current) = destination {
            current.work = current.work.checked_add(work).ok_or("structured window demand overflow")?;
            current.head = current.head.min(head);
            current.tail = current.tail.min(tail);
        } else { *destination = Some(Self { work, head, tail }); }
        Ok(())
    }
    pub(super) fn lower_bound(&self, capacity: u64) -> Result<u64, String> {
        let duration = u64::try_from(self.work.div_ceil(u128::from(capacity))).map_err(|_| "structured window duration overflow")?;
        self.head.checked_add(duration).and_then(|t| t.checked_add(self.tail)).ok_or("structured window bound overflow".into())
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    #[test]
    fn dependency_windows_are_sound_for_mixed_offsets_and_serial_children() {
        for capacity in [1, 2] {
            for left_offset in [0, 1] {
                for right_offset in [0, 2] {
                    let leaf = |name: &str, offset: u64| Arc::new(Node::Operation(Operation { name: name.into(),
                        predecessors: vec![], start_predecessors: vec![], latency: 4,
                        reservations: vec![Reservation { resource: 0, offset, duration: 1, units: 1 }] }));
                    let model = Structured { relationship: crate::authority::ModelRelationship::hypothetical_execution(),
                        identity: "dependency window oracle".into(), timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
                        resources: vec![Resource { name: "service".into(), capacity, unit: CapacityUnit::Slots }],
                        root: Arc::new(Node::Compose { order: Order::Parallel, children: vec![
                            Arc::new(Node::Compose { order: Order::Serial, children: vec![leaf("left first", left_offset), leaf("left second", left_offset)] }),
                            leaf("right", right_offset),
                        ] }), unmapped: vec![] };
                    let exact = model.expand(3).unwrap().solve(100_000).unwrap();
                    assert!(exact.is_optimal());
                    assert!(model.lower_bound().unwrap() <= exact.schedule().completion);
                }
            }
        }
    }
}
