//! Ownership for the watcher's serial-device progress protocol.
//! Only a later observed route/scratch event, or the actual step completion,
//! proves that a submitted frame's conversions and consumers have finished.
use super::arena::{
    Arena, Error as ArenaError, ExpertId, FillLease, Key, LayerId, Location, Phase, ReadLease,
};
use super::shared_bank::SharedBankPlan;
use std::collections::HashMap;

/// A point in the device's visit order of streamed layers. Steps run in
/// submission order, each through its layers in order, except that a
/// layer-major prompt group interleaves its steps' layers: every step of the
/// group at a layer, in sequence order, before any at the next. So points
/// order by `group` (the sequence of the group's first step; a lone step is
/// its own group), then layer, then sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Clock {
    pub group: i32,
    pub layer: usize,
    pub sequence: i32,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Error {
    InvalidRequest,
    BackwardClock,
    DuplicateRequest,
    NotSubmitted,
    Arena(ArenaError),
}
impl From<ArenaError> for Error {
    fn from(e: ArenaError) -> Self {
        Self::Arena(e)
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Choice {
    pub location: Location,
    pub needs_conversion: bool,
}
#[derive(Debug)]
pub(crate) struct Resolution {
    pub choices: Vec<Choice>,
    pub evicted: Vec<Location>,
}
struct Frame {
    clock: Clock,
    group: usize,
    reads: Vec<ReadLease>,
    fills: Vec<FillLease>,
    submitted: bool,
}
pub(crate) struct SharedDirectory {
    groups: Vec<Arena>,
    group_of_layer: Vec<usize>,
    capacities: Vec<usize>,
    experts: u32,
    observed: Option<Clock>,
    frames: Vec<Frame>,
}
impl SharedDirectory {
    #[cfg(test)]
    pub(crate) fn controlled() -> Self {
        Self::from_parts(vec![8], vec![0], 256).unwrap()
    }
    pub fn new(plan: &SharedBankPlan, experts: u64) -> Result<Self, Error> {
        let experts = u32::try_from(experts).map_err(|_| Error::InvalidRequest)?;
        let capacities = plan
            .groups
            .iter()
            .map(|g| usize::try_from(g.records).map_err(|_| Error::InvalidRequest))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_parts(capacities, plan.group_of_layer.clone(), experts)
    }
    fn from_parts(
        capacities: Vec<usize>,
        group_of_layer: Vec<usize>,
        experts: u32,
    ) -> Result<Self, Error> {
        if group_of_layer.is_empty() || group_of_layer.iter().any(|g| *g >= capacities.len()) {
            return Err(Error::InvalidRequest);
        }
        let mut groups = Vec::new();
        for (index, &capacity) in capacities.iter().enumerate() {
            if capacity > i32::MAX as usize {
                return Err(Error::InvalidRequest);
            }
            let members = group_of_layer
                .iter()
                .enumerate()
                .filter(|(_, g)| **g == index)
                .map(|(l, _)| {
                    u32::try_from(l)
                        .map(|l| (LayerId(l), experts))
                        .map_err(|_| Error::InvalidRequest)
                })
                .collect::<Result<Vec<_>, _>>()?;
            groups.push(Arena::new(capacity, &members)?);
        }
        Ok(Self {
            groups,
            group_of_layer,
            capacities,
            experts,
            observed: None,
            frames: Vec::new(),
        })
    }
    fn valid_clock(&self, clock: Clock) -> Result<(), Error> {
        if clock.group <= 0
            || clock.group > clock.sequence
            || clock.layer >= self.group_of_layer.len()
        {
            return Err(Error::InvalidRequest);
        }
        Ok(())
    }
    /// Records of the bank group `layer` belongs to: the most experts one
    /// request may hold.
    pub fn capacity(&self, layer: usize) -> Result<usize, Error> {
        let group = self
            .group_of_layer
            .get(layer)
            .ok_or(Error::InvalidRequest)?;
        Ok(self.capacities[*group])
    }
    pub fn validate_ports(&self, sizes: &[(usize, usize)]) -> Result<(), Error> {
        if sizes.len() != self.group_of_layer.len() {
            return Err(Error::InvalidRequest);
        }
        for (layer, &(stamps, commits)) in sizes.iter().enumerate() {
            let capacity = self.capacities[self.group_of_layer[layer]];
            if stamps != capacity || commits != capacity {
                return Err(Error::InvalidRequest);
            }
        }
        Ok(())
    }
    /// Consume per-slot GPU evidence only after a later published device
    /// point. Uncommitted fills become empty, never readable canonical data.
    pub fn retire_progress(
        &mut self,
        clock: Clock,
        read_commit: impl Fn(usize, usize) -> i32,
    ) -> Result<Vec<(Clock, Location)>, Error> {
        self.valid_clock(clock)?;
        if self.observed.is_some_and(|old| clock < old) {
            return Err(Error::BackwardClock);
        }
        let invalid = self.retire_where(|f| f.clock < clock, read_commit)?;
        self.observed = Some(clock);
        Ok(invalid)
    }
    pub fn retire_step(
        &mut self,
        sequence: i32,
        read_commit: impl Fn(usize, usize) -> i32,
    ) -> Result<Vec<(Clock, Location)>, Error> {
        if sequence <= 0 {
            return Err(Error::InvalidRequest);
        }
        self.retire_where(|f| f.clock.sequence <= sequence, read_commit)
    }
    fn retire_where(
        &mut self,
        done: impl Fn(&Frame) -> bool,
        read_commit: impl Fn(usize, usize) -> i32,
    ) -> Result<Vec<(Clock, Location)>, Error> {
        if self.frames.iter().any(|f| done(f) && !f.submitted) {
            return Err(Error::NotSubmitted);
        }
        let mut invalid = Vec::new();
        let mut index = 0;
        while index < self.frames.len() {
            if !done(&self.frames[index]) {
                index += 1;
                continue;
            }
            let frame = self.frames.remove(index);
            let arena = &mut self.groups[frame.group];
            for fill in frame.fills {
                let location = fill.location();
                if read_commit(frame.clock.layer, location.slot) == frame.clock.sequence {
                    let read = arena.publish_after_conversion(fill)?;
                    arena.release_after_reader_completion(read)?;
                } else {
                    arena.fail_fill(fill)?;
                    invalid.push((frame.clock, location));
                }
            }
            for read in frame.reads {
                arena.release_after_reader_completion(read)?;
            }
        }
        Ok(invalid)
    }
    /// The watcher passes the group's coherent GPU stamps, which include
    /// all-hit layers that never generated a host demand request.
    pub fn sync_stamps(
        &mut self,
        layer: usize,
        stamps: impl Iterator<Item = i32>,
    ) -> Result<(), Error> {
        let group = *self
            .group_of_layer
            .get(layer)
            .ok_or(Error::InvalidRequest)?;
        let stamps: Vec<_> = stamps.collect();
        if stamps.len() != self.capacities[group] {
            return Err(Error::InvalidRequest);
        }
        for (slot, stamp) in stamps.into_iter().enumerate() {
            if stamp > 0 {
                self.groups[group].observe_ready_phase(slot, Phase(stamp as u64))?;
            }
        }
        Ok(())
    }
    /// Pin every hit before reserving ANY miss. Repeated choices share one
    /// lease and conversion identity, while their original order is retained.
    pub fn begin(&mut self, clock: Clock, routes: &[i32]) -> Result<Resolution, Error> {
        self.valid_clock(clock)?;
        if self.observed != Some(clock) {
            return Err(Error::BackwardClock);
        }
        if self.frames.iter().any(|f| f.clock == clock) {
            return Err(Error::DuplicateRequest);
        }
        if routes.is_empty() || routes.iter().any(|e| *e < 0 || *e as u32 >= self.experts) {
            return Err(Error::InvalidRequest);
        }
        let group = self.group_of_layer[clock.layer];
        let mut unique = Vec::new();
        for &expert in routes {
            if !unique.contains(&expert) {
                unique.push(expert);
            }
        }
        if unique.len() > self.capacities[group] {
            return Err(Error::InvalidRequest);
        }
        let arena = &mut self.groups[group];
        let mut reads = Vec::new();
        let mut fills = Vec::new();
        let mut by_expert = HashMap::new();
        let mut missing = Vec::new();
        let mut evicted = Vec::new();
        let result = (|| {
            for &expert in &unique {
                let key = Key {
                    layer: LayerId(clock.layer as u32),
                    expert: ExpertId(expert as u32),
                };
                match arena.acquire(key, Phase(clock.sequence as u64)) {
                    Ok(read) => {
                        by_expert.insert(
                            expert,
                            Choice {
                                location: read.location(),
                                needs_conversion: false,
                            },
                        );
                        reads.push(read);
                    }
                    Err(ArenaError::Missing) => missing.push((expert, key)),
                    Err(e) => return Err(Error::Arena(e)),
                }
            }
            for (expert, key) in missing {
                let fill = arena.reserve(key, Phase(clock.sequence as u64))?;
                if let Some(victim) = fill.evicted() {
                    evicted.push(victim);
                }
                by_expert.insert(
                    expert,
                    Choice {
                        location: fill.location(),
                        needs_conversion: true,
                    },
                );
                fills.push(fill);
            }
            Ok(())
        })();
        if let Err(e) = result {
            for read in reads {
                arena.release_after_reader_completion(read)?;
            }
            for fill in fills {
                arena.rollback_unwritten_fill(fill)?;
            }
            return Err(e);
        }
        let choices = routes.iter().map(|e| by_expert[e]).collect();
        self.frames.push(Frame {
            clock,
            group,
            reads,
            fills,
            submitted: false,
        });
        Ok(Resolution { choices, evicted })
    }
    /// Call only after immutable descriptors and staging are published and
    /// the GPU acknowledgment permits this frame to convert and consume.
    pub fn submitted(&mut self, clock: Clock) -> Result<(), Error> {
        let frame = self
            .frames
            .iter_mut()
            .find(|f| f.clock == clock)
            .ok_or(Error::InvalidRequest)?;
        if frame.submitted {
            return Err(Error::DuplicateRequest);
        }
        frame.submitted = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn clock(sequence: i32, layer: usize) -> Clock {
        Clock {
            group: sequence,
            layer,
            sequence,
        }
    }
    fn directory(capacity: usize) -> SharedDirectory {
        SharedDirectory::from_parts(vec![capacity], vec![0, 0], 16).unwrap()
    }
    /// Device progress at `c` that retires no frame.
    fn observe(d: &mut SharedDirectory, c: Clock) -> Result<(), Error> {
        d.retire_progress(c, |_, _| panic!("no frame retires here")).map(|_| ())
    }
    /// A request at `c` whose conversions the device committed.
    fn finish(d: &mut SharedDirectory, c: Clock, routes: &[i32]) -> Resolution {
        observe(d, c).unwrap();
        let r = d.begin(c, routes).unwrap();
        d.submitted(c).unwrap();
        assert!(d.retire_step(c.sequence, |_, _| c.sequence).unwrap().is_empty());
        r
    }
    #[test]
    fn all_hits_are_pinned_before_a_route_order_miss_selects_victims() {
        let mut d = directory(2);
        let old = finish(&mut d, clock(1, 0), &[0, 1]);
        observe(&mut d, clock(2, 0)).unwrap();
        let r = d.begin(clock(2, 0), &[2, 0]).unwrap();
        assert_eq!(r.choices[1].location, old.choices[0].location);
        assert!(!r.choices[1].needs_conversion);
        assert_eq!(r.evicted[0].key.expert, ExpertId(1));
    }
    #[test]
    fn current_progress_and_read_completion_are_distinct() {
        let mut d = directory(2);
        let c = clock(1, 0);
        observe(&mut d, c).unwrap();
        d.begin(c, &[0]).unwrap();
        assert_eq!(observe(&mut d, clock(1, 1)), Err(Error::NotSubmitted));
        d.submitted(c).unwrap();
        observe(&mut d, c).unwrap();
        let key = Key {
            layer: LayerId(0),
            expert: ExpertId(0),
        };
        assert_eq!(d.groups[0].acquire(key, Phase(1)).unwrap_err(), ArenaError::ConversionPending);
        // A later point retires the frame with the device's commit of slot 0.
        assert!(d.retire_progress(clock(1, 1), |_, s| [1, 0][s]).unwrap().is_empty());
        assert!(d.groups[0].acquire(key, Phase(1)).is_ok());
    }
    #[test]
    fn cross_layer_eviction_reports_the_old_gpu_map_identity() {
        let mut d = directory(1);
        let old = finish(&mut d, clock(1, 0), &[3]);
        observe(&mut d, clock(2, 1)).unwrap();
        let r = d.begin(clock(2, 1), &[3]).unwrap();
        assert!(r.choices[0].needs_conversion);
        assert_eq!(r.evicted, [old.choices[0].location]);
        assert_eq!(r.choices[0].location.key.layer, LayerId(1));
        assert_ne!(
            r.choices[0].location.generation,
            old.choices[0].location.generation
        );
    }
    #[test]
    fn duplicate_choices_share_a_single_fill_and_invalid_union_mutates_nothing() {
        let mut d = directory(2);
        let c = clock(1, 0);
        observe(&mut d, c).unwrap();
        assert_eq!(d.begin(c, &[0, 1, 2]).unwrap_err(), Error::InvalidRequest);
        let r = d.begin(c, &[4, 4]).unwrap();
        assert_eq!(r.choices[0].location, r.choices[1].location);
        assert_eq!(d.frames[0].fills.len(), 1);
        assert_eq!(d.begin(c, &[5]).unwrap_err(), Error::DuplicateRequest);
    }
    #[test]
    fn gpu_stamps_protect_unreported_all_hit_use() {
        let mut d = directory(2);
        let old = finish(&mut d, clock(1, 0), &[0, 1]);
        observe(&mut d, clock(3, 1)).unwrap();
        d.sync_stamps(1, [2, 1].into_iter()).unwrap();
        let r = d.begin(clock(3, 1), &[2]).unwrap();
        assert_eq!(r.evicted[0], old.choices[1].location);
    }
    #[test]
    fn completion_does_not_advance_past_a_queued_future_request() {
        let mut d = directory(2);
        finish(&mut d, clock(1, 0), &[0]);
        let c = clock(2, 0);
        observe(&mut d, c).unwrap();
        d.begin(c, &[1]).unwrap();
        d.submitted(c).unwrap();
        assert!(d
            .retire_step(1, |_, _| panic!("future commit must not be read"))
            .unwrap()
            .is_empty());
        assert_eq!(d.frames.len(), 1);
        assert_eq!(observe(&mut d, clock(1, 1)), Err(Error::BackwardClock));
    }
    #[test]
    fn retirement_keeps_only_gpu_committed_fills_and_preserves_future_frames() {
        let mut d = directory(3);
        let c = clock(1, 0);
        observe(&mut d, c).unwrap();
        let r = d.begin(c, &[0, 1]).unwrap();
        d.submitted(c).unwrap();
        let commits = [1, 0, 0];
        let invalid = d.retire_progress(clock(1, 1), |_, s| commits[s]).unwrap();
        assert_eq!(invalid, [(c, r.choices[1].location)]);
        let next = d.begin(clock(1, 1), &[2]).unwrap();
        d.submitted(clock(1, 1)).unwrap();
        assert_eq!(next.choices[0].location.slot, r.choices[1].location.slot);
        d.retire_step(1, |_, _| 1).unwrap();
        observe(&mut d, clock(2, 0)).unwrap();
        let reused = d.begin(clock(2, 0), &[0, 1]).unwrap();
        assert!(!reused.choices[0].needs_conversion);
        assert!(reused.choices[1].needs_conversion);
    }
    #[test]
    fn stale_or_future_commits_cannot_validate_current_conversion() {
        for bad in [0, 1, 3] {
            let mut d = directory(1);
            let c = clock(2, 0);
            observe(&mut d, c).unwrap();
            let r = d.begin(c, &[0]).unwrap();
            d.submitted(c).unwrap();
            assert_eq!(
                d.retire_step(2, |_, _| bad).unwrap(),
                [(c, r.choices[0].location)]
            );
            observe(&mut d, clock(3, 0)).unwrap();
            assert!(d.begin(clock(3, 0), &[0]).unwrap().choices[0].needs_conversion);
        }
    }
    /// A layer-major group of steps 2 and 3 visits (2, layer 0), (3, layer 0),
    /// (2, layer 1), (3, layer 1). Step 3 at layer 0 finds what step 2 filled
    /// there, and the later step's earlier layer never reads as progress past
    /// the earlier step's later one.
    #[test]
    fn a_layer_major_group_orders_layers_before_steps() {
        let mut d = directory(4);
        let at = |sequence, layer| Clock {
            group: 2,
            layer,
            sequence,
        };
        observe(&mut d, at(2, 0)).unwrap();
        let first = d.begin(at(2, 0), &[5]).unwrap();
        d.submitted(at(2, 0)).unwrap();
        assert!(first.choices[0].needs_conversion);
        let slot = first.choices[0].location.slot;
        let commits = |_: usize, s: usize| if s == slot { 2 } else { 0 };
        assert!(d.retire_progress(at(3, 0), commits).unwrap().is_empty());
        let second = d.begin(at(3, 0), &[5]).unwrap();
        d.submitted(at(3, 0)).unwrap();
        assert!(!second.choices[0].needs_conversion);
        assert_eq!(second.choices[0].location, first.choices[0].location);
        assert!(at(3, 0) < at(2, 1));
        assert!(d.retire_progress(at(2, 1), commits).unwrap().is_empty());
        assert_eq!(observe(&mut d, at(3, 0)), Err(Error::BackwardClock));
        assert!(at(3, 1) < clock(4, 0));
    }
}
