//! Pure placement snapshots and relocation publication. This module does not
//! allocate or copy device memory: the executor must complete every copy and
//! hold both backings before it marks a transaction ready to commit.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LogicalId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AllocationId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Generation(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhysicalSlot {
    pub allocation: AllocationId,
    pub index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacementError {
    UnknownAllocation(AllocationId),
    SlotOutOfBounds(PhysicalSlot),
    DuplicateSlot(PhysicalSlot),
    DestinationIsSource(AllocationId),
    InsufficientSlots {
        required: usize,
        available: usize,
    },
    GenerationExhausted,
    StaleGeneration {
        expected: Generation,
        actual: Generation,
    },
    IncompleteCopies {
        remaining: usize,
    },
    CopyOutOfBounds(usize),
    UnknownResource(LogicalId),
}

/// A published, immutable mapping. The allocation extents describe physical
/// slots, while resource identities survive a change of backing or slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    generation: Generation,
    allocations: BTreeMap<AllocationId, usize>,
    resources: BTreeMap<LogicalId, PhysicalSlot>,
}

impl Placement {
    pub fn new(
        generation: Generation,
        allocations: BTreeMap<AllocationId, usize>,
        resources: BTreeMap<LogicalId, PhysicalSlot>,
    ) -> Result<Self, PlacementError> {
        let placement = Self {
            generation,
            allocations,
            resources,
        };
        placement.validate()?;
        Ok(placement)
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn resources(&self) -> &BTreeMap<LogicalId, PhysicalSlot> {
        &self.resources
    }

    pub fn resolve(&self, id: LogicalId) -> Option<PhysicalSlot> {
        self.resources.get(&id).copied()
    }

    pub fn allocation_capacity(&self, id: AllocationId) -> Option<usize> {
        self.allocations.get(&id).copied()
    }

    pub fn highest_occupied_slot(&self, allocation: AllocationId) -> Option<usize> {
        self.resources
            .values()
            .filter(|slot| slot.allocation == allocation)
            .map(|slot| slot.index)
            .max()
    }

    /// Publish a changed logical ownership map without changing the backing.
    /// Every change advances the generation so a pending relocation based on
    /// the old ownership cannot publish over it.
    pub fn with_resources(
        &self,
        resources: BTreeMap<LogicalId, PhysicalSlot>,
    ) -> Result<Self, PlacementError> {
        let generation = Generation(
            self.generation
                .0
                .checked_add(1)
                .ok_or(PlacementError::GenerationExhausted)?,
        );
        Self::new(generation, self.allocations.clone(), resources)
    }

    pub fn with_capacity(
        &self,
        allocation: AllocationId,
        capacity: usize,
    ) -> Result<Self, PlacementError> {
        let generation = Generation(
            self.generation
                .0
                .checked_add(1)
                .ok_or(PlacementError::GenerationExhausted)?,
        );
        let mut allocations = self.allocations.clone();
        if !allocations.contains_key(&allocation) {
            return Err(PlacementError::UnknownAllocation(allocation));
        }
        allocations.insert(allocation, capacity);
        Self::new(generation, allocations, self.resources.clone())
    }

    fn validate(&self) -> Result<(), PlacementError> {
        let mut occupied = BTreeSet::new();
        for slot in self.resources.values().copied() {
            let capacity = self
                .allocations
                .get(&slot.allocation)
                .ok_or(PlacementError::UnknownAllocation(slot.allocation))?;
            if slot.index >= *capacity {
                return Err(PlacementError::SlotOutOfBounds(slot));
            }
            if !occupied.insert(slot) {
                return Err(PlacementError::DuplicateSlot(slot));
            }
        }
        Ok(())
    }

    /// Plan a dense prefix in a distinct backing. Requiring a distinct
    /// allocation keeps every source slot readable until atomic publication;
    /// copying through holes of the published backing would not do so.
    pub fn plan_dense_prefix(
        &self,
        destination: AllocationId,
        capacity_slots: usize,
    ) -> Result<RelocationPlan, PlacementError> {
        if self.allocations.contains_key(&destination) {
            return Err(PlacementError::DestinationIsSource(destination));
        }
        if capacity_slots < self.resources.len() {
            return Err(PlacementError::InsufficientSlots {
                required: self.resources.len(),
                available: capacity_slots,
            });
        }
        let next = Generation(
            self.generation
                .0
                .checked_add(1)
                .ok_or(PlacementError::GenerationExhausted)?,
        );
        let resources = self
            .resources
            .keys()
            .copied()
            .enumerate()
            .map(|(index, id)| {
                (
                    id,
                    PhysicalSlot {
                        allocation: destination,
                        index,
                    },
                )
            })
            .collect();
        let destination = Placement::new(
            next,
            BTreeMap::from([(destination, capacity_slots)]),
            resources,
        )?;
        let copies = self
            .resources
            .iter()
            .map(|(&resource, &from)| CopyOp {
                resource,
                from,
                to: destination.resources[&resource],
            })
            .collect();
        Ok(RelocationPlan {
            source: self.clone(),
            destination,
            copies,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyOp {
    pub resource: LogicalId,
    pub from: PhysicalSlot,
    pub to: PhysicalSlot,
}

/// The source is a complete snapshot, so publication cannot silently accept
/// a plan made before another relocation or ownership change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelocationPlan {
    source: Placement,
    destination: Placement,
    copies: Vec<CopyOp>,
}

impl RelocationPlan {
    pub fn source(&self) -> &Placement {
        &self.source
    }

    pub fn destination(&self) -> &Placement {
        &self.destination
    }

    pub fn copies(&self) -> &[CopyOp] {
        &self.copies
    }

    pub fn begin(self) -> RelocationTxn {
        RelocationTxn {
            completed: vec![false; self.copies.len()],
            plan: self,
        }
    }
}

/// Copy completion is explicit. Aborting consumes the transaction without
/// touching the published placement; backend cleanup belongs to its executor.
pub struct RelocationTxn {
    plan: RelocationPlan,
    completed: Vec<bool>,
}

impl RelocationTxn {
    pub fn copies(&self) -> &[CopyOp] {
        self.plan.copies()
    }

    pub fn mark_copied(&mut self, index: usize) -> Result<(), PlacementError> {
        let completed = self
            .completed
            .get_mut(index)
            .ok_or(PlacementError::CopyOutOfBounds(index))?;
        *completed = true;
        Ok(())
    }

    pub fn commit(self, published: &mut Placement) -> Result<Placement, PlacementError> {
        if *published != self.plan.source {
            return Err(PlacementError::StaleGeneration {
                expected: self.plan.source.generation,
                actual: published.generation,
            });
        }
        let remaining = self.completed.iter().filter(|&&done| !done).count();
        if remaining != 0 {
            return Err(PlacementError::IncompleteCopies { remaining });
        }
        let retired = std::mem::replace(published, self.plan.destination);
        Ok(retired)
    }

    pub fn abort(self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placement(slots: &[(u64, usize)], capacity: usize) -> Placement {
        Placement::new(
            Generation(3),
            BTreeMap::from([(AllocationId(1), capacity)]),
            slots
                .iter()
                .map(|&(id, index)| {
                    (
                        LogicalId(id),
                        PhysicalSlot {
                            allocation: AllocationId(1),
                            index,
                        },
                    )
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn dense_placement_has_a_contiguous_prefix() {
        let published = placement(&[(1, 0), (2, 1), (3, 2)], 4);
        assert_eq!(published.highest_occupied_slot(AllocationId(1)), Some(2));
        assert_eq!(published.resources().len(), 3);
    }

    #[test]
    fn fragmented_placement_tracks_the_live_tail() {
        let published = placement(&[(1, 0), (2, 6), (3, 2)], 8);
        assert_eq!(published.highest_occupied_slot(AllocationId(1)), Some(6));
        assert_eq!(published.resolve(LogicalId(2)).unwrap().index, 6);
        assert_eq!(
            Placement::new(
                Generation(0),
                BTreeMap::from([(AllocationId(1), 2)]),
                BTreeMap::from([
                    (
                        LogicalId(1),
                        PhysicalSlot {
                            allocation: AllocationId(1),
                            index: 0
                        }
                    ),
                    (
                        LogicalId(2),
                        PhysicalSlot {
                            allocation: AllocationId(1),
                            index: 0
                        }
                    ),
                ]),
            ),
            Err(PlacementError::DuplicateSlot(PhysicalSlot {
                allocation: AllocationId(1),
                index: 0
            }))
        );
    }

    #[test]
    fn relocation_publishes_a_dense_prefix_after_every_copy() {
        let mut published = placement(&[(1, 0), (2, 6), (3, 2)], 8);
        let snapshot = published.clone();
        let plan = published.plan_dense_prefix(AllocationId(2), 3).unwrap();
        assert_eq!(plan.copies().len(), 3);
        assert_eq!(published, snapshot);
        let mut txn = plan.begin();
        for index in 0..txn.copies().len() {
            txn.mark_copied(index).unwrap();
        }
        let retired = txn.commit(&mut published).unwrap();
        assert_eq!(retired, snapshot);
        assert_eq!(published.generation(), Generation(4));
        assert_eq!(published.highest_occupied_slot(AllocationId(2)), Some(2));
        assert_eq!(published.resolve(LogicalId(2)).unwrap().index, 1);
    }

    #[test]
    fn failed_relocation_preserves_the_source() {
        let mut published = placement(&[(1, 0), (2, 6)], 8);
        let snapshot = published.clone();
        let mut txn = published
            .plan_dense_prefix(AllocationId(2), 2)
            .unwrap()
            .begin();
        txn.mark_copied(0).unwrap();
        assert!(matches!(
            txn.commit(&mut published),
            Err(PlacementError::IncompleteCopies { remaining: 1 })
        ));
        assert_eq!(published, snapshot);
        published
            .plan_dense_prefix(AllocationId(2), 2)
            .unwrap()
            .begin()
            .abort();
        assert_eq!(published, snapshot);
    }

    #[test]
    fn stale_generation_cannot_publish_over_a_new_placement() {
        let mut published = placement(&[(1, 0), (2, 6)], 8);
        let first = published.plan_dense_prefix(AllocationId(2), 2).unwrap();
        let stale = published.plan_dense_prefix(AllocationId(3), 2).unwrap();
        let mut first = first.begin();
        for index in 0..first.copies().len() {
            first.mark_copied(index).unwrap();
        }
        first.commit(&mut published).unwrap();
        let current = published.clone();
        let mut stale = stale.begin();
        for index in 0..stale.copies().len() {
            stale.mark_copied(index).unwrap();
        }
        assert_eq!(
            stale.commit(&mut published),
            Err(PlacementError::StaleGeneration {
                expected: Generation(3),
                actual: Generation(4)
            })
        );
        assert_eq!(published, current);
    }
}
