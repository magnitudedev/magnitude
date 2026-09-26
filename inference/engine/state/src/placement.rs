//! Pure recurrent-bank placement. A bank is a slab item, addressed by its
//! slot in the store's slab list. Ownership changes advance the generation;
//! the state store publishes a relocation only after its copies complete.

use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LogicalId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Generation(pub u64);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacementError {
    SlotOutOfBounds(usize),
    DuplicateSlot(usize),
    GenerationExhausted,
}

/// The published logical-bank-to-slot map. The slab list owns storage; this
/// map says only which backed slots are claimed by live banks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    generation: Generation,
    capacity: usize,
    resources: BTreeMap<LogicalId, usize>,
}

impl Placement {
    pub fn new(
        generation: Generation,
        capacity: usize,
        resources: BTreeMap<LogicalId, usize>,
    ) -> Result<Self, PlacementError> {
        let placement = Self {
            generation,
            capacity,
            resources,
        };
        placement.validate()?;
        Ok(placement)
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn resources(&self) -> &BTreeMap<LogicalId, usize> {
        &self.resources
    }

    pub fn resolve(&self, id: LogicalId) -> Option<usize> {
        self.resources.get(&id).copied()
    }

    pub fn with_resources(
        &self,
        resources: BTreeMap<LogicalId, usize>,
    ) -> Result<Self, PlacementError> {
        Self::new(self.next_generation()?, self.capacity, resources)
    }

    pub fn with_capacity(&self, capacity: usize) -> Result<Self, PlacementError> {
        Self::new(self.next_generation()?, capacity, self.resources.clone())
    }

    fn next_generation(&self) -> Result<Generation, PlacementError> {
        self.generation
            .0
            .checked_add(1)
            .map(Generation)
            .ok_or(PlacementError::GenerationExhausted)
    }

    fn validate(&self) -> Result<(), PlacementError> {
        let mut occupied = BTreeSet::new();
        for &slot in self.resources.values() {
            if slot >= self.capacity {
                return Err(PlacementError::SlotOutOfBounds(slot));
            }
            if !occupied.insert(slot) {
                return Err(PlacementError::DuplicateSlot(slot));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_rejects_duplicate_and_out_of_bounds_slots() {
        assert_eq!(
            Placement::new(Generation(0), 2, BTreeMap::from([(LogicalId(1), 2)])),
            Err(PlacementError::SlotOutOfBounds(2))
        );
        assert_eq!(
            Placement::new(
                Generation(0),
                2,
                BTreeMap::from([(LogicalId(1), 0), (LogicalId(2), 0)]),
            ),
            Err(PlacementError::DuplicateSlot(0))
        );
    }

    #[test]
    fn ownership_and_capacity_changes_advance_generation() {
        let original = Placement::new(
            Generation(3),
            4,
            BTreeMap::from([(LogicalId(1), 0), (LogicalId(2), 3)]),
        )
        .unwrap();
        let changed = original
            .with_resources(BTreeMap::from([(LogicalId(1), 0), (LogicalId(2), 2)]))
            .unwrap();
        assert_eq!(changed.generation(), Generation(4));
        assert_eq!(changed.resolve(LogicalId(2)), Some(2));
        assert_eq!(original.resolve(LogicalId(2)), Some(3));
        assert_eq!(
            changed.with_capacity(2),
            Err(PlacementError::SlotOutOfBounds(2))
        );
        assert_eq!(
            changed.with_capacity(3).unwrap().generation(),
            Generation(5)
        );
    }
}
