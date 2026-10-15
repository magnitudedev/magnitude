//! Shared canonical-record ownership prototype; no model or device execution.
//!
//! A caller supplies a fully verified encoding group. Byte-size equality is
//! insufficient to establish representation/layout compatibility. The native
//! adapter must hold read leases until actual device completion, and publish a
//! fill only after its entire canonical record has been converted.
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LayerId(pub u32);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ExpertId(pub u32);
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Key {
    pub layer: LayerId,
    pub expert: ExpertId,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Phase(pub u64);

/// Immutable native operand identity. It is not permission to release a lease.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Location {
    pub slot: usize,
    pub generation: u64,
    pub key: Key,
}
#[derive(Debug)]
pub struct ReadLease {
    location: Location,
}
impl ReadLease {
    pub fn location(&self) -> Location {
        self.location
    }
}
#[derive(Debug)]
pub struct FillLease {
    location: Location,
    evicted: Option<Location>,
    evicted_phase: Phase,
}
impl FillLease {
    pub fn location(&self) -> Location {
        self.location
    }
    pub fn evicted(&self) -> Option<Location> {
        self.evicted
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    WrongGroup,
    OutOfBounds,
    Missing,
    ConversionPending,
    AlreadyPresent,
    AllSlotsLeased,
    StaleLocation,
    WrongState,
    CounterOverflow,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Content {
    Empty,
    Filling(Key),
    Ready(Key),
    /// Lent to a prompt's conveyor (spec 4.9): neither a victim nor a place.
    Lent,
}
#[derive(Debug)]
struct Slot {
    generation: u64,
    content: Content,
    phase: Phase,
    readers: u32,
}
pub struct Arena {
    /// Membership includes each layer's true expert count, independently of
    /// physical arena capacity. The native format key belongs to the adapter.
    layers: HashMap<LayerId, u32>,
    slots: Vec<Slot>,
    mapped: HashMap<Key, usize>,
}
impl Arena {
    pub fn new(capacity: usize, layers: &[(LayerId, u32)]) -> Result<Self, Error> {
        if capacity == 0 || layers.is_empty() || layers.iter().any(|(_, n)| *n == 0) {
            return Err(Error::OutOfBounds);
        }
        let members: HashMap<_, _> = layers.iter().copied().collect();
        if members.len() != layers.len() {
            return Err(Error::WrongGroup);
        }
        Ok(Self {
            layers: members,
            slots: (0..capacity)
                .map(|_| Slot {
                    generation: 0,
                    content: Content::Empty,
                    phase: Phase(0),
                    readers: 0,
                })
                .collect(),
            mapped: HashMap::new(),
        })
    }
    fn check_key(&self, key: Key) -> Result<(), Error> {
        match self.layers.get(&key.layer) {
            None => Err(Error::WrongGroup),
            Some(n) if key.expert.0 >= *n => Err(Error::OutOfBounds),
            Some(_) => Ok(()),
        }
    }
    fn slot_for(&self, location: Location) -> Result<&Slot, Error> {
        let slot = self.slots.get(location.slot).ok_or(Error::OutOfBounds)?;
        if slot.generation != location.generation
            || !matches!(slot.content, Content::Ready(k) | Content::Filling(k) if k == location.key)
        {
            return Err(Error::StaleLocation);
        }
        Ok(slot)
    }
    pub fn acquire(&mut self, key: Key, phase: Phase) -> Result<ReadLease, Error> {
        self.check_key(key)?;
        let index = *self.mapped.get(&key).ok_or(Error::Missing)?;
        let slot = &mut self.slots[index];
        if !matches!(slot.content, Content::Ready(k) if k == key) {
            return Err(Error::ConversionPending);
        }
        slot.readers = slot.readers.checked_add(1).ok_or(Error::CounterOverflow)?;
        slot.phase = slot.phase.max(phase);
        Ok(ReadLease {
            location: Location {
                slot: index,
                generation: slot.generation,
                key,
            },
        })
    }
    /// Reserve after pinning all current hits. Pending fills and live readers
    /// are never victims. Future queued work must acquire its own leases.
    pub fn reserve(&mut self, key: Key, phase: Phase) -> Result<FillLease, Error> {
        self.check_key(key)?;
        if self.mapped.contains_key(&key) {
            return Err(Error::AlreadyPresent);
        }
        let empty = self.slots.iter().position(|s| s.content == Content::Empty);
        let index = empty
            .or_else(|| {
                self.slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| matches!(s.content, Content::Ready(_)) && s.readers == 0)
                    .min_by_key(|(index, s)| {
                        let Content::Ready(owner) = s.content else {
                            unreachable!()
                        };
                        (s.phase.0, owner.layer.0, *index)
                    })
                    .map(|(index, _)| index)
            })
            .ok_or(Error::AllSlotsLeased)?;
        let generation = self.slots[index]
            .generation
            .checked_add(1)
            .ok_or(Error::CounterOverflow)?;
        let evicted_phase = self.slots[index].phase;
        let evicted = if let Content::Ready(victim) = self.slots[index].content {
            self.mapped.remove(&victim);
            Some(Location {
                slot: index,
                generation: self.slots[index].generation,
                key: victim,
            })
        } else {
            None
        };
        self.slots[index] = Slot {
            generation,
            content: Content::Filling(key),
            phase,
            readers: 0,
        };
        self.mapped.insert(key, index);
        Ok(FillLease {
            location: Location {
                slot: index,
                generation,
                key,
            },
            evicted,
            evicted_phase,
        })
    }
    /// Abort BEFORE any canonical write is submitted. Restore the unchanged
    /// victim data's ownership while keeping the new generation (no ABA).
    pub fn rollback_unwritten_fill(&mut self, fill: FillLease) -> Result<(), Error> {
        let location = fill.location;
        if self.slot_for(location)?.content != Content::Filling(location.key) {
            return Err(Error::WrongState);
        }
        self.mapped.remove(&location.key);
        let slot = &mut self.slots[location.slot];
        slot.phase = fill.evicted_phase;
        slot.content = if let Some(victim) = fill.evicted {
            self.mapped.insert(victim.key, location.slot);
            Content::Ready(victim.key)
        } else {
            Content::Empty
        };
        Ok(())
    }
    /// Reconcile GPU step stamps before choosing victims. Empty and pending
    /// records have no readable canonical contents to touch.
    pub fn observe_ready_phase(&mut self, slot: usize, phase: Phase) -> Result<(), Error> {
        let slot = self.slots.get_mut(slot).ok_or(Error::OutOfBounds)?;
        if matches!(slot.content, Content::Ready(_)) {
            slot.phase = slot.phase.max(phase);
        }
        Ok(())
    }
    /// This requires the adapter's conversion-completion evidence, not merely
    /// source-read completion. The returned lease protects the current reader.
    pub fn publish_after_conversion(&mut self, fill: FillLease) -> Result<ReadLease, Error> {
        let location = fill.location;
        if self.slot_for(location)?.content != Content::Filling(location.key) {
            return Err(Error::WrongState);
        }
        let slot = &mut self.slots[location.slot];
        slot.content = Content::Ready(location.key);
        slot.readers = 1;
        Ok(ReadLease { location })
    }
    pub fn fail_fill(&mut self, fill: FillLease) -> Result<(), Error> {
        let location = fill.location;
        if self.slot_for(location)?.content != Content::Filling(location.key) {
            return Err(Error::WrongState);
        }
        self.mapped.remove(&location.key);
        self.slots[location.slot].content = Content::Empty;
        Ok(())
    }
    /// Invoke only after the actual consumer fence has passed. A Rust lease is
    /// intentionally not Clone; immutable GPU descriptors can still be copied.
    pub fn release_after_reader_completion(&mut self, lease: ReadLease) -> Result<(), Error> {
        let location = lease.location;
        let slot = self.slot_for(location)?;
        if slot.content != Content::Ready(location.key) || slot.readers == 0 {
            return Err(Error::WrongState);
        }
        self.slots[location.slot].readers -= 1;
        Ok(())
    }
    /// Empty `key`'s slot: the expert it held, or None when the key is not
    /// held or its slot is read or filled now.
    pub fn discard(&mut self, key: Key) -> Result<Option<Location>, Error> {
        self.check_key(key)?;
        let Some(&index) = self.mapped.get(&key) else {
            return Ok(None);
        };
        let slot = &mut self.slots[index];
        if slot.readers > 0 || slot.content != Content::Ready(key) {
            return Ok(None);
        }
        let location = Location {
            slot: index,
            generation: slot.generation,
            key,
        };
        *slot = Slot {
            generation: slot
                .generation
                .checked_add(1)
                .ok_or(Error::CounterOverflow)?,
            content: Content::Empty,
            phase: Phase(0),
            readers: 0,
        };
        self.mapped.remove(&key);
        Ok(Some(location))
    }
    /// Lend `slots` to a prompt's conveyor (spec 4.9), evicting the experts
    /// they hold: all of them, or none while one is read or filled.
    pub fn lend(&mut self, slots: std::ops::Range<usize>) -> Result<Vec<Location>, Error> {
        let lent = self.slots.get(slots.clone()).ok_or(Error::OutOfBounds)?;
        if lent
            .iter()
            .any(|s| s.readers > 0 || matches!(s.content, Content::Filling(_) | Content::Lent))
        {
            return Err(Error::WrongState);
        }
        let mut evicted = Vec::new();
        for index in slots {
            let slot = &mut self.slots[index];
            let generation = slot.generation.checked_add(1).ok_or(Error::CounterOverflow)?;
            if let Content::Ready(key) = slot.content {
                self.mapped.remove(&key);
                evicted.push(Location {
                    slot: index,
                    generation: slot.generation,
                    key,
                });
            }
            *slot = Slot {
                generation,
                content: Content::Lent,
                phase: Phase(0),
                readers: 0,
            };
        }
        Ok(evicted)
    }
    /// Take lent `slots` back, empty.
    pub fn restore(&mut self, slots: std::ops::Range<usize>) -> Result<(), Error> {
        let lent = self.slots.get_mut(slots).ok_or(Error::OutOfBounds)?;
        if lent.iter().any(|s| s.content != Content::Lent) {
            return Err(Error::WrongState);
        }
        for slot in lent {
            slot.content = Content::Empty;
        }
        Ok(())
    }
    #[cfg(test)]
    pub fn validate_read(&self, location: Location) -> Result<(), Error> {
        let slot = self.slot_for(location)?;
        if slot.content != Content::Ready(location.key) || slot.readers == 0 {
            return Err(Error::WrongState);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unwritten_reservation_rollback_restores_victim_without_aba() {
        let mut a = Arena::new(1, &[(LayerId(0), 4)]).unwrap();
        let old = load(&mut a, key(0, 0), 1);
        let old_location = old.location();
        a.release_after_reader_completion(old).unwrap();
        let fill = a.reserve(key(0, 1), Phase(2)).unwrap();
        let abandoned = fill.location();
        assert_eq!(fill.evicted(), Some(old_location));
        a.rollback_unwritten_fill(fill).unwrap();
        let restored = a.acquire(key(0, 0), Phase(3)).unwrap();
        assert_eq!(restored.location().slot, old_location.slot);
        assert_ne!(restored.location().generation, old_location.generation);
        assert_eq!(a.validate_read(abandoned), Err(Error::StaleLocation));
        assert_eq!(a.acquire(key(0, 1), Phase(3)).unwrap_err(), Error::Missing);
        a.release_after_reader_completion(restored).unwrap();
    }
    fn key(layer: u32, expert: u32) -> Key {
        Key {
            layer: LayerId(layer),
            expert: ExpertId(expert),
        }
    }
    fn load(arena: &mut Arena, k: Key, phase: u64) -> ReadLease {
        let fill = arena.reserve(k, Phase(phase)).unwrap();
        arena.publish_after_conversion(fill).unwrap()
    }
    #[test]
    fn a_discarded_expert_leaves_an_empty_slot_but_a_read_one_stays() {
        let mut a = Arena::new(1, &[(LayerId(0), 4)]).unwrap();
        let read = load(&mut a, key(0, 0), 1);
        let kept = read.location();
        assert_eq!(a.discard(key(0, 0)), Ok(None), "a step reads it");
        a.release_after_reader_completion(read).unwrap();
        assert_eq!(a.discard(key(0, 0)), Ok(Some(kept)));
        assert_eq!(a.discard(key(0, 0)), Ok(None), "no longer held");
        assert_eq!(a.validate_read(kept), Err(Error::StaleLocation));
        // The emptied slot is a place: the next reservation evicts nothing.
        let fill = a.reserve(key(0, 1), Phase(2)).unwrap();
        assert_eq!(fill.evicted(), None);
        assert_eq!(fill.location().slot, kept.slot);
    }
    #[test]
    fn reader_prevents_eviction_until_its_fence() {
        let mut a = Arena::new(1, &[(LayerId(0), 4)]).unwrap();
        let read = load(&mut a, key(0, 0), 0);
        assert_eq!(
            a.reserve(key(0, 1), Phase(1)).unwrap_err(),
            Error::AllSlotsLeased
        );
        a.validate_read(read.location()).unwrap();
        a.release_after_reader_completion(read).unwrap();
        a.reserve(key(0, 1), Phase(1)).unwrap();
    }
    #[test]
    fn source_arrival_alone_does_not_make_weights_readable() {
        let mut a = Arena::new(1, &[(LayerId(0), 4)]).unwrap();
        let fill = a.reserve(key(0, 0), Phase(0)).unwrap();
        assert_eq!(
            a.acquire(key(0, 0), Phase(0)).unwrap_err(),
            Error::ConversionPending
        );
        assert_eq!(
            a.validate_read(fill.location()).unwrap_err(),
            Error::WrongState
        );
        assert_eq!(
            a.reserve(key(0, 1), Phase(0)).unwrap_err(),
            Error::AllSlotsLeased
        );
        let read = a.publish_after_conversion(fill).unwrap();
        a.validate_read(read.location()).unwrap();
    }
    #[test]
    fn stale_completion_cannot_publish_reused_slot() {
        let mut a = Arena::new(1, &[(LayerId(0), 4)]).unwrap();
        let first = a.reserve(key(0, 0), Phase(0)).unwrap();
        let stale = first.location();
        a.fail_fill(first).unwrap();
        let second = a.reserve(key(0, 1), Phase(1)).unwrap();
        assert_eq!(
            a.publish_after_conversion(FillLease {
                location: stale,
                evicted: None,
                evicted_phase: Phase(0)
            })
            .unwrap_err(),
            Error::StaleLocation
        );
        let read = a.publish_after_conversion(second).unwrap();
        a.validate_read(read.location()).unwrap();
    }
    #[test]
    fn stale_operand_descriptor_cannot_read_new_owner() {
        let mut a = Arena::new(1, &[(LayerId(0), 4)]).unwrap();
        let read = load(&mut a, key(0, 0), 0);
        let stale = read.location();
        a.release_after_reader_completion(read).unwrap();
        let current = load(&mut a, key(0, 1), 1);
        assert_eq!(a.validate_read(stale), Err(Error::StaleLocation));
        a.validate_read(current.location()).unwrap();
    }
    #[test]
    fn one_record_can_serve_several_readers() {
        let mut a = Arena::new(1, &[(LayerId(0), 4)]).unwrap();
        let first = load(&mut a, key(0, 0), 0);
        let second = a.acquire(key(0, 0), Phase(1)).unwrap();
        a.release_after_reader_completion(first).unwrap();
        assert_eq!(
            a.reserve(key(0, 1), Phase(2)).unwrap_err(),
            Error::AllSlotsLeased
        );
        a.validate_read(second.location()).unwrap();
        a.release_after_reader_completion(second).unwrap();
        a.reserve(key(0, 1), Phase(2)).unwrap();
    }
    #[test]
    fn layer_is_part_of_the_expert_key_and_eviction_clock() {
        let mut a = Arena::new(2, &[(LayerId(0), 4), (LayerId(1), 4)]).unwrap();
        let late = load(&mut a, key(1, 0), 4);
        let early = load(&mut a, key(0, 0), 4);
        a.release_after_reader_completion(late).unwrap();
        a.release_after_reader_completion(early).unwrap();
        let new = load(&mut a, key(1, 1), 5);
        assert_eq!(a.acquire(key(0, 0), Phase(5)).unwrap_err(), Error::Missing);
        a.acquire(key(1, 0), Phase(5)).unwrap();
        a.validate_read(new.location()).unwrap();
    }
    #[test]
    fn group_membership_and_expert_bounds_are_explicit() {
        let mut a = Arena::new(1, &[(LayerId(0), 4)]).unwrap();
        assert_eq!(
            a.reserve(key(1, 0), Phase(0)).unwrap_err(),
            Error::WrongGroup
        );
        assert_eq!(
            a.reserve(key(0, 4), Phase(0)).unwrap_err(),
            Error::OutOfBounds
        );
    }
    /// A lend evicts what its slots hold, waits for their readers, and its
    /// slots are neither victims nor places until they come back empty.
    #[test]
    fn lent_slots_leave_the_bank_until_restored() {
        let mut a = Arena::new(3, &[(LayerId(0), 8)]).unwrap();
        let held = load(&mut a, key(0, 0), 0);
        let free = load(&mut a, key(0, 1), 1);
        let free_location = free.location();
        a.release_after_reader_completion(free).unwrap();
        assert_eq!(a.lend(0..2).unwrap_err(), Error::WrongState, "slot 0 is read");
        a.validate_read(held.location()).unwrap();
        a.release_after_reader_completion(held).unwrap();
        let evicted = a.lend(1..3).unwrap();
        assert_eq!(evicted.iter().map(|l| l.key).collect::<Vec<_>>(), vec![key(0, 1)]);
        assert_eq!(a.validate_read(free_location), Err(Error::StaleLocation));
        assert_eq!(a.acquire(key(0, 1), Phase(3)).unwrap_err(), Error::Missing);
        // Only slot 0 remains: the next fill evicts expert 0, never a lent slot.
        let fill = a.reserve(key(0, 2), Phase(3)).unwrap();
        assert_eq!(fill.location().slot, 0);
        let read = a.publish_after_conversion(fill).unwrap();
        a.release_after_reader_completion(read).unwrap();
        assert_eq!(a.reserve(key(0, 3), Phase(4)).map(|f| f.location().slot), Ok(0));
        assert_eq!(a.lend(1..3).unwrap_err(), Error::WrongState, "already lent");
        a.restore(1..3).unwrap();
        assert_eq!(a.restore(1..3).unwrap_err(), Error::WrongState);
        assert_eq!(a.reserve(key(0, 4), Phase(5)).map(|f| f.location().slot), Ok(1));
    }
    #[test]
    fn an_older_reader_cannot_regress_the_use_clock() {
        let mut a = Arena::new(2, &[(LayerId(0), 4)]).unwrap();
        let first = load(&mut a, key(0, 0), 0);
        a.release_after_reader_completion(first).unwrap();
        let newer = a.acquire(key(0, 0), Phase(10)).unwrap();
        let older = a.acquire(key(0, 0), Phase(3)).unwrap();
        a.release_after_reader_completion(newer).unwrap();
        a.release_after_reader_completion(older).unwrap();
        let other = load(&mut a, key(0, 1), 5);
        a.release_after_reader_completion(other).unwrap();
        let new = load(&mut a, key(0, 2), 11);
        assert_eq!(a.acquire(key(0, 1), Phase(11)).unwrap_err(), Error::Missing);
        a.acquire(key(0, 0), Phase(11)).unwrap();
        a.validate_read(new.location()).unwrap();
    }
}
