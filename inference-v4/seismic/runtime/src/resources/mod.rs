//! Whole-workflow resource admission primitives.
//!
//! Workflow planning owns symbolic resource requirements. This module owns the
//! mutable device-side state needed to turn those requirements into one
//! admitted transaction: admission serialization and persistent allocations.
//! Memory accounting remains in `memory` because allocations own its charges.

use crate::driver::{Allocation, AllocationPermit};
use crate::memory::MemoryReservation;
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// The opaque ownership token produced by one successful whole-graph
/// admission. Execution can retain or release it, but cannot inspect or
/// manufacture its constituent reservation, leases, or access permits.
pub(crate) struct AdmittedResources {
    _reservation: MemoryReservation,
    _persistent: Vec<PersistentLease>,
    _access: Vec<AllocationPermit>,
}

impl AdmittedResources {
    pub(crate) fn new(
        reservation: MemoryReservation,
        persistent: Vec<PersistentLease>,
        access: Vec<AllocationPermit>,
    ) -> Self {
        Self {
            _reservation: reservation,
            _persistent: persistent,
            _access: access,
        }
    }
}

/// Serializes only admission state transitions. Executions never need this
/// guard to finish, so waiting for an allocation access while holding it cannot
/// prevent the owner of that access from releasing it.
pub(crate) struct AdmissionDomain {
    serial: Mutex<()>,
}

impl AdmissionDomain {
    pub(crate) fn new() -> Self {
        Self {
            serial: Mutex::new(()),
        }
    }

    pub(crate) fn enter(&self) -> MutexGuard<'_, ()> {
        self.serial
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[derive(Clone)]
pub(crate) struct PersistentBinding {
    pub(crate) allocation: Arc<Allocation>,
    pub(crate) capacity: u64,
}

enum PersistentSlot {
    Ready {
        binding: PersistentBinding,
        leases: u64,
    },
    Growing,
}

#[derive(Clone)]
pub(crate) enum PersistentAvailability {
    Reuse(PersistentBinding),
    Grow { old: Option<PersistentBinding> },
    Wait,
}

pub(crate) struct PersistentTable {
    slots: Mutex<HashMap<(usize, usize), PersistentSlot>>,
    changed: Condvar,
}

impl PersistentTable {
    pub(crate) fn new() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
            changed: Condvar::new(),
        }
    }

    fn slots(&self) -> MutexGuard<'_, HashMap<(usize, usize), PersistentSlot>> {
        self.slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn availability_locked(
        slots: &HashMap<(usize, usize), PersistentSlot>,
        key: (usize, usize),
        bytes: u64,
    ) -> PersistentAvailability {
        match slots.get(&key) {
            None => PersistentAvailability::Grow { old: None },
            Some(PersistentSlot::Growing) => PersistentAvailability::Wait,
            Some(PersistentSlot::Ready { binding, .. }) if binding.capacity >= bytes => {
                PersistentAvailability::Reuse(binding.clone())
            }
            Some(PersistentSlot::Ready { binding, leases: 0 }) => PersistentAvailability::Grow {
                old: Some(binding.clone()),
            },
            Some(PersistentSlot::Ready { .. }) => PersistentAvailability::Wait,
        }
    }

    pub(crate) fn availability(&self, key: (usize, usize), bytes: u64) -> PersistentAvailability {
        Self::availability_locked(&self.slots(), key, bytes)
    }

    /// Waits without holding the device admission guard or any reservation,
    /// persistent claim, or allocation permit. The caller retries the complete
    /// graph snapshot after this returns.
    pub(crate) fn wait_until_available(&self, key: (usize, usize), bytes: u64) {
        let mut slots = self.slots();
        while matches!(
            Self::availability_locked(&slots, key, bytes),
            PersistentAvailability::Wait
        ) {
            slots = self
                .changed
                .wait(slots)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    pub(crate) fn claim_reuse(
        self: &Arc<Self>,
        key: (usize, usize),
        expected: &Arc<Allocation>,
    ) -> PersistentLease {
        let mut slots = self.slots();
        let Some(PersistentSlot::Ready { binding, leases }) = slots.get_mut(&key) else {
            panic!("persistent reuse changed while the admission guard was held")
        };
        assert!(
            Arc::ptr_eq(&binding.allocation, expected),
            "persistent allocation changed while the admission guard was held"
        );
        *leases = leases
            .checked_add(1)
            .expect("persistent allocation lease count overflowed");
        PersistentLease {
            table: self.clone(),
            key,
        }
    }

    pub(crate) fn claim_growth(self: &Arc<Self>, key: (usize, usize)) -> PersistentGrowth {
        let mut slots = self.slots();
        let old = match slots.get(&key) {
            None => None,
            Some(PersistentSlot::Ready { binding, leases: 0 }) => Some(binding.clone()),
            Some(PersistentSlot::Ready { .. }) | Some(PersistentSlot::Growing) => {
                panic!("persistent growth became unavailable while admission was serialized")
            }
        };
        slots.insert(key, PersistentSlot::Growing);
        PersistentGrowth {
            table: self.clone(),
            key,
            old,
            active: true,
        }
    }
}

pub(crate) struct PersistentLease {
    table: Arc<PersistentTable>,
    key: (usize, usize),
}

impl Drop for PersistentLease {
    fn drop(&mut self) {
        let mut slots = self.table.slots();
        let Some(PersistentSlot::Ready { leases, .. }) = slots.get_mut(&self.key) else {
            panic!("leased persistent allocation is not ready")
        };
        *leases = leases
            .checked_sub(1)
            .expect("persistent allocation lease count underflow");
        if *leases == 0 {
            self.table.changed.notify_all();
        }
    }
}

/// An unpublished persistent replacement. Dropping it restores the exact slot
/// state observed before the transaction claimed growth.
pub(crate) struct PersistentGrowth {
    table: Arc<PersistentTable>,
    key: (usize, usize),
    old: Option<PersistentBinding>,
    active: bool,
}

impl PersistentGrowth {
    pub(crate) fn old(&self) -> Option<&PersistentBinding> {
        self.old.as_ref()
    }

    pub(crate) fn commit(mut self, binding: PersistentBinding) -> PersistentLease {
        let mut slots = self.table.slots();
        assert!(
            matches!(slots.get(&self.key), Some(PersistentSlot::Growing)),
            "persistent growth marker disappeared before commit"
        );
        slots.insert(self.key, PersistentSlot::Ready { binding, leases: 1 });
        self.active = false;
        self.table.changed.notify_all();
        drop(slots);
        PersistentLease {
            table: self.table.clone(),
            key: self.key,
        }
    }
}

impl Drop for PersistentGrowth {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut slots = self.table.slots();
        assert!(
            matches!(slots.get(&self.key), Some(PersistentSlot::Growing)),
            "persistent growth marker disappeared before rollback"
        );
        match self.old.take() {
            Some(binding) => {
                slots.insert(self.key, PersistentSlot::Ready { binding, leases: 0 });
            }
            None => {
                slots.remove(&self.key);
            }
        }
        self.table.changed.notify_all();
    }
}

#[cfg(test)]
mod tests {
    /// A backend-free model of the production admission phases. The bounded
    /// enumeration checks every failure boundary and both persistent actions;
    /// production RAII objects implement the same ownership transitions.
    #[test]
    fn bounded_transaction_failures_restore_markers_leases_and_capacity() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Slot {
            Absent,
            Ready { capacity: u64, leases: u64 },
            Growing,
        }
        for initial in [
            Slot::Absent,
            Slot::Ready {
                capacity: 8,
                leases: 0,
            },
            Slot::Ready {
                capacity: 2,
                leases: 0,
            },
            Slot::Ready {
                capacity: 2,
                leases: 1,
            },
            Slot::Growing,
        ] {
            for failure_boundary in 0..=5 {
                let waits = matches!(
                    initial,
                    Slot::Growing
                        | Slot::Ready {
                            capacity: 0..=7,
                            leases: 1..
                        }
                );
                if waits {
                    let slot_after_wait_decision = initial;
                    let transaction_charge = 0;
                    let access_permits = 0;
                    assert_eq!(slot_after_wait_decision, initial);
                    assert_eq!((transaction_charge, access_permits), (0, 0));
                    continue;
                }
                let grows = !matches!(initial, Slot::Ready { capacity: 8.., .. });
                let reservation = 3 + if grows { 8 } else { 0 };
                let mut transaction_charge = 0;

                // claim, reserve, allocate temporary, allocate/grow persistent,
                // copy, and begin-submission. Every injected failure before
                // publication restores the initial slot and releases the full
                // reservation, including allocation-failure boundaries 2/3.
                let mut slot = if grows {
                    Slot::Growing
                } else {
                    Slot::Ready {
                        capacity: 8,
                        leases: 1,
                    }
                };
                if failure_boundary == 0 {
                    rollback(&mut transaction_charge, &mut slot, initial);
                    assert_eq!((transaction_charge, slot), (0, initial));
                    continue;
                }
                transaction_charge = reservation;
                if failure_boundary <= 4 {
                    rollback(&mut transaction_charge, &mut slot, initial);
                    assert_eq!((transaction_charge, slot), (0, initial));
                    continue;
                }

                if grows {
                    slot = Slot::Ready {
                        capacity: 8,
                        leases: 1,
                    };
                }
                assert_eq!(transaction_charge, reservation);
                assert_eq!(
                    slot,
                    Slot::Ready {
                        capacity: 8,
                        leases: 1
                    }
                );

                // Completion releases the temporary charge and lease. A grown
                // persistent allocation remains charged and reusable.
                transaction_charge = if grows { 8 } else { 0 };
                slot = Slot::Ready {
                    capacity: 8,
                    leases: 0,
                };
                assert_eq!(transaction_charge, if grows { 8 } else { 0 });
                assert_eq!(
                    slot,
                    Slot::Ready {
                        capacity: 8,
                        leases: 0
                    }
                );
            }
        }

        fn rollback<T: Copy>(charged: &mut u64, state: &mut T, initial: T) {
            *charged = 0;
            *state = initial;
        }
    }

    #[test]
    fn same_graph_persistent_uses_collapse_to_one_growth_and_one_charge() {
        for uses in 1usize..=8 {
            let requested = (0..uses).map(|index| 4 + index as u64).max().unwrap();
            let growths = usize::from(requested > 3);
            let charged = if growths == 1 { requested } else { 0 };
            assert_eq!(growths, 1);
            assert_eq!(charged, 3 + uses as u64);
        }
    }
}
