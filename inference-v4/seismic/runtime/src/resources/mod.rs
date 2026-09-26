//! Whole-workflow resource admission primitives.
//!
//! Workflow planning owns symbolic resource requirements. This module owns the
//! mutable device-side state needed to turn those requirements into one
//! admitted transaction: admission serialization and persistent allocations.
//! Memory accounting remains in `memory` because allocations own its charges.

use crate::driver::{Allocation, AllocationPermit};
use crate::memory::MemoryReservation;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// The opaque ownership token produced by one successful whole-graph
/// admission. Execution retains it through terminal completion and borrows
/// its actual allocation access for observation. It cannot manufacture the
/// constituent reservation, leases, or access permits.
pub(crate) struct AdmittedResources {
    _reservation: MemoryReservation,
    preclaims: BTreeMap<(u64, usize, usize), PersistentGrowth>,
    // Each permit owns its backing Arc. One slot per physical allocation,
    // shared by every staged alias in the run.
    slots: BTreeMap<u64, Option<AllocationPermit>>,
    reached_live: u64,
    reached_budget: u64,
    reached_allocated: u64,
    reached_peak: u64,
}

impl AdmittedResources {
    /// Borrow the actual admission-owned access for terminal observation. No
    /// reacquisition is needed while that same exclusive access is still held.
    pub(crate) fn access(&self, allocation: &Arc<Allocation>) -> &AllocationPermit {
        let permit = self
            .slots
            .values()
            .filter_map(Option::as_ref)
            .find(|permit| permit.owns(allocation))
            .expect("completed observation reads backing outside its admitted resources");
        assert!(
            permit.owns(allocation),
            "physical identity names different backing"
        );
        permit
    }
    pub(crate) fn allocation(&self, identity: u64) -> &Arc<Allocation> {
        self.slots
            .get(&identity)
            .and_then(Option::as_ref)
            .expect("staged allocation has no admitted physical backing")
            .allocation()
    }
    pub(crate) fn slot_for(&self, allocation: &Arc<Allocation>) -> u64 {
        *self
            .slots
            .iter()
            .find(|(_, permit)| {
                permit
                    .as_ref()
                    .is_some_and(|permit| permit.owns(allocation))
            })
            .map(|(slot, _)| slot)
            .expect("backing has no run-owned physical slot")
    }
    pub(crate) fn retain_preclaims(
        &mut self,
        claims: BTreeMap<(u64, usize, usize), PersistentGrowth>,
    ) {
        assert!(self.preclaims.is_empty(), "portfolio keys already claimed");
        self.preclaims = claims;
    }
    pub(crate) fn preclaimed_binding(
        &self,
        key: (u64, usize, usize),
    ) -> Option<&PersistentBinding> {
        self.preclaims
            .get(&key)
            .expect("selected persistent key was not inventoried")
            .old()
    }
    pub(crate) fn install_preclaimed(
        &mut self,
        key: (u64, usize, usize),
        binding: PersistentBinding,
    ) {
        self.preclaims
            .get_mut(&key)
            .expect("selected persistent key was not inventoried")
            .install_reached(binding);
    }
    pub(crate) fn set_reached_budget(&mut self, bytes: u64) {
        self.reached_budget = bytes;
    }
    pub(crate) fn reached_allocated(&self) -> u64 {
        self.reached_allocated
    }
    pub(crate) fn check_reached_capacity(
        &self,
        bytes: u64,
    ) -> Result<(), seismic_compiler::errors::ExecutionError> {
        let available = self.reached_budget.saturating_sub(self.reached_live);
        if bytes > available {
            return Err(
                seismic_compiler::errors::ExecutionError::AllocationCapacity {
                    required: bytes.into(),
                    available,
                },
            );
        }
        Ok(())
    }
    pub(crate) fn declare_private(&mut self, slot: u64) {
        self.slots.entry(slot).or_insert(None);
    }
    pub(crate) fn private_backing(&self, slot: u64) -> Option<&Arc<Allocation>> {
        self.slots
            .get(&slot)
            .expect("private slot was never declared")
            .as_ref()
            .map(AllocationPermit::allocation)
    }
    pub(crate) fn retire_private(&mut self, slot: u64) {
        if let Some(old) = self
            .slots
            .get_mut(&slot)
            .expect("private slot was never declared")
            .take()
        {
            self.reached_live -= old.allocation().bytes();
        }
    }
    pub(crate) fn install_private(&mut self, slot: u64, permit: AllocationPermit) {
        let destination = self
            .slots
            .get_mut(&slot)
            .expect("private slot was never declared");
        assert!(
            destination.is_none(),
            "private slot replacement did not retire its old backing"
        );
        let bytes = permit.allocation().bytes();
        self.reached_live = self
            .reached_live
            .checked_add(bytes)
            .expect("admitted reached bytes overflow");
        self.reached_allocated = self.reached_allocated.saturating_add(bytes);
        self.reached_peak = self.reached_peak.max(self.reached_live);
        *destination = Some(permit);
    }
    pub(crate) fn new(reservation: MemoryReservation, access: Vec<AllocationPermit>) -> Self {
        let mut slots = BTreeMap::new();
        for permit in access {
            let identity = permit.allocation().identity();
            assert!(
                slots.insert(identity, Some(permit)).is_none(),
                "physical allocation admitted twice"
            );
        }
        Self {
            _reservation: reservation,
            preclaims: BTreeMap::new(),
            slots,
            reached_live: 0,
            reached_budget: u64::MAX,
            reached_allocated: 0,
            reached_peak: 0,
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
    Ready { binding: PersistentBinding },
    Growing,
}

#[derive(Clone)]
pub(crate) enum PersistentAvailability {
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

    /// A portfolio claim owns the key independently of whether this run ever
    /// selects it. No backing or capacity is acquired by this operation.
    pub(crate) fn preclaim_availability(&self, key: (usize, usize)) -> PersistentAvailability {
        Self::preclaim_availability_locked(&self.slots(), key)
    }

    fn preclaim_availability_locked(
        slots: &HashMap<(usize, usize), PersistentSlot>,
        key: (usize, usize),
    ) -> PersistentAvailability {
        match slots.get(&key) {
            None => PersistentAvailability::Grow { old: None },
            Some(PersistentSlot::Ready { binding }) => PersistentAvailability::Grow {
                old: Some(binding.clone()),
            },
            Some(_) => PersistentAvailability::Wait,
        }
    }

    pub(crate) fn wait_until_preclaimable(&self, key: (usize, usize)) {
        let mut slots = self.slots();
        while matches!(
            Self::preclaim_availability_locked(&slots, key),
            PersistentAvailability::Wait
        ) {
            slots = self
                .changed
                .wait(slots)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    pub(crate) fn claim_growth(self: &Arc<Self>, key: (usize, usize)) -> PersistentGrowth {
        let mut slots = self.slots();
        let old = match slots.get(&key) {
            None => None,
            Some(PersistentSlot::Ready { binding }) => Some(binding.clone()),
            Some(PersistentSlot::Growing) => {
                panic!("persistent growth became unavailable while admission was serialized")
            }
        };
        slots.insert(key, PersistentSlot::Growing);
        PersistentGrowth {
            table: self.clone(),
            key,
            old,
        }
    }
}

/// Exclusive ownership of a portfolio key through terminal completion. An
/// unused claim restores its original backing; a reached replacement becomes
/// the next persistent backing when the run releases its claim.
pub(crate) struct PersistentGrowth {
    table: Arc<PersistentTable>,
    key: (usize, usize),
    old: Option<PersistentBinding>,
}

impl PersistentGrowth {
    pub(crate) fn old(&self) -> Option<&PersistentBinding> {
        self.old.as_ref()
    }

    /// A run that preclaimed a portfolio key may replace its physical backing
    /// after reaching the corresponding demand. The key remains exclusively
    /// claimed through terminal completion; dropping the claim publishes this
    /// latest backing, or restores absence when the key was never selected.
    pub(crate) fn install_reached(&mut self, binding: PersistentBinding) {
        assert!(
            matches!(
                self.table.slots().get(&self.key),
                Some(PersistentSlot::Growing)
            ),
            "persistent key lost its run claim"
        );
        self.old = Some(binding);
    }
}

impl Drop for PersistentGrowth {
    fn drop(&mut self) {
        let mut slots = self.table.slots();
        assert!(
            matches!(slots.get(&self.key), Some(PersistentSlot::Growing)),
            "persistent growth marker disappeared before rollback"
        );
        match self.old.take() {
            Some(binding) => {
                slots.insert(self.key, PersistentSlot::Ready { binding });
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
    #[test]
    fn unused_portfolio_claim_preserves_absence_and_never_acquires_capacity() {
        use super::*;
        let table = Arc::new(PersistentTable::new());
        assert!(matches!(
            table.preclaim_availability((3, 7)),
            PersistentAvailability::Grow { old: None }
        ));
        let claim = table.claim_growth((3, 7));
        assert!(claim.old().is_none());
        assert!(matches!(
            table.preclaim_availability((3, 7)),
            PersistentAvailability::Wait
        ));
        // The key claim has no memory-domain handle and cannot allocate or
        // reserve bytes. An unvisited branch simply drops its unused claim.
        drop(claim);
        assert!(table.slots().is_empty());
    }
}
