//! Atomic memory reservation and allocation lifetime accounting.
//! No backend or submission service is needed to exercise this state machine.
//!
//! One [`PoolLedger`] exists per physical memory pool in this process. Every
//! opened device whose allocations reside in that pool charges it through
//! its own [`MemoryDomain`], which adds the device's enforceable limit. A
//! charge therefore counts once in the pool and once against the device that
//! made it; aliases of one allocation are views, never additional charges.
//! This is process accounting, not an OS-wide reservation.

use std::sync::{Arc, Mutex, MutexGuard};

/// Seismic-owned charges and enforced limits. This is not global driver or
/// OS usage: other processes and non-Seismic allocations are absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryUsage {
    /// Bytes charged by allocations made through this opened device.
    pub charged: u64,
    /// Device memory this opened device holds: `charged` plus what its
    /// backend holds beyond its allocations (the free space of the blocks it
    /// sub-allocates from, and driver objects such as instantiated CUDA
    /// graphs). What the device's memory pool actually loses to Seismic.
    pub held: u64,
    /// The part of `held` beyond `charged` that later allocations take
    /// without the device holding more: the free space of its blocks.
    pub unallocated: u64,
    /// The device's enforced allocation limit.
    pub limit: Option<u64>,
    /// Bytes charged by every opened device in this process whose
    /// allocations reside in the same physical pool, including `charged`.
    pub pool_charged: u64,
    /// The most `charged` has been since the device was opened or its mark
    /// was last restarted ([`MemoryDomain::restart_high_water`]): what was
    /// charged and released again in between is seen here and nowhere else.
    pub high_water: u64,
    /// The most `charged` would have been, over the same span as
    /// `high_water`, had the enforced limit granted a reservation it
    /// refused: what was charged then and what was asked for. None when
    /// the limit refused nothing in that span.
    pub refused_demand: Option<u64>,
}

/// Process-wide charges against one physical memory pool.
pub(crate) struct PoolLedger {
    charged: Mutex<u64>,
}

impl PoolLedger {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            charged: Mutex::new(0),
        })
    }

    fn charged(&self) -> MutexGuard<'_, u64> {
        self.charged
            .lock()
            .expect("memory pool ledger lock poisoned")
    }
}

struct MemoryState {
    charged: u64,
    limit: Option<u64>,
    high_water: u64,
    refused_demand: Option<u64>,
}

/// One opened device's accounting scope within its pool ledger. Lock order
/// is always device scope, then pool ledger.
pub(crate) struct MemoryDomain {
    pool: Arc<PoolLedger>,
    state: Mutex<MemoryState>,
}

impl MemoryDomain {
    pub(crate) fn new(pool: Arc<PoolLedger>) -> Arc<Self> {
        Arc::new(Self {
            pool,
            state: Mutex::new(MemoryState {
                charged: 0,
                limit: None,
                high_water: 0,
                refused_demand: None,
            }),
        })
    }

    /// Start the high-water mark again from what is charged now, and forget
    /// the refusals before, so that the next reading is of the work between
    /// the two.
    pub(crate) fn restart_high_water(&self) {
        let mut state = self.state();
        state.high_water = state.charged;
        state.refused_demand = None;
    }

    fn state(&self) -> MutexGuard<'_, MemoryState> {
        self.state
            .lock()
            .expect("device memory-accounting lock poisoned")
    }

    pub(crate) fn usage(&self) -> MemoryUsage {
        let state = self.state();
        let pool_charged = *self.pool.charged();
        MemoryUsage {
            charged: state.charged,
            held: state.charged,
            unallocated: 0,
            limit: state.limit,
            pool_charged,
            high_water: state.high_water,
            refused_demand: state.refused_demand,
        }
    }

    pub(crate) fn set_limit(&self, limit: Option<u64>) {
        let mut state = self.state();
        // A falling platform limit cannot revoke allocations already held by
        // the device. Keep the lower limit so every further reservation is
        // refused until releases bring the charge back below it.
        state.limit = limit;
    }

    pub(crate) fn reserve(
        self: &Arc<Self>,
        bytes: u64,
    ) -> Result<MemoryReservation, MemoryCapacity> {
        let mut state = self.state();
        let mut pool = self.pool.charged();
        // The pool total bounds every scope total, so its overflow check
        // covers the device scope as well.
        let pool_next = pool.checked_add(bytes).ok_or(MemoryCapacity {
            required: bytes,
            available: 0,
        })?;
        let next = state.charged + bytes;
        if let Some(limit) = state.limit {
            if next > limit {
                state.refused_demand = state.refused_demand.max(Some(next));
                return Err(MemoryCapacity {
                    required: bytes,
                    available: limit.saturating_sub(state.charged),
                });
            }
        }
        *pool = pool_next;
        state.charged = next;
        state.high_water = state.high_water.max(next);
        Ok(MemoryReservation {
            domain: self.clone(),
            remaining: bytes,
        })
    }

    fn release(&self, bytes: u64, what: &str) {
        let mut state = self.state();
        let mut pool = self.pool.charged();
        state.charged = state
            .charged
            .checked_sub(bytes)
            .unwrap_or_else(|| panic!("{what} exceeded the device accounting total"));
        *pool = pool
            .checked_sub(bytes)
            .unwrap_or_else(|| panic!("{what} exceeded the pool accounting total"));
    }

    /// Charge `bytes` an allocation held before and holds again. This is the
    /// return of a released part of a live allocation, not a reservation, so
    /// it is never refused: like a falling limit, it can leave the device
    /// above its limit, and every further reservation is then refused until
    /// releases bring the charge back below it.
    fn restore(&self, bytes: u64) {
        let mut state = self.state();
        let mut pool = self.pool.charged();
        *pool = pool
            .checked_add(bytes)
            .expect("restored charge overflowed the pool accounting total");
        state.charged += bytes;
        state.high_water = state.high_water.max(state.charged);
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MemoryCapacity {
    pub(crate) required: u64,
    pub(crate) available: u64,
}

/// One atomic reservation for a complete invocation. It is split into the
/// allocation-owned charges as physical allocations are created; any unused
/// tail is released on drop. This prevents concurrent public allocations
/// from invalidating a successful invocation-capacity check.
pub(crate) struct MemoryReservation {
    domain: Arc<MemoryDomain>,
    remaining: u64,
}

impl MemoryReservation {
    /// Reserved bytes not yet taken by an allocation.
    pub(crate) fn remaining(&self) -> u64 {
        self.remaining
    }

    pub(crate) fn take(&mut self, bytes: u64) -> MemoryCharge {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .expect("invocation allocated more memory than it atomically reserved");
        MemoryCharge {
            domain: self.domain.clone(),
            bytes,
        }
    }

    /// Transfer an already reserved increase into the charge for one
    /// physical allocation whose backing is being resized in place.
    pub(crate) fn extend(&mut self, charge: &mut MemoryCharge, bytes: u64) {
        assert!(Arc::ptr_eq(&self.domain, &charge.domain));
        let increased = charge
            .bytes
            .checked_add(bytes)
            .expect("recommit charge overflow");
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .expect("recommit exceeded reservation");
        charge.bytes = increased;
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if self.remaining == 0 {
            return;
        }
        self.domain
            .release(self.remaining, "unused memory reservation");
    }
}

pub(crate) struct MemoryCharge {
    domain: Arc<MemoryDomain>,
    bytes: u64,
}

impl MemoryCharge {
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn shrink(&mut self, bytes: u64) {
        assert!(bytes <= self.bytes);
        self.domain
            .release(self.bytes - bytes, "in-place recommit release");
        self.bytes = bytes;
    }

    /// Bring the charge to the `bytes` its allocation held before an
    /// in-place recommit that was abandoned, once the backend has restored
    /// that backing: abandoned growth releases what it added, and an
    /// abandoned shrink takes back what it released (never refused; see
    /// `MemoryDomain::restore`).
    pub(crate) fn restore(&mut self, bytes: u64) {
        if bytes <= self.bytes {
            self.shrink(bytes);
        } else {
            self.domain.restore(bytes - self.bytes);
            self.bytes = bytes;
        }
    }
}

impl Drop for MemoryCharge {
    fn drop(&mut self) {
        self.domain.release(self.bytes, "device allocation charge");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domain() -> Arc<MemoryDomain> {
        MemoryDomain::new(PoolLedger::new())
    }

    #[test]
    fn reservation_transfer_and_drop_conserve_all_charges() {
        for limit in 0..8 {
            for requested in 0..10 {
                let domain = domain();
                domain.set_limit(Some(limit));
                let reservation = domain.reserve(requested);
                if requested > limit {
                    assert!(reservation.is_err());
                    assert_eq!(domain.usage().charged, 0);
                    assert_eq!(domain.usage().pool_charged, 0);
                    continue;
                }
                for transferred in 0..=requested {
                    // Each iteration replays the production reserve/take/drop operations.
                    let domain = self::domain();
                    domain.set_limit(Some(limit));
                    let mut reservation = domain.reserve(requested).unwrap();
                    let charge = reservation.take(transferred);
                    assert_eq!(domain.usage().charged, requested);
                    drop(reservation);
                    assert_eq!(domain.usage().charged, transferred);
                    assert_eq!(domain.usage().pool_charged, transferred);
                    drop(charge);
                    assert_eq!(domain.usage().charged, 0);
                    assert_eq!(domain.usage().pool_charged, 0);
                }
            }
        }
    }

    #[test]
    fn resized_charge_transfers_only_growth_and_releases_shrink() {
        let domain = domain();
        domain.set_limit(Some(24));
        let mut initial = domain.reserve(16).unwrap();
        let mut charge = initial.take(16);
        drop(initial);
        assert_eq!(domain.usage().charged, 16);
        let mut growth = domain.reserve(8).unwrap();
        growth.extend(&mut charge, 8);
        drop(growth);
        assert_eq!(charge.bytes(), 24);
        assert_eq!(domain.usage().charged, 24);
        charge.shrink(8);
        assert_eq!(domain.usage().charged, 8);
        drop(charge);
        assert_eq!(domain.usage().charged, 0);
    }

    #[test]
    fn restored_charge_takes_back_released_bytes_whatever_the_limit() {
        let domain = domain();
        domain.set_limit(Some(16));
        let mut initial = domain.reserve(16).unwrap();
        let mut charge = initial.take(16);
        drop(initial);
        charge.shrink(4);
        // Another allocation spends what the shrink released.
        let other = domain.reserve(12).unwrap();
        charge.restore(16);
        assert_eq!(charge.bytes(), 16);
        assert_eq!(domain.usage().charged, 28);
        assert_eq!(domain.usage().pool_charged, 28);
        assert_eq!(domain.usage().high_water, 28);
        assert!(domain.reserve(1).is_err());
        drop(other);
        assert_eq!(domain.usage().charged, 16);
        // Restoring to fewer bytes releases the difference.
        charge.restore(8);
        assert_eq!(domain.usage().charged, 8);
        drop(charge);
        assert_eq!(domain.usage().charged, 0);
        assert_eq!(domain.usage().pool_charged, 0);
    }

    #[test]
    fn a_refused_reservation_is_read_as_the_demand_it_made_until_the_mark_restarts() {
        let domain = domain();
        domain.set_limit(Some(10));
        let held = domain.reserve(6).unwrap();
        assert_eq!(domain.usage().refused_demand, None);
        assert!(domain.reserve(7).is_err());
        assert!(domain.reserve(5).is_err());
        let usage = domain.usage();
        assert_eq!((usage.charged, usage.high_water), (6, 6));
        // The largest refused demand: what was charged and what was asked.
        assert_eq!(usage.refused_demand, Some(13));
        domain.restart_high_water();
        assert_eq!(domain.usage().refused_demand, None);
        drop(held);
    }

    #[test]
    fn overflow_and_lower_limit_leave_existing_reservations_intact() {
        let domain = domain();
        let reservation = domain.reserve(u64::MAX).unwrap();
        assert!(domain.reserve(1).is_err());
        domain.set_limit(Some(0));
        assert_eq!(domain.usage().charged, u64::MAX);
        assert_eq!(domain.usage().limit, Some(0));
        assert!(domain.reserve(1).is_err());
        drop(reservation);
        assert_eq!(domain.usage().charged, 0);
    }

    #[test]
    fn lower_limit_blocks_growth_until_charges_fall_below_it() {
        let domain = domain();
        let first = domain.reserve(4).unwrap();
        let second = domain.reserve(4).unwrap();
        domain.set_limit(Some(5));
        assert_eq!(domain.usage().charged, 8);
        assert_eq!(domain.reserve(1).err().unwrap().available, 0);
        drop(second);
        assert_eq!(domain.usage().charged, 4);
        assert_eq!(domain.reserve(2).err().unwrap().available, 1);
        let growth = domain.reserve(1).unwrap();
        drop(growth);
        drop(first);
        assert_eq!(domain.usage().charged, 0);
    }

    #[test]
    fn devices_sharing_a_pool_charge_it_once_each_under_their_own_limits() {
        let pool = PoolLedger::new();
        let metal = MemoryDomain::new(pool.clone());
        let cpu = MemoryDomain::new(pool.clone());
        metal.set_limit(Some(8));
        let mut gpu = metal.reserve(6).unwrap();
        let _gpu = gpu.take(6);
        let host = cpu.reserve(5).unwrap();
        assert_eq!(metal.usage().charged, 6);
        assert_eq!(cpu.usage().charged, 5);
        assert_eq!(metal.usage().pool_charged, 11);
        assert_eq!(cpu.usage().pool_charged, 11);
        // The device limit constrains only this device's own charges.
        assert!(metal.reserve(3).is_err());
        drop(host);
        assert_eq!(metal.usage().pool_charged, 6);
        // A separate pool is a separate ledger.
        let dedicated = MemoryDomain::new(PoolLedger::new());
        let _dedicated = dedicated.reserve(4).unwrap();
        assert_eq!(metal.usage().pool_charged, 6);
        assert_eq!(dedicated.usage().pool_charged, 4);
    }

    #[test]
    fn concurrent_reservations_cannot_both_spend_the_same_capacity() {
        let domain = domain();
        domain.set_limit(Some(10));
        let ready = std::sync::Barrier::new(3);
        let release = std::sync::Barrier::new(3);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        let reservation = domain.reserve(6);
                        ready.wait();
                        release.wait();
                        reservation.is_ok()
                    })
                })
                .collect();
            ready.wait();
            assert_eq!(domain.usage().charged, 6);
            release.wait();
            assert_eq!(
                workers
                    .into_iter()
                    .map(|worker| u32::from(worker.join().unwrap()))
                    .sum::<u32>(),
                1
            );
        });
        assert_eq!(domain.usage().charged, 0);
    }
}
