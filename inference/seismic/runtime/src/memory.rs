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
    /// The device's enforced allocation limit.
    pub limit: Option<u64>,
    /// Bytes charged by every opened device in this process whose
    /// allocations reside in the same physical pool, including `charged`.
    pub pool_charged: u64,
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
            }),
        })
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
            limit: state.limit,
            pool_charged,
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
                return Err(MemoryCapacity {
                    required: bytes,
                    available: limit.saturating_sub(state.charged),
                });
            }
        }
        *pool = pool_next;
        state.charged = next;
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
