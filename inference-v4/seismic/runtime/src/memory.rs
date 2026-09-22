//! Atomic memory reservation and allocation lifetime accounting.
//! No backend or submission service is needed to exercise this state machine.

use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MemoryUsage {
    pub(crate) charged: u64,
    pub(crate) limit: Option<u64>,
}

struct MemoryState {
    charged: u64,
    limit: Option<u64>,
}

pub(crate) struct MemoryDomain {
    state: Mutex<MemoryState>,
}

impl MemoryDomain {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
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
        MemoryUsage {
            charged: state.charged,
            limit: state.limit,
        }
    }

    pub(crate) fn set_limit(&self, limit: Option<u64>) -> Result<(), crate::api::MemoryLimitError> {
        let mut state = self.state();
        if let Some(limit) = limit.filter(|limit| *limit < state.charged) {
            return Err(crate::api::MemoryLimitError {
                limit,
                charged: state.charged,
            });
        }
        state.limit = limit;
        Ok(())
    }

    pub(crate) fn reserve(
        self: &Arc<Self>,
        bytes: u64,
    ) -> Result<MemoryReservation, MemoryCapacity> {
        let mut state = self.state();
        let next = state.charged.checked_add(bytes).ok_or(MemoryCapacity {
            required: bytes,
            available: 0,
        })?;
        if let Some(limit) = state.limit {
            if next > limit {
                return Err(MemoryCapacity {
                    required: bytes,
                    available: limit.saturating_sub(state.charged),
                });
            }
        }
        state.charged = next;
        Ok(MemoryReservation {
            domain: self.clone(),
            remaining: bytes,
        })
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
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if self.remaining == 0 {
            return;
        }
        let mut state = self.domain.state();
        state.charged = state
            .charged
            .checked_sub(self.remaining)
            .expect("unused memory reservation exceeded the accounting total");
    }
}

pub(crate) struct MemoryCharge {
    domain: Arc<MemoryDomain>,
    bytes: u64,
}

impl Drop for MemoryCharge {
    fn drop(&mut self) {
        let mut state = self.domain.state();
        state.charged = state
            .charged
            .checked_sub(self.bytes)
            .expect("device allocation charge exceeded the accounting total");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservation_transfer_and_drop_conserve_all_charges() {
        for limit in 0..8 {
            for requested in 0..10 {
                let domain = MemoryDomain::new();
                domain.set_limit(Some(limit)).unwrap();
                let reservation = domain.reserve(requested);
                if requested > limit {
                    assert!(reservation.is_err());
                    assert_eq!(domain.usage().charged, 0);
                    continue;
                }
                for transferred in 0..=requested {
                    // Each iteration replays the production reserve/take/drop operations.
                    let domain = MemoryDomain::new();
                    domain.set_limit(Some(limit)).unwrap();
                    let mut reservation = domain.reserve(requested).unwrap();
                    let charge = reservation.take(transferred);
                    assert_eq!(domain.usage().charged, requested);
                    drop(reservation);
                    assert_eq!(domain.usage().charged, transferred);
                    drop(charge);
                    assert_eq!(domain.usage().charged, 0);
                }
            }
        }
    }

    #[test]
    fn overflow_and_limit_changes_leave_existing_reservations_intact() {
        let domain = MemoryDomain::new();
        let reservation = domain.reserve(u64::MAX).unwrap();
        assert!(domain.reserve(1).is_err());
        assert!(domain.set_limit(Some(0)).is_err());
        assert_eq!(domain.usage().charged, u64::MAX);
        assert_eq!(domain.usage().limit, None);
        drop(reservation);
        assert_eq!(domain.usage().charged, 0);
    }

    #[test]
    fn concurrent_reservations_cannot_both_spend_the_same_capacity() {
        let domain = MemoryDomain::new();
        domain.set_limit(Some(10)).unwrap();
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
