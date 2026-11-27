//! Scoped RAII [`BufferLease`] replacing `Arc<dyn Destination>`, plus
//! [`ReaderRetirementTracker`] enforcing overwrite safety across both active
//! readers and reserved-but-unpublished passes.

use crate::fabric::DomainId;
use std::collections::HashMap;
use std::fmt;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// Pool/allocator callback invoked when a [`BufferLease`] is released normally
/// or quarantined on ambiguous hardware failure.
pub trait LeasePool: Send + Sync {
    /// Return `(slot, generation)` to the allocator once hardware AND all
    /// subscribers have retired.
    fn release_slot(&self, slot: u32, generation: u64);

    /// Record that `(slot, generation)` was quarantined due to an ambiguous or
    /// partial driver failure; the slot MUST NOT be reused until explicit device
    /// teardown or completion proof.
    fn quarantine_slot(&self, _slot: u32, _generation: u64) {}
}

/// Inline lease releaser (zero per-lease heap allocation on handoff).
struct LeaseReleaser {
    pool: Option<Arc<dyn LeasePool>>,
    slot: u32,
    quarantined: bool,
}

/// Concrete, fixed-size RAII lease over a memory region in a [`DomainId`].
///
/// - Normal `Drop` returns `(slot, generation)` to its [`LeasePool`].
/// - [`BufferLease::quarantine`] disarms normal release so quarantined backing
///   remains permanently pinned until explicit teardown.
pub struct BufferLease {
    pub domain: DomainId,
    pub ptr: NonNull<u8>,
    pub buffer_handle: u64,
    pub byte_offset: u64,
    pub len: usize,
    pub alignment: u32,
    pub generation: u64,
    releaser: LeaseReleaser,
}

// SAFETY: `BufferLease` represents exclusive/leased ownership of a memory
// region in `domain` whose lifetime is pinned until `Drop` (or permanently
// pinned if quarantined).
unsafe impl Send for BufferLease {}
unsafe impl Sync for BufferLease {}

impl BufferLease {
    /// Create a pool-backed [`BufferLease`] for `(slot, generation)`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_pool(
        domain: DomainId,
        ptr: NonNull<u8>,
        buffer_handle: u64,
        byte_offset: u64,
        len: usize,
        alignment: u32,
        slot: u32,
        generation: u64,
        pool: Arc<dyn LeasePool>,
    ) -> Self {
        Self {
            domain,
            ptr,
            buffer_handle,
            byte_offset,
            len,
            alignment,
            generation,
            releaser: LeaseReleaser {
                pool: Some(pool),
                slot,
                quarantined: false,
            },
        }
    }

    /// Create an unmanaged lease (e.g. for static/caller-owned regions or tests).
    pub fn unmanaged(
        domain: DomainId,
        ptr: NonNull<u8>,
        buffer_handle: u64,
        byte_offset: u64,
        len: usize,
        alignment: u32,
        generation: u64,
    ) -> Self {
        Self {
            domain,
            ptr,
            buffer_handle,
            byte_offset,
            len,
            alignment,
            generation,
            releaser: LeaseReleaser {
                pool: None,
                slot: 0,
                quarantined: false,
            },
        }
    }

    /// Slot index within the owning pool (0 for unmanaged leases).
    #[inline]
    pub fn slot(&self) -> u32 {
        self.releaser.slot
    }

    /// Whether this lease has been quarantined.
    #[inline]
    pub fn is_quarantined(&self) -> bool {
        self.releaser.quarantined
    }

    /// Permanently quarantine this lease: notifies the owning [`LeasePool`] and
    /// disarms normal `Drop` reclamation so the backing memory is NEVER returned
    /// to the allocator until explicit device teardown.
    pub fn quarantine(&mut self) {
        if !self.releaser.quarantined {
            self.releaser.quarantined = true;
            if let Some(pool) = &self.releaser.pool {
                pool.quarantine_slot(self.releaser.slot, self.generation);
            }
        }
    }

    /// Explicitly release a previously quarantined lease during device teardown
    /// or after explicit hardware completion proof.
    pub fn release_on_teardown(mut self) {
        self.releaser.quarantined = false;
        drop(self);
    }
}

impl Drop for BufferLease {
    fn drop(&mut self) {
        if self.releaser.quarantined {
            return;
        }
        if let Some(pool) = self.releaser.pool.take() {
            pool.release_slot(self.releaser.slot, self.generation);
        }
    }
}

impl fmt::Debug for BufferLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BufferLease")
            .field("domain", &self.domain)
            .field("ptr", &self.ptr)
            .field("buffer_handle", &self.buffer_handle)
            .field("byte_offset", &self.byte_offset)
            .field("len", &self.len)
            .field("alignment", &self.alignment)
            .field("slot", &self.releaser.slot)
            .field("generation", &self.generation)
            .field("quarantined", &self.releaser.quarantined)
            .finish()
    }
}

/// Identifies a reusable slot in a memory domain for reader-retirement tracking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotKey {
    pub domain: DomainId,
    pub slot: u32,
}

/// RAII pin representing either an active reader or a reserved-but-unpublished
/// pass that already resolved a slot's map entry.
pub struct ReaderPin {
    tracker: Arc<ReaderRetirementTracker>,
    slot: SlotKey,
    generation: u64,
    reserved_unpublished: bool,
    released: bool,
}

impl ReaderPin {
    /// Promote a reserved-but-unpublished pass pin to an active submitted reader.
    pub fn mark_submitted(&mut self) {
        if self.reserved_unpublished && !self.released {
            let mut inner = self.tracker.inner.lock().expect("reader retirement tracker");
            if let Some(entry) = inner.get_mut(&self.slot) {
                if entry.generation == self.generation {
                    entry.reserved_unpublished = entry.reserved_unpublished.saturating_sub(1);
                    entry.active_readers += 1;
                }
            }
            self.reserved_unpublished = false;
        }
    }

    /// Explicitly retire this reader pin.
    pub fn retire(mut self) {
        self.release_inner();
    }

    fn release_inner(&mut self) {
        if self.released {
            return;
        }
        self.released = true;
        let mut inner = self.tracker.inner.lock().expect("reader retirement tracker");
        if let Some(entry) = inner.get_mut(&self.slot) {
            if entry.generation == self.generation {
                if self.reserved_unpublished {
                    entry.reserved_unpublished = entry.reserved_unpublished.saturating_sub(1);
                } else {
                    entry.active_readers = entry.active_readers.saturating_sub(1);
                }
                if entry.active_readers == 0 && entry.reserved_unpublished == 0 {
                    inner.remove(&self.slot);
                }
            }
        }
    }
}

impl Drop for ReaderPin {
    fn drop(&mut self) {
        self.release_inner();
    }
}

#[derive(Debug)]
struct SlotReaders {
    generation: u64,
    active_readers: u32,
    reserved_unpublished: u32,
}

/// Tracks both active readers and reserved-but-unpublished passes per slot so
/// overwrite safety (`can_overwrite`) is enforced by construction.
#[derive(Default)]
pub struct ReaderRetirementTracker {
    inner: Mutex<HashMap<SlotKey, SlotReaders>>,
}

impl ReaderRetirementTracker {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Pin `(slot, generation)` for a reserved pass that has resolved the map
    /// but has not yet been submitted to the device queue.
    pub fn reserve_unpublished(self: &Arc<Self>, slot: SlotKey, generation: u64) -> ReaderPin {
        let mut inner = self.inner.lock().expect("reader retirement tracker");
        let entry = inner.entry(slot).or_insert(SlotReaders {
            generation,
            active_readers: 0,
            reserved_unpublished: 0,
        });
        assert_eq!(
            entry.generation, generation,
            "slot {slot:?} pinned at generation {generation} while generation {} still has live readers",
            entry.generation
        );
        entry.reserved_unpublished += 1;
        ReaderPin {
            tracker: Arc::clone(self),
            slot,
            generation,
            reserved_unpublished: true,
            released: false,
        }
    }

    /// Pin `(slot, generation)` for an active reader.
    pub fn pin_active_reader(self: &Arc<Self>, slot: SlotKey, generation: u64) -> ReaderPin {
        let mut inner = self.inner.lock().expect("reader retirement tracker");
        let entry = inner.entry(slot).or_insert(SlotReaders {
            generation,
            active_readers: 0,
            reserved_unpublished: 0,
        });
        assert_eq!(
            entry.generation, generation,
            "slot {slot:?} pinned at generation {generation} while generation {} still has live readers",
            entry.generation
        );
        entry.active_readers += 1;
        ReaderPin {
            tracker: Arc::clone(self),
            slot,
            generation,
            reserved_unpublished: false,
            released: false,
        }
    }

    /// True **if and only if** zero active readers AND zero reserved unpublished
    /// passes reference `slot`.
    pub fn can_overwrite(&self, slot: SlotKey) -> bool {
        let inner = self.inner.lock().expect("reader retirement tracker");
        inner
            .get(&slot)
            .is_none_or(|e| e.active_readers == 0 && e.reserved_unpublished == 0)
    }

    /// `(active_readers, reserved_unpublished)` for diagnostics and tests.
    pub fn counts(&self, slot: SlotKey) -> (u32, u32) {
        let inner = self.inner.lock().expect("reader retirement tracker");
        inner
            .get(&slot)
            .map_or((0, 0), |e| (e.active_readers, e.reserved_unpublished))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::DeviceId;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct CountingPool {
        released: AtomicU32,
        quarantined: AtomicU32,
    }

    impl LeasePool for CountingPool {
        fn release_slot(&self, _slot: u32, _generation: u64) {
            self.released.fetch_add(1, Ordering::AcqRel);
        }
        fn quarantine_slot(&self, _slot: u32, _generation: u64) {
            self.quarantined.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[test]
    fn quarantined_lease_never_releases_on_drop_until_teardown() {
        let pool = Arc::new(CountingPool {
            released: AtomicU32::new(0),
            quarantined: AtomicU32::new(0),
        });
        let domain = DomainId::DeviceVram(DeviceId(0));
        let mut lease = BufferLease::from_pool(
            domain,
            NonNull::dangling(),
            1,
            0,
            4096,
            4096,
            7,
            1,
            pool.clone(),
        );
        lease.quarantine();
        assert_eq!(pool.quarantined.load(Ordering::Acquire), 1);
        drop(lease);
        assert_eq!(pool.released.load(Ordering::Acquire), 0, "quarantined lease must not release on drop");

        let mut lease2 = BufferLease::from_pool(
            domain,
            NonNull::dangling(),
            1,
            4096,
            4096,
            4096,
            8,
            1,
            pool.clone(),
        );
        lease2.quarantine();
        lease2.release_on_teardown();
        assert_eq!(pool.released.load(Ordering::Acquire), 1, "teardown explicitly releases quarantined lease");
    }

    #[test]
    fn reader_retirement_blocks_overwrite_for_reserved_unpublished_pass() {
        let tracker = ReaderRetirementTracker::new();
        let slot = SlotKey {
            domain: DomainId::DeviceVram(DeviceId(0)),
            slot: 3,
        };
        assert!(tracker.can_overwrite(slot));

        let mut reserved = tracker.reserve_unpublished(slot, 1);
        assert!(!tracker.can_overwrite(slot));
        assert_eq!(tracker.counts(slot), (0, 1));

        reserved.mark_submitted();
        assert!(!tracker.can_overwrite(slot));
        assert_eq!(tracker.counts(slot), (1, 0));

        reserved.retire();
        assert!(tracker.can_overwrite(slot));
    }
}
