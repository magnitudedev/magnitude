//! Where an opened device's memory is charged: the embedder's ledger, never
//! a count of Seismic's own.
//!
//! A device opens with the embedder's [`Store`] of every store it reaches.
//! Each byte its backend takes is charged to one [`Entry`] of that store, at
//! the moment it is taken, at its exact size (rounding, block free space and
//! driver objects included), and freed from it as it is freed: the backend
//! holds each [`Charge`] with the native object it backs. An allocation is
//! charged to the entry its caller names for the store ([`charging`]); the
//! device's own memory (staging, contexts, driver objects) to the store's
//! own entry ([`Store::own`]), which its backend names explicitly; and an
//! allocation that names none, to the store's sole entry where it counts
//! every byte in one ([`Store::sole`], a tally). In an embedder's ledger
//! such an allocation has no owner: an invariant violation, never charged
//! to a default.
//!
//! An allocation of a contract lands in one of the kinds its placement
//! names, and lies along that kind's access path. A caller may confine a
//! device's allocations to one kind ([`DeviceStores::confined`]).

pub use seismic_memory::entry::{Backed, Carved, Change, Charge, Charging, DropScope, Entry, Refused, Store, Tally, Wiring};
pub use seismic_memory::{AllocationFailure, HostBlock, PageSize};
use seismic_compiler::errors::ExecutionError;
use seismic_native_target::process::{ProcessFootprint, ProcessId};
use seismic_topology::{AccessPathId, Contract, DeviceTopology, Executor, PhysicalIdentity, StoreId, StoreIdentity};
use std::collections::BTreeMap;
use std::cell::RefCell;
use std::sync::Arc;

thread_local! {
    /// The entries the allocations this thread makes now are charged to, by
    /// store: for each store, the innermost [`charging`] scope's that names
    /// it.
    static NAMED: RefCell<Vec<Arc<dyn Entry>>> = const { RefCell::new(Vec::new()) };

    /// The kind each device's allocations this thread makes now are
    /// confined to, by device: innermost scope last
    /// ([`DeviceStores::confined`]).
    static CONFINED: RefCell<Vec<(usize, u32)>> = const { RefCell::new(Vec::new()) };
}

/// Run `allocate`, charging every allocation it makes on this thread in a
/// store one of `entries` lies in to that entry: the caller names the ledger
/// entry each allocation is held under. Scopes nest: an inner scope names
/// only its own stores, and an allocation in any other store stays charged
/// to the entry an outer scope named for it. An allocation of an embedder's
/// device in a store no scope names panics ([`DeviceStores::entry`]).
///
/// # Panics
/// Where two of `entries` lie in one store.
pub fn charging<T>(entries: &[Arc<dyn Entry>], allocate: impl FnOnce() -> T) -> T {
    for (index, entry) in entries.iter().enumerate() {
        assert!(
            entries[..index].iter().all(|other| other.store() != entry.store()),
            "invariant violated: two entries named for {:?}",
            entry.store()
        );
    }
    let outer = NAMED.with(|named| {
        let mut named = named.borrow_mut();
        let inner: Vec<_> = named
            .iter()
            .filter(|outer| entries.iter().all(|entry| entry.store() != outer.store()))
            .chain(entries)
            .cloned()
            .collect();
        std::mem::replace(&mut *named, inner)
    });
    struct Restore(Option<Vec<Arc<dyn Entry>>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let outer = self.0.take().expect("restored once");
            NAMED.with(|named| *named.borrow_mut() = outer);
        }
    }
    let _restore = Restore(Some(outer));
    allocate()
}

/// The entry named for `store` on this thread ([`charging`]), where one is.
fn named_in(store: &Arc<dyn Store>) -> Option<Arc<dyn Entry>> {
    NAMED.with(|named| named.borrow().iter().find(|entry| entry.store() == store.identity()).cloned())
}

/// The bytes held along `topology`'s access paths `charges` on each store,
/// each store once, in store order so that equal charges compare equal.
pub(crate) fn store_charges(
    topology: &DeviceTopology,
    charges: impl IntoIterator<Item = (AccessPathId, u64)>,
) -> BTreeMap<StoreId, u64> {
    let mut stores = BTreeMap::new();
    for (path, bytes) in charges {
        let store = topology.access_path(path).expect("charged path belongs to topology").store();
        *stores.entry(store).or_default() += bytes;
    }
    stores
}

/// What `process` holds in `store` now, by the figure the system reports of
/// it there: in host RAM its footprint
/// ([`seismic_native_target::process::host_footprint`]); in a GPU's own
/// memory what NVML counts of it ([`seismic_cuda::process_footprint`]).
/// Unobserved in every other store, and wherever the system reports no
/// figure: no figure is guessed.
pub fn process_footprint(process: ProcessId, store: &StoreIdentity) -> ProcessFootprint {
    match store {
        StoreIdentity::HostRam => seismic_native_target::process::host_footprint(process),
        StoreIdentity::DeviceMemory(PhysicalIdentity::Uuid(uuid)) => seismic_cuda::process_footprint(process, *uuid),
        StoreIdentity::DeviceMemory(PhysicalIdentity::Host { .. } | PhysicalIdentity::Registry(_)) | StoreIdentity::Volume(_) => {
            ProcessFootprint::Unobserved
        }
    }
}

/// What a driver grants the process of a device store beyond what it uses
/// now, read in the same query as the usage its driver's memory was just
/// settled from ([`crate::api::device::DeviceInner::reconcile_driver_memory`]):
/// the two are of one instant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grant {
    pub left: u64,
}

/// Where one contract's storage lies.
struct Placed {
    store: Arc<dyn Store>,
    kinds: Vec<u32>,
}

/// One of the executor's kinds: the backend's name of it, the access path
/// its memory lies along, and that path's store (Vulkan's pools allocate by
/// kind, [`DeviceStores::kind_stores`]).
struct Kind {
    id: u32,
    path: AccessPathId,
    #[cfg(not(target_os = "macos"))]
    store: Arc<dyn Store>,
}

/// An opened device's stores: the embedder's ledger of each store it
/// reaches, and which one each contract's storage and each of its backend's
/// kinds lies in.
pub(crate) struct DeviceStores {
    /// Each contract's store and the kinds it is allocated from as the
    /// backend names them, in [`Contract::ALL`] order.
    placed: [Placed; Contract::COUNT],
    /// Each of the executor's kinds.
    kinds: Vec<Kind>,
    /// Host RAM's: transfer staging and host-side driver objects.
    host: Arc<dyn Store>,
    /// Every store it was opened with.
    stores: Vec<Arc<dyn Store>>,
    /// Whether they are an embedder's ledger, not tallies of the device's
    /// own ([`crate::devices::Devices::open_with`]).
    embedded: bool,
}

impl DeviceStores {
    /// The stores `executor` of `topology` reaches, of `stores`, the
    /// embedder's ledger of each.
    ///
    /// # Panics
    /// Where `stores` lacks a store the executor places storage in, or holds
    /// one twice.
    pub(crate) fn new(topology: &DeviceTopology, executor: &Executor, stores: Vec<Arc<dyn Store>>, embedded: bool) -> Arc<Self> {
        for (index, store) in stores.iter().enumerate() {
            assert!(
                stores[..index].iter().all(|other| other.identity() != store.identity()),
                "invariant violated: a device opened with {:?} twice",
                store.identity()
            );
        }
        let of = |id| {
            let identity = topology.store_identity(id).expect("a placement's store belongs to its topology");
            stores
                .iter()
                .find(|store| store.identity() == &identity)
                .cloned()
                .unwrap_or_else(|| panic!("invariant violated: {} opened without a ledger of {identity:?}", executor.selector()))
        };
        let native = |id| topology.kind(id).expect("a placement names kinds of its revision").native().id;
        let placed = Contract::ALL.map(|contract| {
            let path = *executor.paths(contract).first().expect("an opened executor places every contract");
            let store = topology.access_path(path).expect("a placement's path belongs to its topology").store();
            Placed { store: of(store), kinds: executor.placement(contract).iter().map(|id| native(*id)).collect() }
        });
        let kinds = topology
            .kinds()
            .iter()
            .filter(|kind| kind.executor() == executor.id())
            .map(|kind| {
                let path = *kind.paths().first().expect("a kind lies on an access path");
                Kind {
                    id: kind.native().id,
                    path,
                    #[cfg(not(target_os = "macos"))]
                    store: of(topology.access_path(path).expect("a kind's path belongs to its topology").store()),
                }
            })
            .collect();
        let host = of(topology.host_ram().id());
        Arc::new(Self { placed, kinds, host, stores, embedded })
    }

    /// Whether a device opened in these may be shared by an open in an
    /// embedder's `stores`: these are an embedder's ledger of the same
    /// stores, never tallies of the device's own. The process has one
    /// embedder's ledger per store.
    pub(crate) fn opened_with(&self, stores: &[Arc<dyn Store>]) -> bool {
        self.embedded
            && stores.len() == self.stores.len()
            && stores.iter().all(|store| self.stores.iter().any(|held| held.identity() == store.identity()))
    }

    fn placed(&self, contract: Contract) -> &Placed {
        &self.placed[Contract::ALL.iter().position(|of| *of == contract).expect("every contract is placed")]
    }

    /// The store `contract`'s storage lies in.
    pub(crate) fn store(&self, contract: Contract) -> Arc<dyn Store> {
        self.placed(contract).store.clone()
    }

    /// The kinds an allocation of `contract` made now on this thread is
    /// allocated from, in order, as the backend names them: its
    /// placement's, as [`Self::allocated_from`] confines them.
    pub(crate) fn kinds(&self, contract: Contract) -> Result<&[u32], ExecutionError> {
        self.allocated_from(&self.placed(contract).kinds)
    }

    /// Of `kinds`, which an allocation through this device may land in,
    /// those an allocation made now on this thread is allocated from: all,
    /// or the one its allocations are confined to ([`Self::confined`]),
    /// refused where `kinds` does not name it.
    pub(crate) fn allocated_from<'a>(&self, kinds: &'a [u32]) -> Result<&'a [u32], ExecutionError> {
        let Some(confined) = CONFINED.with(|confined| {
            confined.borrow().iter().rev().find(|(device, _)| *device == self.key()).map(|(_, kind)| *kind)
        }) else {
            return Ok(kinds);
        };
        kinds
            .iter()
            .find(|kind| **kind == confined)
            .map(std::slice::from_ref)
            .ok_or_else(|| {
                ExecutionError::AllocationFailed(format!(
                    "an allocation confined to kind {confined} lies in kinds {kinds:?}"
                ))
            })
    }

    /// Run `allocate`, confining every allocation it makes on this thread
    /// through this device to `kind`: one that may lie in `kind` lands
    /// there or is refused, never moving on to another kind, and any other
    /// is refused. Allocations on other threads and of other devices are
    /// not confined; scopes nest.
    ///
    /// # Panics
    /// Where `kind` is none of the executor's.
    pub(crate) fn confined<T>(&self, kind: u32, allocate: impl FnOnce() -> T) -> T {
        assert!(
            self.kinds.iter().any(|of| of.id == kind),
            "invariant violated: allocations confined to kind {kind}, none of the executor's"
        );
        CONFINED.with(|confined| confined.borrow_mut().push((self.key(), kind)));
        struct Release;
        impl Drop for Release {
            fn drop(&mut self) {
                CONFINED.with(|confined| confined.borrow_mut().pop().expect("a confinement scope is released once"));
            }
        }
        let _release = Release;
        allocate()
    }

    /// The device's key among this thread's confinements: its stores'
    /// address, fixed while the device is open.
    fn key(&self) -> usize {
        std::ptr::from_ref(self) as usize
    }

    /// The access path the memory of `kind`, one of the executor's, lies
    /// along: what an allocation that landed in it is charged along.
    ///
    /// # Panics
    /// Where `kind` is none of the executor's.
    pub(crate) fn path_of(&self, kind: u32) -> AccessPathId {
        self.kinds
            .iter()
            .find(|of| of.id == kind)
            .unwrap_or_else(|| panic!("invariant violated: an allocation landed in kind {kind}, none of the executor's"))
            .path
    }

    /// Each of the executor's kinds and its store.
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn kind_stores(&self) -> Vec<(u32, Arc<dyn Store>)> {
        self.kinds.iter().map(|kind| (kind.id, kind.store.clone())).collect()
    }

    /// Host RAM's store.
    pub(crate) fn host_store(&self) -> Arc<dyn Store> {
        self.host.clone()
    }

    /// The entry an allocation of `contract` made now on this thread is
    /// charged to: the one named for its store ([`charging`]), else the
    /// store's sole entry where it counts every byte in one ([`Store::sole`],
    /// a [`Tally`] of a device opened outside any embedder's ledger).
    ///
    /// # Panics
    /// Where neither is: an embedder's ledger has no owner for the bytes.
    pub(crate) fn entry(&self, contract: Contract) -> Arc<dyn Entry> {
        let store = &self.placed(contract).store;
        named_in(store).or_else(|| store.sole()).unwrap_or_else(|| {
            panic!(
                "invariant violated: an allocation of {contract:?} in {:?} outside a charging scope names no entry of the embedder's ledger",
                store.identity()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_topology::StoreIdentity;

    /// An allocation inside a charging scope is charged to the entry named
    /// for its store; outside one, a store's tally, its sole entry. Scopes
    /// nest: an inner scope that names no entry of the store leaves its
    /// allocations the outer scope's, and one that names one takes them.
    #[test]
    fn a_named_entry_takes_its_stores_allocations_within_its_scope() {
        let own = Tally::unbounded(StoreIdentity::HostRam);
        let store: Arc<dyn Store> = Arc::new(own.clone());
        let named = Tally::unbounded(StoreIdentity::HostRam);
        let named_entry: Arc<dyn Entry> = Arc::new(named.clone());
        let inner = Tally::unbounded(StoreIdentity::HostRam);
        let inner_entry: Arc<dyn Entry> = Arc::new(inner.clone());
        let take = || {
            let entry = named_in(&store).or_else(|| store.sole()).expect("a tally is its store's sole entry");
            entry.took(&entry.change(), 1, Wiring::Pageable);
        };
        take();
        charging(&[named_entry.clone()], || {
            take();
            charging(&[], take);
            charging(&[inner_entry.clone()], take);
            take();
        });
        take();
        assert_eq!((named.held(), inner.held(), own.held()), (3, 1, 2));
    }
}
