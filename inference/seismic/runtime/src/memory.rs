//! Where an opened device's memory is charged: the embedder's ledger, never
//! a count of Seismic's own.
//!
//! A device opens with the embedder's [`Store`] of every store it reaches.
//! Each byte its backend takes is charged to one [`Entry`] of that store, at
//! the moment it is taken, at its exact size (rounding, block free space and
//! driver objects included), and freed from it as it is freed: the backend
//! holds each [`Charge`] with the native object it backs. Every allocation
//! is made through the [`Charges`] it is charged to, one entry of each store
//! the device reaches, which the embedder's ledger mints for the device
//! ([`crate::api::device::DeviceInner::charges`]). Each store's own entry
//! ([`Store::own`]) is charged by name, and only for a purpose no holder
//! causes: the backend's own memory (staging, contexts, driver objects);
//! every allocation of a device opened on tallies ([`Charges::tally`],
//! refused on a device opened in an embedder's ledger); a CUDA graph's
//! recording placeholders ([`Charges::recording`]); a kernel's scalar
//! result slots ([`Charges::result_slots`]). An embedder's ledger mints
//! charges from each store's identity alone, so no store's own entry is
//! within its reach there. Work whose plan needs no storage runs
//! through [`Charges::none`], and work that allocates in one store alone
//! (a granule pool's) through [`Charges::in_store`]: an allocation of any
//! byte through them in a store they name no entry of is an invariant
//! violation, which panics, never a refusal a caller could read as memory
//! pressure.
//! No allocation can be made that names no entry, so none is ever charged
//! by default or found without an owner.
//!
//! A record table over a view's records is device storage made through the
//! charges it is formed through, named where it is formed
//! (`crate::driver::Allocation::view_table`), as every other allocation
//! through them is: to their entry, in the kind they are confined to. An
//! allocation holds no entry of its own for tables formed later.
//!
//! An allocation of a contract lands in one of the kinds its placement
//! names, and lies along that kind's access path. Charges may confine the
//! allocations made through them to one kind ([`Charges::confined`]).

pub use seismic_memory::entry::{Backed, Carved, Change, Charge, Charging, DropScope, Entry, Refused, Store, Tally, Wiring};
use crate::api::device::DeviceInner;
pub use seismic_memory::{AllocationFailure, HostBlock, PageSize};
use seismic_compiler::errors::ExecutionError;
use seismic_native_target::process::{ProcessFootprint, ProcessId};
use seismic_topology::{AccessPathId, Contract, DeviceTopology, Executor, PhysicalIdentity, StoreId, StoreIdentity};
use std::collections::BTreeMap;
use std::sync::Arc;

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
    /// Its store's index in [`DeviceStores::stores`].
    store: usize,
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
    /// Each store's own entry ([`Store::own`]), in `stores` order, taken
    /// once as the device opens: what [`Charges`]' holderless purposes
    /// charge.
    own: Arc<[Arc<dyn Entry>]>,
    /// Whether any store is an embedder's ledger rather than a tally
    /// ([`Store::tally`]): [`Charges::tally`] is refused on such a device.
    embedded: bool,
    /// Whether the device was opened in caller-given stores
    /// ([`crate::devices::Devices::open_in`]), which a later open in the
    /// same stores shares ([`Self::opened_with`]).
    shared_in: bool,
}

impl DeviceStores {
    /// The stores `executor` of `topology` reaches, of `stores`, the
    /// embedder's ledger of each.
    ///
    /// # Panics
    /// Where `stores` lacks a store the executor places storage in, or holds
    /// one twice.
    pub(crate) fn new(topology: &DeviceTopology, executor: &Executor, stores: Vec<Arc<dyn Store>>, shared_in: bool) -> Arc<Self> {
        for (index, store) in stores.iter().enumerate() {
            assert!(
                stores[..index].iter().all(|other| other.identity() != store.identity()),
                "invariant violated: a device opened with {:?} twice",
                store.identity()
            );
        }
        let index = |id| {
            let identity = topology.store_identity(id).expect("a placement's store belongs to its topology");
            stores
                .iter()
                .position(|store| store.identity() == &identity)
                .unwrap_or_else(|| panic!("invariant violated: {} opened without a ledger of {identity:?}", executor.selector()))
        };
        let of = |id| stores[index(id)].clone();
        let native = |id| topology.kind(id).expect("a placement names kinds of its revision").native().id;
        let placed = Contract::ALL.map(|contract| {
            let path = *executor.paths(contract).first().expect("an opened executor places every contract");
            let store = topology.access_path(path).expect("a placement's path belongs to its topology").store();
            Placed { store: index(store), kinds: executor.placement(contract).iter().map(|id| native(*id)).collect() }
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
        let own = stores.iter().map(|store| store.own()).collect();
        let embedded = stores.iter().any(|store| !store.tally());
        Arc::new(Self { placed, kinds, host, stores, own, embedded, shared_in })
    }

    /// Whether a device opened in these may be shared by an open in
    /// caller-given `stores`: these were given by a caller too, of the same
    /// stores, never tallies of the device's own. The process has one
    /// embedder's ledger per store.
    pub(crate) fn opened_with(&self, stores: &[Arc<dyn Store>]) -> bool {
        self.shared_in
            && stores.len() == self.stores.len()
            && stores.iter().all(|store| self.stores.iter().any(|held| held.identity() == store.identity()))
    }

    fn placed(&self, contract: Contract) -> &Placed {
        &self.placed[Contract::ALL.iter().position(|of| *of == contract).expect("every contract is placed")]
    }

    /// The store `contract`'s storage lies in.
    pub(crate) fn store(&self, contract: Contract) -> Arc<dyn Store> {
        self.stores[self.placed(contract).store].clone()
    }

    /// The kinds `contract`'s placement allocates from, in order, as the
    /// backend names them.
    pub(crate) fn kinds(&self, contract: Contract) -> &[u32] {
        &self.placed(contract).kinds
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
}

/// What every allocation through a device is charged to: one entry of the
/// embedder's ledger in each store the device reaches, minted for the
/// device by that ledger ([`DeviceInner::charges`]), each store's own entry
/// for a named purpose no holder causes ([`Self::tally`],
/// [`Self::recording`], [`Self::result_slots`]), one entry of a single
/// store's ledger, for work that allocates in it alone ([`Self::in_store`]),
/// or none, for work that allocates nothing ([`Self::none`]). An allocation is
/// made only through charges, which carry their device, so every one names
/// its entry and its device by construction. Clones charge the same entries.
#[derive(Clone)]
pub struct Charges {
    device: Arc<DeviceInner>,
    entries: Entries,
    /// The kind allocations through these land in, where they are confined
    /// to one ([`Self::confined`]).
    confined: Option<u32>,
}

/// The entries charges name, by the stores they allocate in.
#[derive(Clone)]
enum Entries {
    /// One per store, in [`DeviceStores::stores`] order.
    Each(Arc<[Arc<dyn Entry>]>),
    /// One of the store at `store` in [`DeviceStores::stores`], none in
    /// every other ([`Charges::in_store`]).
    In { store: usize, entry: Arc<dyn Entry> },
    /// None in any store ([`Charges::none`]).
    None,
}

impl Entries {
    /// Their entry of the store at `store`, where they name one.
    fn of(&self, store: usize) -> Option<&Arc<dyn Entry>> {
        match self {
            Self::Each(entries) => Some(&entries[store]),
            Self::In { store: of, entry } => (*of == store).then_some(entry),
            Self::None => None,
        }
    }
}

impl Charges {
    /// Charges of `device`, each store's entry the one `entry_in` names for
    /// it: called once with the identity of each store the device reaches,
    /// and never handed the store, whose own entry ([`Store::own`]) is
    /// charged only for the purposes named here.
    ///
    /// # Panics
    /// Where `entry_in` names an entry of another store than it was given.
    pub(crate) fn mint(device: &Arc<DeviceInner>, mut entry_in: impl FnMut(&StoreIdentity) -> Arc<dyn Entry>) -> Self {
        let entries = device
            .memory
            .stores
            .iter()
            .map(|store| {
                let entry = entry_in(store.identity());
                assert_eq!(
                    entry.store(),
                    store.identity(),
                    "invariant violated: an entry of {:?} named for {:?}",
                    entry.store(),
                    store.identity()
                );
                entry
            })
            .collect();
        Self { device: device.clone(), entries: Entries::Each(entries), confined: None }
    }

    /// Charges of `device` in the store `contract`'s storage lies in alone,
    /// to `entry`, an entry of that store: what a granule pool of it is
    /// charged through ([`crate::pool::PoolCharges`]). An allocation of any
    /// byte through them in another store panics, as through
    /// [`Self::none`].
    ///
    /// # Panics
    /// Where `entry` is an entry of another store.
    pub fn in_store(device: &Arc<DeviceInner>, contract: Contract, entry: Arc<dyn Entry>) -> Self {
        let store = device.memory.placed(contract).store;
        let identity = device.memory.stores[store].identity();
        assert_eq!(entry.store(), identity, "invariant violated: an entry of {:?} named for {identity:?}", entry.store());
        Self { device: device.clone(), entries: Entries::In { store, entry }, confined: None }
    }

    /// Each of `device`'s stores' own entry ([`Store::own`]).
    fn own_entries(device: &Arc<DeviceInner>) -> Self {
        Self { device: device.clone(), entries: Entries::Each(device.memory.own.clone()), confined: None }
    }

    /// Each store's one entry of `device`, opened on tallies of its own
    /// (a test's, a tool's, a frontend with no ledger of its own): every
    /// allocation on it.
    ///
    /// # Panics
    /// Where `device` was opened in an embedder's ledger, whose holders
    /// each have an entry the ledger mints ([`DeviceInner::charges`]).
    pub fn tally(device: &Arc<DeviceInner>) -> Self {
        assert!(!device.memory.embedded, "invariant violated: tally charges of a device opened in an embedder's ledger");
        Self::own_entries(device)
    }

    /// Charges of `device` for work whose plan needs no storage: an
    /// allocation of any byte through them is an invariant violation, and
    /// panics. A request of none succeeds and charges no holder: what its
    /// backend takes for it (an empty buffer's rounding, its native object)
    /// is the device's own, as a driver object is.
    pub fn none(device: &Arc<DeviceInner>) -> Self {
        Self { device: device.clone(), entries: Entries::None, confined: None }
    }

    /// Each of `device`'s stores' own entry, for a CUDA graph's recording
    /// placeholders, which no holder causes.
    pub(crate) fn recording(device: &Arc<DeviceInner>) -> Self {
        Self::own_entries(device)
    }

    /// Each of `device`'s stores' own entry, for a kernel's scalar result
    /// slots, which every call of it shares and no holder causes.
    pub(crate) fn result_slots(device: &Arc<DeviceInner>) -> Self {
        Self::own_entries(device)
    }

    /// These charges, every allocation made through them landing in `kind`
    /// or refused, never moving on to another kind.
    ///
    /// # Panics
    /// Where `kind` is none of the device's executor's.
    pub fn confined(&self, kind: seismic_topology::NativeKind) -> Self {
        assert_eq!(kind.backend, self.device.info.backend(), "invariant violated: allocations confined to a kind of another backend");
        assert!(
            self.device.memory.kinds.iter().any(|of| of.id == kind.id),
            "invariant violated: allocations confined to kind {}, none of the executor's",
            kind.id
        );
        Self { confined: Some(kind.id), ..self.clone() }
    }

    /// The device allocations through these are made on.
    pub fn device(&self) -> &Arc<DeviceInner> {
        &self.device
    }

    /// These charges, which must be `device`'s: what is made for storage of
    /// `device` (a slot, a slab store's table, a view's record table, a
    /// call's or a workflow's storage) is charged through `device`'s
    /// charges alone.
    ///
    /// # Panics
    /// Where they are another device's.
    pub(crate) fn of_device(&self, device: &Arc<DeviceInner>) -> &Self {
        assert!(
            Arc::ptr_eq(&self.device, device),
            "invariant violated: storage of one device charged through another device's charges"
        );
        self
    }

    /// Their entry of the store `contract`'s storage lies in. None where
    /// they allocate nothing there ([`Self::none`], [`Self::in_store`]).
    pub fn entry(&self, contract: Contract) -> Option<&Arc<dyn Entry>> {
        self.entries.of(self.device.memory.placed(contract).store)
    }

    /// Their entry of `contract`'s store for a request of `bytes`: where
    /// they allocate nothing in it ([`Self::none`], [`Self::in_store`]),
    /// the store's own entry for a request of none.
    ///
    /// # Panics
    /// For a request of any byte in a store they allocate nothing in: work
    /// that runs through them was planned to need no storage there.
    fn charged(&self, contract: Contract, bytes: u64) -> &Arc<dyn Entry> {
        let store = self.device.memory.placed(contract).store;
        match (self.entries.of(store), bytes) {
            (Some(entry), _) => entry,
            (None, 0) => &self.device.memory.own[store],
            (None, bytes) => panic!(
                "invariant violated: an allocation of {bytes} bytes of {contract:?} through charges that allocate nothing in its store"
            ),
        }
    }

    /// How an allocation of `contract` through these is charged: to their
    /// entry of its store, in the kinds of its placement or the one they are
    /// confined to (refused where the placement does not name it).
    ///
    /// # Panics
    /// For any byte in a store they allocate nothing in ([`Self::none`],
    /// [`Self::in_store`]).
    pub(crate) fn of(&self, contract: Contract, bytes: u64) -> Result<AllocationCharge<'_>, ExecutionError> {
        let entry = self.charged(contract, bytes);
        let memory = &self.device.memory;
        let kinds = memory.kinds(contract);
        let kinds = match self.confined {
            None => kinds,
            Some(confined) => kinds.iter().find(|kind| **kind == confined).map(std::slice::from_ref).ok_or_else(|| {
                ExecutionError::AllocationFailed(format!("an allocation confined to kind {confined} lies in kinds {kinds:?}"))
            })?,
        };
        Ok(AllocationCharge { entry, kinds })
    }
}

/// How one allocation is charged as its backend makes it: the entry its
/// bytes are charged to and the kinds it may land in (it lands in the first
/// with room).
pub(crate) struct AllocationCharge<'a> {
    pub(crate) entry: &'a Arc<dyn Entry>,
    pub(crate) kinds: &'a [u32],
}

impl Charges {
    /// A fixed buffer of `contract` through these
    /// ([`crate::api::device::DeviceInner::allocate_contract`]), ordered by
    /// `access`'s hazard domain where given, else by a domain of its own.
    pub(crate) fn allocate_contract(
        &self,
        contract: Contract,
        bytes: u64,
        alignment: u64,
        access: Option<Arc<crate::driver::AccessGroup>>,
    ) -> Result<Arc<crate::driver::Allocation>, ExecutionError> {
        self.device.allocate_contract(self.of(contract, bytes)?, bytes, alignment, access)
    }

    /// Device storage through these, in `access`'s hazard domain where given.
    pub(crate) fn allocate_in(
        &self,
        bytes: u64,
        alignment: u64,
        access: Option<Arc<crate::driver::AccessGroup>>,
    ) -> Result<Arc<crate::driver::Allocation>, ExecutionError> {
        self.allocate_contract(Contract::DeviceOnly, bytes, alignment, access)
    }

    /// Device storage through these, in a hazard domain of its own.
    pub(crate) fn allocate(&self, bytes: u64, alignment: u64) -> Result<Arc<crate::driver::Allocation>, ExecutionError> {
        self.allocate_in(bytes, alignment, None)
    }

    /// Device storage through these that reads zero, backed
    /// ([`crate::backends::OpenedKind::zeroed`]).
    pub(crate) fn allocate_zeroed(&self, bytes: u64, alignment: u64) -> Result<Arc<crate::driver::Allocation>, ExecutionError> {
        self.device.kind.zeroed(self.allocate(bytes, alignment)?, bytes)
    }

    /// A fixed buffer of `contract`, which the host reaches in place,
    /// through these. CPU and Metal have one kind.
    pub(crate) fn allocate_mapped(
        &self,
        contract: Contract,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<crate::driver::Allocation>, ExecutionError> {
        self.allocate_contract(contract, bytes, alignment, None)
    }

    /// A [`Self::allocate_mapped`] buffer that reads zero, backed.
    pub(crate) fn allocate_mapped_zeroed(
        &self,
        contract: Contract,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<crate::driver::Allocation>, ExecutionError> {
        self.device.kind.zeroed(self.allocate_mapped(contract, bytes, alignment)?, bytes)
    }
}
