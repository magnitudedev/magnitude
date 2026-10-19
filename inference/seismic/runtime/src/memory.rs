//! Allocation lifetime accounting over an executor's ordered access paths
//! (spec §2.7).
//!
//! One [`StoreLedger`] exists per store in this process: every opened
//! executor whose access paths reach the store charges it, so host RAM is
//! counted once whoever charges it. An executor has one account per access
//! path, held to the path's grant, and one [`MemoryDomain`] per storage kind,
//! which spends its placement's paths in order and adds the device's
//! enforceable limit. A domain has two views of one state: the logical view
//! charges allocations (tensors, ranges of blocks), the physical view
//! charges what the backend holds natively (blocks, granules, driver
//! objects). A native block owns the ranges suballocated from it, so a
//! path's claim is the larger of its two charges, never their sum. This is
//! process accounting, not an OS-wide reservation.

use seismic_topology::{
    AccessPath, AccessPathId, AccessRoute, DeviceTopology, Executor, ExecutorSelector, StorageKind,
    StoreId, StoreIdentity,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

/// Seismic-owned charges and enforced limits. This is not global driver or
/// OS usage: other processes and non-Seismic allocations are absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryUsage {
    /// Bytes charged by allocations made through this opened device.
    pub charged: u64,
    /// Device memory this opened device holds: `charged` plus what its
    /// backend holds beyond its allocations (the free space of the blocks it
    /// sub-allocates from, and driver objects such as instantiated CUDA
    /// graphs). What the device's memory store actually loses to Seismic.
    pub held: u64,
    /// The part of `held` beyond `charged` that later allocations take
    /// without the device holding more: the free space of its blocks.
    pub unallocated: u64,
    /// The device's enforced allocation limit.
    pub limit: Option<u64>,
    /// Bytes charged by every opened device in this process to the stores
    /// this device's placement reaches, each store once, including
    /// `charged`.
    pub store_charged: u64,
    /// The most `charged` has been since the device was opened or its mark
    /// was last restarted ([`MemoryDomain::restart_high_water`]): what was
    /// charged and released again in between is seen here and nowhere else.
    pub high_water: u64,
    /// The most the device's charges would have been, over the same span as
    /// `high_water`, had it granted a reservation it refused: what was
    /// charged then and what was asked for. None when nothing was refused in
    /// that span.
    pub refused_demand: Option<u64>,
}

/// What one store's ledger has charged, and the limit it is held to.
struct StoreCharges {
    charged: u64,
    limit: Option<u64>,
}

/// Process-wide charges against one store.
pub(crate) struct StoreLedger {
    charges: Mutex<StoreCharges>,
}

impl StoreLedger {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            charges: Mutex::new(StoreCharges {
                charged: 0,
                limit: None,
            }),
        })
    }

    fn charges(&self) -> MutexGuard<'_, StoreCharges> {
        self.charges.lock().expect("store ledger lock poisoned")
    }

    /// Bytes charged to the store.
    pub(crate) fn bytes(&self) -> u64 {
        self.charges().charged
    }

    /// Hold the store's charges to `limit`, on top of its capacity.
    pub(crate) fn set_limit(&self, limit: Option<u64>) {
        self.charges().limit = limit;
    }

    /// What the store can still take within `capacity` and its limit.
    fn room(&self, capacity: u64) -> u64 {
        self.charges().room(capacity)
    }
}

impl StoreCharges {
    fn room(&self, capacity: u64) -> u64 {
        capacity
            .min(self.limit.unwrap_or(u64::MAX))
            .saturating_sub(self.charged)
    }
}

/// The process's ledger of `topology`'s store `id`: one per store identity,
/// shared by every executor and revision that reaches the store.
pub(crate) fn store_ledger(topology: &DeviceTopology, id: StoreId) -> Arc<StoreLedger> {
    static LEDGERS: OnceLock<Mutex<HashMap<StoreIdentity, Weak<StoreLedger>>>> = OnceLock::new();
    let identity = topology
        .store_identity(id)
        .expect("store belongs to topology");
    let mut ledgers = LEDGERS
        .get_or_init(Mutex::default)
        .lock()
        .expect("store registry lock poisoned");
    if let Some(ledger) = ledgers.get(&identity).and_then(Weak::upgrade) {
        return ledger;
    }
    let ledger = StoreLedger::new();
    ledgers.insert(identity, Arc::downgrade(&ledger));
    ledger
}

/// One scope's charges: logical (`charged`) and native (`held`), whose
/// larger is the scope's claim on its grant and store.
struct Account {
    charged: u64,
    held: u64,
    limit: Option<u64>,
}

impl Account {
    fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self {
            charged: 0,
            held: 0,
            limit: None,
        }))
    }

    fn claim(&self) -> u64 {
        self.charged.max(self.held)
    }

    fn own(&mut self, physical: bool) -> &mut u64 {
        if physical {
            &mut self.held
        } else {
            &mut self.charged
        }
    }
}

/// One accounting scope: its account, held to a grant and its own limit,
/// charging its store's ledger. An access path's scope ([`AccessScope`]) is
/// held to the path's grant, and device and upload placements that name one
/// path share it; the backend's staging has a scope of its own, along no
/// path. Lock order is always the scope's account, then the store's ledger.
pub(crate) struct Scope {
    store: Arc<StoreLedger>,
    /// The store's capacity.
    capacity: u64,
    grant: Option<u64>,
    account: Arc<Mutex<Account>>,
}

/// One of the topology's access paths of an opened executor, with its
/// scope.
pub(crate) struct AccessScope {
    pub(crate) id: AccessPathId,
    pub(crate) scope: Arc<Scope>,
}

impl Scope {
    fn account(&self) -> MutexGuard<'_, Account> {
        self.account.lock().expect("path charge lock poisoned")
    }

    /// The scope's claim: the larger of its logical and native charges.
    pub(crate) fn charged(&self) -> u64 {
        self.account().claim()
    }

    /// Hold the scope's claim to `limit`, on top of its grant.
    pub(crate) fn set_limit(&self, limit: Option<u64>) {
        self.account().limit = limit;
    }

    /// What the scope's claim can still grow by, within its grant, its limit
    /// and `store_room`.
    fn growth(account: &Account, grant: Option<u64>, store_room: u64) -> u64 {
        grant
            .unwrap_or(u64::MAX)
            .min(account.limit.unwrap_or(u64::MAX))
            .saturating_sub(account.claim())
            .min(store_room)
    }

    /// Charge up to `requested` bytes to the view (`physical`) and the
    /// store; what was charged. A view's charge within the other view's
    /// larger claim takes no grant or store room.
    fn reserve(&self, requested: u64, physical: bool) -> u64 {
        let mut account = self.account();
        let mut store = self.store.charges();
        let prior = account.claim();
        let within = prior - *account.own(physical);
        let growth = Self::growth(&account, self.grant, store.room(self.capacity));
        let bytes = requested.min(within.saturating_add(growth));
        *account.own(physical) += bytes;
        store.charged += account.claim() - prior;
        bytes
    }

    /// Release `bytes` the view charged.
    fn release(&self, bytes: u64, physical: bool) {
        let mut account = self.account();
        let prior = account.claim();
        let own = account.own(physical);
        *own = own
            .checked_sub(bytes)
            .expect("released more than the scope owns");
        self.store.charges().charged -= prior - account.claim();
    }

    /// Charge again `bytes` the view held before, never refused.
    fn restore(&self, bytes: u64, physical: bool) {
        let mut account = self.account();
        let prior = account.claim();
        let own = account.own(physical);
        *own = own
            .checked_add(bytes)
            .expect("restored scope charge overflow");
        self.store.charges().charged += account.claim() - prior;
    }
}

/// An opened executor's accounting: a domain per storage kind over its
/// placement, every access path's scope, and the domain its backend's
/// transfer staging charges in host RAM.
pub(crate) struct ExecutorMemory {
    pub(crate) device: Arc<MemoryDomain>,
    pub(crate) upload: Arc<MemoryDomain>,
    pub(crate) paths: Vec<AccessScope>,
    pub(crate) staging: Arc<MemoryDomain>,
}

/// What an account is for: an access path along a route, or the backend's
/// staging.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum AccountOf {
    Access(AccessRoute),
    Staging,
}

/// An account's key. An executor has one access path per store and route,
/// so the two name it across revisions; its account outlives one revision's
/// identifiers.
type AccountKey = (ExecutorSelector, StoreIdentity, AccountOf);

/// The accounts opened executors hold, by key.
fn accounts() -> MutexGuard<'static, HashMap<AccountKey, Weak<Mutex<Account>>>> {
    static ACCOUNTS: OnceLock<Mutex<HashMap<AccountKey, Weak<Mutex<Account>>>>> = OnceLock::new();
    ACCOUNTS
        .get_or_init(Mutex::default)
        .lock()
        .expect("account registry lock poisoned")
}

/// The key of `executor`'s account for `of` on `topology`'s store `store`.
fn account_key(
    topology: &DeviceTopology,
    executor: &Executor,
    store: StoreId,
    of: AccountOf,
) -> AccountKey {
    let identity = topology
        .store_identity(store)
        .expect("path store belongs to topology");
    (executor.selector(), identity, of)
}

/// What `executor`'s scope along its access path `path` claims now: none
/// while no opened executor holds that scope.
pub(crate) fn path_claim(topology: &DeviceTopology, executor: &Executor, path: &AccessPath) -> u64 {
    let key = account_key(
        topology,
        executor,
        path.store(),
        AccountOf::Access(path.route()),
    );
    accounts()
        .get(&key)
        .and_then(Weak::upgrade)
        .map_or(0, |account| {
            account.lock().expect("path charge lock poisoned").claim()
        })
}

impl ExecutorMemory {
    pub(crate) fn new(topology: &DeviceTopology, executor: &Executor) -> Arc<Self> {
        let mut accounts = accounts();
        let mut scope = |store: StoreId, grant, of| {
            let key = account_key(topology, executor, store, of);
            let account = accounts
                .get(&key)
                .and_then(Weak::upgrade)
                .unwrap_or_else(|| {
                    let account = Account::new();
                    accounts.insert(key, Arc::downgrade(&account));
                    account
                });
            Arc::new(Scope {
                store: store_ledger(topology, store),
                capacity: topology
                    .store(store)
                    .expect("path store belongs to topology")
                    .capacity()
                    .value,
                grant,
                account,
            })
        };
        let paths = topology
            .access_paths()
            .iter()
            .filter(|path| path.executor() == executor.id())
            .map(|path| AccessScope {
                id: path.id(),
                scope: scope(
                    path.store(),
                    path.grant().map(|grant| grant.value),
                    AccountOf::Access(path.route()),
                ),
            })
            .collect::<Vec<_>>();
        let staging = MemoryDomain::placement(vec![scope(
            topology.host_ram().id(),
            None,
            AccountOf::Staging,
        )]);
        let domain = |kind| {
            MemoryDomain::placement(
                executor
                    .placement(kind)
                    .iter()
                    .map(|id| {
                        paths
                            .iter()
                            .find(|path| path.id == *id)
                            .expect("placement path belongs to executor")
                            .scope
                            .clone()
                    })
                    .collect(),
            )
        };
        Arc::new(Self {
            device: domain(StorageKind::Device),
            upload: domain(StorageKind::Upload),
            staging,
            paths,
        })
    }

    /// The scope of this executor's access path `id`.
    pub(crate) fn path(&self, id: AccessPathId) -> Option<&AccessScope> {
        self.paths.iter().find(|path| path.id == id)
    }

    /// What `charge`, an allocation of this executor, holds on each of its
    /// access paths.
    pub(crate) fn path_charges(&self, charge: &MemoryCharge) -> Vec<(AccessPathId, u64)> {
        charge
            .portions
            .iter()
            .map(|(scope, bytes)| {
                let path = self
                    .paths
                    .iter()
                    .find(|path| Arc::ptr_eq(&path.scope, scope))
                    .expect("an executor's allocation is charged along its own paths");
                (path.id, *bytes)
            })
            .collect()
    }
}

/// The bytes `charges` along `topology`'s access paths hold on each store,
/// each store once.
pub(crate) fn store_charges(
    topology: &DeviceTopology,
    charges: impl IntoIterator<Item = (AccessPathId, u64)>,
) -> HashMap<StoreId, u64> {
    let mut stores = HashMap::new();
    for (path, bytes) in charges {
        let store = topology
            .access_path(path)
            .expect("charged path belongs to topology")
            .store();
        *stores.entry(store).or_default() += bytes;
    }
    stores
}

/// A domain's view as the backend's account for its native memory.
pub(crate) struct MemoryAccount(pub(crate) Arc<MemoryDomain>);

impl seismic_compiler::executable::MemoryAccount for MemoryAccount {
    fn available(&self) -> u64 {
        self.0.available()
    }

    /// Charge memory the driver has already taken: along the placement,
    /// each path up to what its grant has left and the last the rest, never
    /// refused.
    fn record(&self, bytes: u64) -> Box<dyn seismic_compiler::executable::MemoryHold> {
        let mut portions = Vec::new();
        let mut left = bytes;
        let paths = &self.0.paths;
        for (index, path) in paths.iter().enumerate() {
            let amount = if index + 1 == paths.len() {
                left
            } else {
                left.min(
                    path.grant
                        .unwrap_or(u64::MAX)
                        .saturating_sub(path.charged()),
                )
            };
            if amount != 0 {
                portions.push((path.clone(), amount));
            }
            left -= amount;
        }
        assert_eq!(left, 0, "physical allocation has a placement");
        self.0.restore(&portions);
        Box::new(MemoryReservation {
            domain: self.0.clone(),
            portions,
            remaining: bytes,
        })
    }

    fn reserve(
        &self,
        bytes: u64,
    ) -> Result<
        Box<dyn seismic_compiler::executable::MemoryHold>,
        seismic_compiler::errors::ExecutionError,
    > {
        MemoryDomain::reserve(&self.0, bytes)
            .map(|hold| Box::new(hold) as Box<dyn seismic_compiler::executable::MemoryHold>)
            .map_err(
                |capacity| seismic_compiler::errors::ExecutionError::AllocationCapacity {
                    required: capacity.required.into(),
                    available: capacity.available,
                },
            )
    }
}

/// One domain's charges, shared by its logical and physical views.
struct MemoryState {
    /// What the logical view charged.
    charged: u64,
    /// What the physical view charged.
    held: u64,
    limit: Option<u64>,
    high_water: u64,
    refused_demand: Option<u64>,
}

impl MemoryState {
    fn own(&mut self, physical: bool) -> &mut u64 {
        if physical {
            &mut self.held
        } else {
            &mut self.charged
        }
    }
}

/// One storage kind's accounting scope of an opened executor: its
/// placement's paths, spent in order, under the device's enforceable limit.
/// A domain is one view (logical or physical) of a state its other view
/// shares, so every charge and refusal of either is in its usage. Lock order
/// is always the domain, then its paths.
pub(crate) struct MemoryDomain {
    paths: Vec<Arc<Scope>>,
    state: Arc<Mutex<MemoryState>>,
    physical: bool,
}

type Portions = Vec<(Arc<Scope>, u64)>;

impl MemoryDomain {
    /// A domain of one unlimited scope on the store `ledger` keeps.
    #[cfg(test)]
    pub(crate) fn new(ledger: Arc<StoreLedger>) -> Arc<Self> {
        Self::placement(vec![Arc::new(Scope {
            store: ledger,
            capacity: u64::MAX,
            grant: None,
            account: Account::new(),
        })])
    }

    /// The logical view of a domain over `paths`, spent in order.
    pub(crate) fn placement(paths: Vec<Arc<Scope>>) -> Arc<Self> {
        Arc::new(Self {
            paths,
            state: Arc::new(Mutex::new(MemoryState {
                charged: 0,
                held: 0,
                limit: None,
                high_water: 0,
                refused_demand: None,
            })),
            physical: false,
        })
    }

    /// The physical view of this domain: what the backend holds natively.
    pub(crate) fn physical(&self) -> Arc<Self> {
        Arc::new(Self {
            paths: self.paths.clone(),
            state: self.state.clone(),
            physical: true,
        })
    }

    /// The physical view of this domain on `path` alone, for a backend that
    /// allocates from one path's memory directly (a Vulkan heap).
    #[cfg(not(target_os = "macos"))]
    pub(crate) fn physical_on(&self, path: &Arc<Scope>) -> Arc<Self> {
        assert!(
            self.paths.iter().any(|own| Arc::ptr_eq(own, path)),
            "a domain's path view is of one of its paths"
        );
        Arc::new(Self {
            paths: vec![path.clone()],
            state: self.state.clone(),
            physical: true,
        })
    }

    /// The claims of the domain's paths together.
    pub(crate) fn scope_charge(&self) -> u64 {
        self.paths.iter().map(|path| path.charged()).sum()
    }

    fn state(&self) -> MutexGuard<'_, MemoryState> {
        self.state.lock().expect("memory scope lock poisoned")
    }

    /// Start the high-water mark again from what is charged now, and forget
    /// the refusals before, so that the next reading is of the work between
    /// the two.
    pub(crate) fn restart_high_water(&self) {
        let mut state = self.state();
        state.high_water = state.charged;
        state.refused_demand = None;
    }

    pub(crate) fn usage(&self) -> MemoryUsage {
        let state = self.state();
        let mut stores: Vec<*const StoreLedger> = Vec::new();
        let mut store_charged = 0;
        for path in &self.paths {
            let store = Arc::as_ptr(&path.store);
            if !stores.contains(&store) {
                stores.push(store);
                store_charged += path.store.bytes();
            }
        }
        MemoryUsage {
            charged: state.charged,
            held: state.charged,
            unallocated: 0,
            limit: state.limit,
            store_charged,
            high_water: state.high_water,
            refused_demand: state.refused_demand,
        }
    }

    /// Hold both views to `limit`. A falling limit cannot revoke what is
    /// held: every further reservation is refused until releases bring the
    /// charge back below it.
    pub(crate) fn set_limit(&self, limit: Option<u64>) {
        self.state().limit = limit;
    }

    /// What this view can still charge: the growth of each path within its
    /// grant and its store (a store reached by several paths counted once),
    /// with what the other view's larger claims leave to it, within the
    /// domain's limit.
    pub(crate) fn available(&self) -> u64 {
        let mut state = self.state();
        let mut stores = HashMap::new();
        let mut available = 0u64;
        for path in &self.paths {
            let mut account = path.account();
            let within = account.claim() - *account.own(self.physical);
            let store_room = stores
                .entry(Arc::as_ptr(&path.store))
                .or_insert_with(|| path.store.room(path.capacity));
            let growth = Scope::growth(&account, path.grant, *store_room);
            *store_room -= growth;
            available = available.saturating_add(growth).saturating_add(within);
        }
        let limit = state.limit.unwrap_or(u64::MAX);
        available.min(limit.saturating_sub(*state.own(self.physical)))
    }

    /// Charge `bytes` along the placement, or refuse them whole: the
    /// domain's limit, then each path in order up to what it can take.
    pub(crate) fn reserve(
        self: &Arc<Self>,
        bytes: u64,
    ) -> Result<MemoryReservation, MemoryCapacity> {
        let mut state = self.state();
        let own = *state.own(self.physical);
        let next = own.checked_add(bytes).ok_or(MemoryCapacity {
            required: bytes,
            available: 0,
        })?;
        if let Some(limit) = state.limit {
            if next > limit {
                state.refused_demand = state.refused_demand.max(Some(next));
                return Err(MemoryCapacity {
                    required: bytes,
                    available: limit.saturating_sub(own),
                });
            }
        }
        let mut remaining = bytes;
        let mut portions = Vec::new();
        for path in &self.paths {
            let amount = path.reserve(remaining, self.physical);
            if amount != 0 {
                portions.push((path.clone(), amount));
                remaining -= amount;
            }
            if remaining == 0 {
                break;
            }
        }
        if remaining != 0 {
            for (path, amount) in portions {
                path.release(amount, self.physical);
            }
            state.refused_demand = state.refused_demand.max(Some(next));
            return Err(MemoryCapacity {
                required: bytes,
                available: bytes - remaining,
            });
        }
        *state.own(self.physical) = next;
        if !self.physical {
            state.high_water = state.high_water.max(next);
        }
        Ok(MemoryReservation {
            domain: self.clone(),
            portions,
            remaining: bytes,
        })
    }

    fn release(&self, portions: &Portions) {
        let mut state = self.state();
        for (path, bytes) in portions {
            path.release(*bytes, self.physical);
            let own = state.own(self.physical);
            *own = own
                .checked_sub(*bytes)
                .expect("released more than the domain owns");
        }
    }

    /// Charge again what was held before. This is the return of a released
    /// part of a live allocation, or memory a driver already took, not a
    /// reservation, so it is never refused: like a falling limit, it can
    /// leave the domain above its limit.
    fn restore(&self, portions: &Portions) {
        let mut state = self.state();
        for (path, bytes) in portions {
            path.restore(*bytes, self.physical);
            let own = state.own(self.physical);
            *own = own
                .checked_add(*bytes)
                .expect("restored domain charge overflow");
        }
        if !self.physical {
            state.high_water = state.high_water.max(state.charged);
        }
    }
}

/// A refused reservation: `required` bytes, of which `available` could be
/// charged.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MemoryCapacity {
    pub(crate) required: u64,
    pub(crate) available: u64,
}

/// Remove a prefix of `bytes` from `portions`, preserving the exact paths
/// charged at admission.
fn take(portions: &mut Portions, mut bytes: u64) -> Portions {
    let mut taken = Vec::new();
    while bytes != 0 {
        let (path, available) = portions
            .first_mut()
            .expect("charge partition covers its byte count");
        let amount = bytes.min(*available);
        taken.push((path.clone(), amount));
        *available -= amount;
        bytes -= amount;
        if *available == 0 {
            portions.remove(0);
        }
    }
    taken
}

/// Append `extra` to `portions`, merging a run on the same path.
fn append(portions: &mut Portions, extra: Portions) {
    for (path, bytes) in extra {
        if let Some((last, held)) = portions.last_mut() {
            if Arc::ptr_eq(last, &path) {
                *held += bytes;
                continue;
            }
        }
        portions.push((path, bytes));
    }
}

/// One atomic reservation for a complete invocation. It is split into the
/// allocation-owned charges as physical allocations are created; any unused
/// tail is released on drop. This prevents concurrent allocations from
/// invalidating a successful invocation-capacity check.
pub(crate) struct MemoryReservation {
    domain: Arc<MemoryDomain>,
    portions: Portions,
    remaining: u64,
}

impl MemoryReservation {
    /// Reserved bytes not yet taken by an allocation.
    pub(crate) fn remaining(&self) -> u64 {
        self.remaining
    }

    /// The charge of one allocation of `bytes`, taken from the reservation.
    pub(crate) fn take(&mut self, bytes: u64) -> MemoryCharge {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .expect("allocation exceeds reservation");
        MemoryCharge {
            domain: self.domain.clone(),
            portions: take(&mut self.portions, bytes),
            released: Vec::new(),
            bytes,
        }
    }

    /// Transfer an already reserved increase into the charge for one
    /// physical allocation whose backing is being resized in place.
    pub(crate) fn extend(&mut self, charge: &mut MemoryCharge, bytes: u64) {
        assert!(Arc::ptr_eq(&self.domain, &charge.domain));
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .expect("extension exceeds reservation");
        charge.bytes = charge.bytes.checked_add(bytes).expect("charge overflow");
        charge.released.clear();
        append(&mut charge.portions, take(&mut self.portions, bytes));
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.domain.release(&self.portions);
    }
}

/// One allocation's charge, released when the allocation drops.
pub(crate) struct MemoryCharge {
    domain: Arc<MemoryDomain>,
    portions: Portions,
    /// What the last shrink released, in path order, which a restore takes
    /// back first.
    released: Portions,
    bytes: u64,
}

impl MemoryCharge {
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Keep only the first `bytes` of the charge, releasing the rest.
    pub(crate) fn shrink(&mut self, bytes: u64) {
        assert!(bytes <= self.bytes);
        let retained = take(&mut self.portions, bytes);
        self.domain.release(&self.portions);
        append(&mut self.portions, std::mem::take(&mut self.released));
        self.released = std::mem::replace(&mut self.portions, retained);
        self.bytes = bytes;
    }

    /// Make the charge `bytes` again: a shrink below the current charge, or
    /// the return of what the last shrink released, never refused.
    pub(crate) fn restore(&mut self, bytes: u64) {
        if bytes <= self.bytes {
            self.shrink(bytes);
        } else {
            let restored = take(&mut self.released, bytes - self.bytes);
            self.domain.restore(&restored);
            append(&mut self.portions, restored);
            self.bytes = bytes;
        }
    }
}

impl Drop for MemoryCharge {
    fn drop(&mut self) {
        self.domain.release(&self.portions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(store: Arc<StoreLedger>, capacity: u64, grant: u64) -> Arc<Scope> {
        Arc::new(Scope {
            store,
            capacity,
            grant: Some(grant),
            account: Account::new(),
        })
    }

    #[test]
    fn native_blocks_and_logical_ranges_share_capacity_without_double_charge() {
        let store = StoreLedger::new();
        store.set_limit(Some(16));
        let logical = MemoryDomain::placement(vec![path(store.clone(), 16, 16)]);
        let physical = logical.physical();
        let range = logical.reserve(4).unwrap();
        let block = physical.reserve(12).unwrap();
        assert_eq!(store.bytes(), 12);
        let second_range = logical.reserve(8).unwrap();
        assert_eq!(store.bytes(), 12);
        let other = MemoryDomain::new(store.clone()).reserve(4).unwrap();
        assert!(physical.reserve(1).is_err());
        drop(range);
        drop(second_range);
        assert_eq!(store.bytes(), 16);
        drop(block);
        assert_eq!(store.bytes(), 4);
        drop(other);
        assert_eq!(store.bytes(), 0);
    }

    #[test]
    fn measured_native_overhead_consumes_shared_admission_until_released() {
        use seismic_compiler::executable::MemoryAccount as _;
        let store = StoreLedger::new();
        store.set_limit(Some(10));
        let logical = MemoryDomain::placement(vec![path(store.clone(), 10, 10)]);
        let physical = logical.physical();
        let block = physical.reserve(8).unwrap();
        let overhead = MemoryAccount(physical).record(4);
        assert_eq!(store.bytes(), 12);
        assert!(MemoryDomain::new(store.clone()).reserve(1).is_err());
        drop(overhead);
        assert!(MemoryDomain::new(store.clone()).reserve(2).is_ok());
        drop(block);
        assert_eq!(store.bytes(), 0);
    }

    /// The physical view is the device domain's: its refusals are in the
    /// device's usage, and its limit bounds what the backend holds.
    #[test]
    fn the_physical_view_shares_the_domains_limit_and_refusals() {
        use seismic_compiler::executable::MemoryAccount as _;
        let logical = MemoryDomain::placement(vec![path(StoreLedger::new(), 100, 100)]);
        logical.set_limit(Some(10));
        let physical = logical.physical();
        let block = physical.reserve(8).unwrap();
        assert_eq!(MemoryAccount(physical.clone()).available(), 2);
        assert!(physical.reserve(4).is_err());
        assert_eq!(logical.usage().refused_demand, Some(12));
        assert_eq!(logical.usage().charged, 0);
        drop(block);
        assert_eq!(MemoryAccount(physical).available(), 10);
    }

    #[test]
    fn placements_share_grants_and_stores_and_release_the_paths_they_reserved() {
        let first_store = StoreLedger::new();
        let second_store = StoreLedger::new();
        let first = path(first_store.clone(), 20, 6);
        let second = path(second_store.clone(), 20, 8);
        let device = MemoryDomain::placement(vec![first.clone(), second.clone()]);
        let upload = MemoryDomain::placement(vec![first.clone(), second.clone()]);
        let mut reservation = device.reserve(10).unwrap();
        assert_eq!((first.charged(), second.charged()), (6, 4));
        let mut charge = reservation.take(8);
        drop(reservation);
        assert_eq!((first_store.bytes(), second_store.bytes()), (6, 2));
        let shared = upload.reserve(6).unwrap();
        assert!(upload.reserve(1).is_err());
        charge.shrink(4);
        assert_eq!((first_store.bytes(), second_store.bytes()), (4, 6));
        charge.restore(8);
        assert_eq!((first_store.bytes(), second_store.bytes()), (6, 8));
        drop(shared);
        drop(charge);
        assert_eq!((first_store.bytes(), second_store.bytes()), (0, 0));
    }

    #[test]
    fn different_paths_cannot_spend_one_stores_capacity_twice() {
        let store = StoreLedger::new();
        let domain =
            MemoryDomain::placement(vec![path(store.clone(), 10, 8), path(store.clone(), 10, 8)]);
        assert!(domain.reserve(11).is_err());
        assert_eq!(store.bytes(), 0);
        let held = domain.reserve(10).unwrap();
        assert_eq!(domain.usage().store_charged, 10);
        drop(held);
        assert_eq!(store.bytes(), 0);
    }

    #[test]
    fn live_limits_are_shared_across_placements_and_executor_paths() {
        let store = StoreLedger::new();
        let a = path(store.clone(), 100, 80);
        let b = path(store.clone(), 100, 80);
        let first = MemoryDomain::placement(vec![a.clone()]);
        let second = MemoryDomain::placement(vec![b]);
        store.set_limit(Some(10));
        a.set_limit(Some(7));
        let held = first.reserve(6).unwrap();
        assert!(first.reserve(2).is_err());
        let other = second.reserve(4).unwrap();
        assert!(first.reserve(1).is_err());
        assert!(second.reserve(1).is_err());
        store.set_limit(Some(5));
        drop(other);
        assert!(second.reserve(1).is_err());
        drop(held);
        assert!(second.reserve(5).is_ok());
    }

    #[test]
    fn backend_staging_holds_share_the_host_ledger_until_native_release() {
        let store = StoreLedger::new();
        store.set_limit(Some(10));
        let tensors = MemoryDomain::new(store.clone());
        let staging = MemoryDomain::new(store.clone());
        let tensor = tensors.reserve(6).unwrap();
        let held = seismic_compiler::executable::MemoryAccount::reserve(&MemoryAccount(staging), 4)
            .unwrap();
        assert_eq!(store.bytes(), 10);
        assert!(tensors.reserve(1).is_err());
        drop(held);
        assert_eq!(store.bytes(), 6);
        assert!(tensors.reserve(4).is_ok());
        drop(tensor);
        assert_eq!(store.bytes(), 0);
    }

    fn domain() -> Arc<MemoryDomain> {
        MemoryDomain::new(StoreLedger::new())
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
                    assert_eq!(domain.usage().store_charged, 0);
                    continue;
                }
                for transferred in 0..=requested {
                    // Each iteration replays the production reserve, take
                    // and drop operations.
                    let domain = self::domain();
                    domain.set_limit(Some(limit));
                    let mut reservation = domain.reserve(requested).unwrap();
                    let charge = reservation.take(transferred);
                    assert_eq!(domain.usage().charged, requested);
                    drop(reservation);
                    assert_eq!(domain.usage().charged, transferred);
                    assert_eq!(domain.usage().store_charged, transferred);
                    drop(charge);
                    assert_eq!(domain.usage().charged, 0);
                    assert_eq!(domain.usage().store_charged, 0);
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
        assert_eq!(domain.usage().store_charged, 28);
        assert_eq!(domain.usage().high_water, 28);
        assert!(domain.reserve(1).is_err());
        drop(other);
        assert_eq!(domain.usage().charged, 16);
        // Restoring to fewer bytes releases the difference.
        charge.restore(8);
        assert_eq!(domain.usage().charged, 8);
        drop(charge);
        assert_eq!(domain.usage().charged, 0);
        assert_eq!(domain.usage().store_charged, 0);
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
    fn devices_sharing_a_store_charge_it_once_each_under_their_own_limits() {
        let store = StoreLedger::new();
        let metal = MemoryDomain::new(store.clone());
        let cpu = MemoryDomain::new(store.clone());
        metal.set_limit(Some(8));
        let mut gpu = metal.reserve(6).unwrap();
        let _gpu = gpu.take(6);
        let host = cpu.reserve(5).unwrap();
        assert_eq!(metal.usage().charged, 6);
        assert_eq!(cpu.usage().charged, 5);
        assert_eq!(metal.usage().store_charged, 11);
        assert_eq!(cpu.usage().store_charged, 11);
        // The device limit constrains only this device's own charges.
        assert!(metal.reserve(3).is_err());
        drop(host);
        assert_eq!(metal.usage().store_charged, 6);
        // A separate store is a separate ledger.
        let dedicated = MemoryDomain::new(StoreLedger::new());
        let _dedicated = dedicated.reserve(4).unwrap();
        assert_eq!(metal.usage().store_charged, 6);
        assert_eq!(dedicated.usage().store_charged, 4);
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
