//! The ledger: one
//! hierarchy of charges over the stores of this process.
//!
//! - A [`StoreLedger`] per store, shared by every executor and revision that
//!   reaches the store, so host RAM is counted once whoever charges it.
//! - An [`Account`] per access path of an executor, shared by every open of
//!   that executor: what this process charges along the path, charging its
//!   store.
//! - A [`DeviceLedger`] per store an opened device reaches: that device's
//!   own charges there, under its own bound.
//! - A [`Share`]: one opened device's part of one path's account, which its
//!   device ledger and the account both count, held to the grant and the
//!   store capacity the open's topology revision reports.
//!
//! A charge is applied to a share, its device ledger, its account and its
//! store in one operation, each level checked against its bound; if any
//! refuses, nothing is charged. It is owned ([`MemoryCharge`]) and released
//! from every level when dropped. Each level has two views of one state: the
//! logical view charges allocations (tensors, ranges of blocks), the
//! physical view what the backend holds natively (blocks, granules, driver
//! objects). A native block owns the ranges suballocated from it, so a
//! level's claim is the larger of its two charges, never their sum. Each
//! level keeps its own totals: a charge updates them together, and debug
//! builds check after every charge that a device ledger's and an account's
//! views are the sums of their shares', and a store's claim the sum of its
//! accounts' claims. This is process accounting, not an OS-wide reservation.

use seismic_topology::{
    AccessPath, AccessPathId, Contract, DeviceTopology, Executor, ExecutorSelector, Placing, StoreId,
    StoreIdentity,
};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

/// A level's charges in its two views, whose larger is its claim.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct Views {
    charged: u64,
    held: u64,
}

impl Views {
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

    fn of(&self, physical: bool) -> u64 {
        if physical {
            self.held
        } else {
            self.charged
        }
    }
}

/// What one store's ledger has claimed, and the bound it is held to.
struct StoreState {
    claim: u64,
    bound: Option<u64>,
}

impl StoreState {
    fn room(&self, capacity: u64) -> u64 {
        capacity
            .min(self.bound.unwrap_or(u64::MAX))
            .saturating_sub(self.claim)
    }
}

/// Process-wide charges against one store: the sum of its accounts' claims.
pub(crate) struct StoreLedger {
    state: Mutex<StoreState>,
    /// Held across each admission against the store
    /// ([`Placement::decide`]): its admissions, then its decision, so no
    /// other admission decides between the two. Taken before any other lock
    /// of the hierarchy.
    deciding: Mutex<()>,
    /// What runs before each admission against the store, each once, all of
    /// them: every backend whose charges on it include memory it learns only
    /// by asking (Vulkan's driver memory) brings them up to date, keyed by
    /// the device.
    admissions: Mutex<Vec<(AdmissionKey, Admission)>>,
    /// The accounts that charge it, for the hierarchy's check.
    accounts: Mutex<Vec<Weak<Account>>>,
}

/// What runs before each admission against a store.
pub(crate) type Admission = Arc<dyn Fn() + Send + Sync>;

/// Which device an admission is: its UUID.
pub(crate) type AdmissionKey = [u8; 16];

/// One admission's decision ([`Placement::decide`]): what the placement has
/// available, and its charge, within the admission.
pub(crate) struct Decision<'a> {
    placement: &'a Arc<Placement>,
}

impl Decision<'_> {
    /// What the placement can still take.
    pub(crate) fn available(&self) -> u64 {
        self.placement.available()
    }

    /// Charge `bytes`, or refuse them whole.
    pub(crate) fn reserve(&self, bytes: u64) -> Result<MemoryReservation, MemoryCapacity> {
        self.placement.charge(bytes)
    }
}

impl StoreLedger {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(StoreState { claim: 0, bound: None }),
            deciding: Mutex::new(()),
            admissions: Mutex::new(Vec::new()),
            accounts: Mutex::new(Vec::new()),
        })
    }

    /// Run `admission` before each admission against the store, from now,
    /// in place of what `key`'s device registered before.
    pub(crate) fn add_admission(&self, key: AdmissionKey, admission: Admission) {
        let mut admissions = self.admissions.lock().expect("store admissions lock poisoned");
        admissions.retain(|(registered, _)| *registered != key);
        admissions.push((key, admission));
    }

    fn admissions(&self) -> Vec<Admission> {
        self.admissions
            .lock()
            .expect("store admissions lock poisoned")
            .iter()
            .map(|(_, admission)| admission.clone())
            .collect()
    }

    fn state(&self) -> MutexGuard<'_, StoreState> {
        self.state.lock().expect("store ledger lock poisoned")
    }

    /// What the store's accounts claim.
    pub(crate) fn claim(&self) -> u64 {
        self.state().claim
    }

    /// The store's bound, on top of its capacity.
    pub(crate) fn bound(&self) -> Option<u64> {
        self.state().bound
    }

    /// Hold the store's claim to `bound`, on top of its capacity. A lower
    /// bound refuses growth; it never revokes.
    pub(crate) fn set_bound(&self, bound: Option<u64>) {
        self.state().bound = bound;
    }

    /// The store's claim is the sum of its accounts' claims: checked in
    /// debug builds, taking the accounts' locks before the store's, as every
    /// charge does.
    fn check(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        let mut accounts = self
            .accounts
            .lock()
            .expect("store accounts lock poisoned")
            .iter()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        accounts.sort_by_key(|account| Arc::as_ptr(account));
        let held = accounts.iter().map(|account| account.state()).collect::<Vec<_>>();
        let sum = held.iter().map(|state| state.views.claim()).sum::<u64>();
        assert_eq!(self.state().claim, sum, "a store's claim is its accounts' claims");
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

/// An account's state: the sums of its shares' views, and its bound.
struct AccountState {
    views: Views,
    bound: Option<u64>,
}

/// What this process charges along one path: the sum of every open's share
/// of it, charging its store.
pub(crate) struct Account {
    store: Arc<StoreLedger>,
    state: Mutex<AccountState>,
    shares: Mutex<Vec<Weak<Share>>>,
}

impl Account {
    fn new(store: Arc<StoreLedger>) -> Arc<Self> {
        let account = Arc::new(Self {
            store: store.clone(),
            state: Mutex::new(AccountState { views: Views::default(), bound: None }),
            shares: Mutex::new(Vec::new()),
        });
        store
            .accounts
            .lock()
            .expect("store accounts lock poisoned")
            .push(Arc::downgrade(&account));
        account
    }

    fn state(&self) -> MutexGuard<'_, AccountState> {
        self.state.lock().expect("account lock poisoned")
    }

    /// The account's claim.
    pub(crate) fn claim(&self) -> u64 {
        self.state().views.claim()
    }

    /// The account's bound, on top of its grant.
    pub(crate) fn bound(&self) -> Option<u64> {
        self.state().bound
    }

    /// Hold the account's claim to `bound`, on top of its grant.
    pub(crate) fn set_bound(&self, bound: Option<u64>) {
        self.state().bound = bound;
    }

    /// The account's views are the sums of its shares': checked in debug
    /// builds, taking the shares' locks before the account's.
    fn check(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        let mut shares = self
            .shares
            .lock()
            .expect("account shares lock poisoned")
            .iter()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        shares.sort_by_key(|share| Arc::as_ptr(share));
        let held = shares.iter().map(|share| share.state()).collect::<Vec<_>>();
        let sum = held.iter().fold(Views::default(), |sum, views| Views {
            charged: sum.charged + views.charged,
            held: sum.held + views.held,
        });
        assert_eq!(self.state().views, sum, "an account's charges are its shares'");
    }
}

/// A device ledger's state: the sums of its shares' views, their claims
/// summed, and its own bound and marks.
struct DeviceState {
    views: Views,
    /// Its shares' claims together.
    claim: u64,
    bound: Option<u64>,
    /// The most `views.charged` has been since the device opened or its mark
    /// last restarted.
    high_water: u64,
    /// The most its charges would have been, over the same span, had it
    /// granted a reservation it refused.
    refused: Option<u64>,
}

/// One opened device's own charges on one store, under its own bound.
pub(crate) struct DeviceLedger {
    store: Arc<StoreLedger>,
    state: Mutex<DeviceState>,
    shares: Mutex<Vec<Weak<Share>>>,
}

impl DeviceLedger {
    pub(crate) fn new(store: Arc<StoreLedger>) -> Arc<Self> {
        Arc::new(Self {
            store,
            state: Mutex::new(DeviceState {
                views: Views::default(),
                claim: 0,
                bound: None,
                high_water: 0,
                refused: None,
            }),
            shares: Mutex::new(Vec::new()),
        })
    }

    fn state(&self) -> MutexGuard<'_, DeviceState> {
        self.state.lock().expect("device ledger lock poisoned")
    }

    /// The ledger of the store this one charges.
    pub(crate) fn store(&self) -> &Arc<StoreLedger> {
        &self.store
    }

    /// What the device's ledger reads now.
    pub(crate) fn view(&self) -> DeviceLedgerView {
        let state = self.state();
        DeviceLedgerView {
            charged: state.views.charged,
            claim: state.claim,
            bound: state.bound,
            high_water: state.high_water,
            refused: state.refused,
        }
    }

    /// Hold the device's claim on the store to `bound`.
    pub(crate) fn set_bound(&self, bound: Option<u64>) {
        self.state().bound = bound;
    }

    /// Start the high-water mark again from what is charged now, and forget
    /// the refusals before.
    pub(crate) fn restart_high_water(&self) {
        let mut state = self.state();
        state.high_water = state.views.charged;
        state.refused = None;
    }

    /// Raise the high-water mark to what is charged now: after a charge has
    /// been taken whole, so a refused one's passing charges never reach it.
    fn mark(&self) {
        let mut state = self.state();
        state.high_water = state.high_water.max(state.views.charged);
    }

    /// A refusal of `bytes` to the view (`physical`) of this device: what
    /// its charges would have been.
    fn refused(&self, bytes: u64, physical: bool) {
        let mut state = self.state();
        let demand = state.views.of(physical).saturating_add(bytes);
        state.refused = state.refused.max(Some(demand));
    }

    /// The ledger's views and claim are its shares': checked in debug
    /// builds, taking the device's lock before its shares', as every charge
    /// does.
    fn check(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        let state = self.state();
        let shares = self
            .shares
            .lock()
            .expect("device shares lock poisoned")
            .iter()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        let (mut views, mut claim) = (Views::default(), 0);
        for share in &shares {
            let share = share.state();
            views.charged += share.charged;
            views.held += share.held;
            claim += share.claim();
        }
        assert_eq!((state.views, state.claim), (views, claim), "a device's charges are its shares'");
    }
}

/// What a device ledger reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceLedgerView {
    /// What the device's allocations charge on the store.
    pub charged: u64,
    /// What the device claims there: per path, the larger of its
    /// allocations' charges and what its backend holds natively.
    pub claim: u64,
    /// Its bound.
    pub bound: Option<u64>,
    /// The most `charged` has been since the device opened or its mark was
    /// last restarted: what was charged and released again in between is
    /// seen here and nowhere else.
    pub high_water: u64,
    /// The most `charged` would have been, over the same span, had it
    /// granted a reservation it refused: none when nothing was refused.
    pub refused: Option<u64>,
}

/// One opened device's part of one path's account. An account outlives one
/// revision's identifiers, so the grant and the store's capacity that hold
/// a charge are those this open's revision reports.
pub(crate) struct Share {
    account: Arc<Account>,
    device: Arc<DeviceLedger>,
    /// The store's capacity.
    capacity: u64,
    grant: Option<u64>,
    state: Mutex<Views>,
}

impl Share {
    fn new(account: Arc<Account>, device: Arc<DeviceLedger>, capacity: u64, grant: Option<u64>) -> Arc<Self> {
        let share = Arc::new(Self {
            account: account.clone(),
            device: device.clone(),
            capacity,
            grant,
            state: Mutex::new(Views::default()),
        });
        account
            .shares
            .lock()
            .expect("account shares lock poisoned")
            .push(Arc::downgrade(&share));
        device
            .shares
            .lock()
            .expect("device shares lock poisoned")
            .push(Arc::downgrade(&share));
        share
    }

    fn state(&self) -> MutexGuard<'_, Views> {
        self.state.lock().expect("share lock poisoned")
    }

    /// The account of the path this share is of.
    pub(crate) fn account(&self) -> &Arc<Account> {
        &self.account
    }

    /// What the view (`physical`) of this share can still grow by, given
    /// `store_room` left in its store and `device_room` left to its device
    /// on that store: within its device's bound, its grant, its account's
    /// bound, and the store. A view's growth within its share's or its
    /// account's larger claim takes none of the room above.
    fn growth(
        device_room: u64,
        share: &Views,
        account: &AccountState,
        grant: Option<u64>,
        store_room: u64,
        physical: bool,
    ) -> u64 {
        let by_device = (share.claim() - share.of(physical)).saturating_add(device_room);
        let account_room = grant
            .unwrap_or(u64::MAX)
            .min(account.bound.unwrap_or(u64::MAX))
            .saturating_sub(account.views.claim())
            .min(store_room);
        let by_account = (account.views.claim() - account.views.of(physical)).saturating_add(account_room);
        by_device.min(by_account)
    }

    /// Apply `bytes` to the view at every level (`add`, or release), with
    /// every level's lock held in the hierarchy's order.
    fn apply(
        device: &mut DeviceState,
        share: &mut Views,
        account: &mut AccountState,
        store: &mut StoreState,
        bytes: u64,
        physical: bool,
        add: bool,
    ) {
        let (share_prior, account_prior) = (share.claim(), account.views.claim());
        let change = |value: &mut u64| {
            *value = if add {
                value.checked_add(bytes).expect("a charge overflows its level")
            } else {
                value.checked_sub(bytes).expect("released more than a level holds")
            }
        };
        change(share.own(physical));
        change(device.views.own(physical));
        change(account.views.own(physical));
        device.claim = device.claim - share_prior + share.claim();
        store.claim = store.claim - account_prior + account.views.claim();
    }

    /// Charge up to `requested` bytes to the view (`physical`) at every
    /// level; what was charged.
    fn reserve(&self, requested: u64, physical: bool) -> u64 {
        let charged = {
            let mut device = self.device.state();
            let mut share = self.state();
            let mut account = self.account.state();
            let mut store = self.account.store.state();
            let store_room = store.room(self.capacity);
            let device_room = device.bound.unwrap_or(u64::MAX).saturating_sub(device.claim);
            let growth = Self::growth(device_room, &share, &account, self.grant, store_room, physical);
            let bytes = requested.min(growth);
            Self::apply(&mut device, &mut share, &mut account, &mut store, bytes, physical, true);
            bytes
        };
        self.check();
        charged
    }

    /// Release `bytes` the view charged, at every level.
    fn release(&self, bytes: u64, physical: bool) {
        {
            let mut device = self.device.state();
            let mut share = self.state();
            let mut account = self.account.state();
            let mut store = self.account.store.state();
            Self::apply(&mut device, &mut share, &mut account, &mut store, bytes, physical, false);
        }
        self.check();
    }

    /// Charge `bytes` of memory a driver already took, never refused.
    fn record(&self, bytes: u64, physical: bool) {
        {
            let mut device = self.device.state();
            let mut share = self.state();
            let mut account = self.account.state();
            let mut store = self.account.store.state();
            Self::apply(&mut device, &mut share, &mut account, &mut store, bytes, physical, true);
        }
        self.check();
    }

    /// The hierarchy's sums above this share, checked in debug builds.
    fn check(&self) {
        self.device.check();
        self.account.check();
        self.account.store.check();
    }
}

/// What an account is for: an access path, by its position among the
/// executor's paths to its store, or the backend's staging.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum AccountOf {
    Access(usize),
    Staging,
}

/// An account's key. An executor's paths to one store are built in the
/// same order from the same reports, so a store and a position among them
/// name a path across revisions; its account outlives one revision's
/// identifiers and is shared by every open of the executor.
type AccountKey = (ExecutorSelector, StoreIdentity, AccountOf);

/// The accounts opened executors hold, by key.
fn accounts() -> MutexGuard<'static, HashMap<AccountKey, Weak<Account>>> {
    static ACCOUNTS: OnceLock<Mutex<HashMap<AccountKey, Weak<Account>>>> = OnceLock::new();
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

/// What an account of `path`, one of `topology`'s, is for: its position
/// among its executor's paths to its store.
fn access_of(topology: &DeviceTopology, path: &AccessPath) -> AccountOf {
    AccountOf::Access(
        topology
            .access_paths()
            .iter()
            .filter(|other| other.executor() == path.executor() && other.store() == path.store())
            .position(|other| other.id() == path.id())
            .expect("a path is among its executor's paths"),
    )
}

/// What `executor`'s account along its access path `path` claims now: none
/// while no opened executor holds that account.
pub(crate) fn path_claim(topology: &DeviceTopology, executor: &Executor, path: &AccessPath) -> u64 {
    let key = account_key(topology, executor, path.store(), access_of(topology, path));
    accounts()
        .get(&key)
        .and_then(Weak::upgrade)
        .map_or(0, |account| account.claim())
}

/// One of the topology's access paths of an opened device, with its share
/// of the path's account.
pub(crate) struct AccessShare {
    pub(crate) id: AccessPathId,
    pub(crate) share: Arc<Share>,
}

/// An opened device's ledgers: its own ledger on each store it reaches, its
/// share of each of its paths' accounts, each storage kind's placement over
/// them, and the placement its backend's transfer staging charges in host
/// RAM: native buffers no allocation lies in, so its physical view.
pub(crate) struct DeviceLedgers {
    /// Each placing's placement, by [`Placing::index`], and its kinds as the
    /// backend names them.
    placed: [(Arc<Placement>, Vec<u32>); Placing::COUNT],
    /// Each kind's own placement, by the backend's name of it: what its
    /// native blocks are charged along.
    kinds: Vec<(u32, Arc<Placement>)>,
    pub(crate) staging: Arc<Placement>,
    pub(crate) paths: Vec<AccessShare>,
    /// The device's own ledger on each store it reaches.
    pub(crate) stores: Vec<(StoreId, Arc<DeviceLedger>)>,
}

impl DeviceLedgers {
    pub(crate) fn new(topology: &DeviceTopology, executor: &Executor) -> Arc<Self> {
        let mut registry = accounts();
        let mut stores: Vec<(StoreId, Arc<DeviceLedger>)> = Vec::new();
        let mut share = |store: StoreId, grant, of| {
            let key = account_key(topology, executor, store, of);
            let ledger = store_ledger(topology, store);
            let account = registry.get(&key).and_then(Weak::upgrade).unwrap_or_else(|| {
                let account = Account::new(ledger.clone());
                registry.insert(key, Arc::downgrade(&account));
                account
            });
            let device = match stores.iter().find(|(id, _)| *id == store) {
                Some((_, device)) => device.clone(),
                None => {
                    let device = DeviceLedger::new(ledger);
                    stores.push((store, device.clone()));
                    device
                }
            };
            let capacity = topology
                .store(store)
                .expect("path store belongs to topology")
                .capacity()
                .value;
            Share::new(account, device, capacity, grant)
        };
        let paths = topology
            .access_paths()
            .iter()
            .filter(|path| path.executor() == executor.id())
            .map(|path| AccessShare {
                id: path.id(),
                share: share(path.store(), path.grant().map(|grant| grant.value), access_of(topology, path)),
            })
            .collect::<Vec<_>>();
        let staging = Placement::new(vec![share(topology.host_ram().id(), None, AccountOf::Staging)]).physical();
        let along = |ids: &[AccessPathId]| {
            Placement::new(
                ids.iter()
                    .map(|id| {
                        paths
                            .iter()
                            .find(|path| path.id == *id)
                            .expect("placement path belongs to executor")
                            .share
                            .clone()
                    })
                    .collect(),
            )
        };
        let native = |id| topology.kind(id).expect("a placement names kinds of its revision").native().id;
        let placed = Placing::ALL.map(|placing| {
            let kinds = executor.placement(placing).iter().map(|id| native(*id)).collect();
            (along(executor.paths(placing)), kinds)
        });
        let kinds = topology
            .kinds()
            .iter()
            .filter(|kind| kind.executor() == executor.id())
            .map(|kind| (kind.native().id, along(kind.paths())))
            .collect();
        Arc::new(Self { placed, kinds, staging, paths, stores })
    }

    /// Ledgers of no topology: `device` under `DeviceOnly`, `upload` under
    /// `Upload`, `host` under `Shared` and `Words`, fixed or reserved.
    #[cfg(test)]
    pub(crate) fn split(device: Arc<Placement>, upload: Arc<Placement>, host: Arc<Placement>, staging: Arc<Placement>) -> Self {
        let placed = Placing::ALL.map(|placing| {
            let placement = match placing.contract() {
                Contract::DeviceOnly => &device,
                Contract::Upload => &upload,
                Contract::Shared | Contract::Words => &host,
            };
            (placement.clone(), Vec::new())
        });
        Self { placed, kinds: Vec::new(), staging: staging.physical(), paths: Vec::new(), stores: Vec::new() }
    }

    /// The placement of a buffer placed as `placing`.
    pub(crate) fn placement(&self, placing: Placing) -> &Arc<Placement> {
        &self.placed[placing.index()].0
    }

    /// The kinds a buffer placed as `placing` is allocated from, as the
    /// backend names them.
    pub(crate) fn kinds(&self, placing: Placing) -> &[u32] {
        &self.placed[placing.index()].1
    }

    /// The executor's kinds, each with its own placement.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    pub(crate) fn kind_placements(&self) -> &[(u32, Arc<Placement>)] {
        &self.kinds
    }

    /// The placement of fixed `DeviceOnly` buffers.
    pub(crate) fn device(&self) -> &Arc<Placement> {
        self.placement(Placing::Fixed(Contract::DeviceOnly))
    }

    /// The placement of `Upload` buffers.
    pub(crate) fn upload(&self) -> &Arc<Placement> {
        self.placement(Placing::Fixed(Contract::Upload))
    }

    /// The device's own ledger on the store fixed `DeviceOnly` buffers are
    /// placed on first.
    pub(crate) fn device_ledger(&self) -> &Arc<DeviceLedger> {
        &self.device().shares.first().expect("an opened executor places device storage").device
    }

    /// This device's share of its access path `id`.
    pub(crate) fn path(&self, id: AccessPathId) -> Option<&AccessShare> {
        self.paths.iter().find(|path| path.id == id)
    }

    /// The device's own ledger on `store`, where it reaches it.
    pub(crate) fn store(&self, store: StoreId) -> Option<&Arc<DeviceLedger>> {
        self.stores.iter().find(|(id, _)| *id == store).map(|(_, ledger)| ledger)
    }

    /// What `charge`, an allocation of this device, holds on each of its
    /// access paths.
    pub(crate) fn path_charges(&self, charge: &MemoryCharge) -> Vec<(AccessPathId, u64)> {
        charge
            .portions
            .iter()
            .map(|(share, bytes)| {
                let path = self
                    .paths
                    .iter()
                    .find(|path| Arc::ptr_eq(&path.share, share))
                    .expect("a device's allocation is charged along its own paths");
                (path.id, *bytes)
            })
            .collect()
    }
}

/// The bytes `charges` along `topology`'s access paths hold on each store,
/// each store once, in store order so that equal charges compare equal.
pub(crate) fn store_charges(
    topology: &DeviceTopology,
    charges: impl IntoIterator<Item = (AccessPathId, u64)>,
) -> BTreeMap<StoreId, u64> {
    let mut stores = BTreeMap::new();
    for (path, bytes) in charges {
        let store = topology
            .access_path(path)
            .expect("charged path belongs to topology")
            .store();
        *stores.entry(store).or_default() += bytes;
    }
    stores
}

/// A placement's view as the backend's account for its native memory.
pub(crate) struct MemoryAccount(pub(crate) Arc<Placement>);

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
        let shares = &self.0.shares;
        for (index, share) in shares.iter().enumerate() {
            let amount = if index + 1 == shares.len() {
                left
            } else {
                left.min(
                    share
                        .grant
                        .unwrap_or(u64::MAX)
                        .saturating_sub(share.account.claim()),
                )
            };
            if amount != 0 {
                portions.push((share.clone(), amount));
            }
            left -= amount;
        }
        assert_eq!(left, 0, "physical allocation has a placement");
        self.0.record(&portions);
        Box::new(MemoryReservation {
            placement: self.0.clone(),
            portions,
        })
    }

    fn reserve_decided(
        &self,
        decide: &mut dyn FnMut(u64) -> Result<u64, seismic_compiler::errors::ExecutionError>,
    ) -> Result<(Box<dyn seismic_compiler::executable::MemoryHold>, u64), seismic_compiler::errors::ExecutionError>
    {
        self.0.decide(|decision| {
            let bytes = decide(decision.available())?;
            let hold = decision.reserve(bytes).map_err(|capacity| {
                seismic_compiler::errors::ExecutionError::AllocationCapacity {
                    required: capacity.required.into(),
                    available: capacity.available,
                }
            })?;
            Ok((Box::new(hold) as Box<dyn seismic_compiler::executable::MemoryHold>, bytes))
        })
    }

    fn reserve(
        &self,
        bytes: u64,
    ) -> Result<
        Box<dyn seismic_compiler::executable::MemoryHold>,
        seismic_compiler::errors::ExecutionError,
    > {
        Placement::reserve(&self.0, bytes)
            .map(|hold| Box::new(hold) as Box<dyn seismic_compiler::executable::MemoryHold>)
            .map_err(
                |capacity| seismic_compiler::errors::ExecutionError::AllocationCapacity {
                    required: capacity.required.into(),
                    available: capacity.available,
                },
            )
    }
}

/// Where a storage kind's charges go: one view (logical or physical) of a
/// device's shares of its placement's paths, spent in order. It keeps no
/// state of its own; every charge is the levels' ([`Share`]). Lock order is
/// always the decision locks of the stores its paths charge (in address
/// order, held across an admission, [`Self::decide`]), then the levels of
/// one share at a time. An admission's registered admissions run under the
/// decision locks alone, so they may charge (a driver's memory recorded)
/// without deadlock.
pub(crate) struct Placement {
    shares: Vec<Arc<Share>>,
    physical: bool,
}

type Portions = Vec<(Arc<Share>, u64)>;

impl Placement {
    /// The logical view of a placement over `shares`, spent in order.
    pub(crate) fn new(shares: Vec<Arc<Share>>) -> Arc<Self> {
        Arc::new(Self { shares, physical: false })
    }

    /// A placement of one unlimited path on the store `ledger` keeps, for a
    /// device of its own.
    #[cfg(test)]
    pub(crate) fn on(ledger: Arc<StoreLedger>) -> Arc<Self> {
        Self::new(vec![Share::new(Account::new(ledger.clone()), DeviceLedger::new(ledger), u64::MAX, None)])
    }

    /// The device ledger of the placement's first path.
    #[cfg(test)]
    pub(crate) fn ledger(&self) -> &Arc<DeviceLedger> {
        &self.shares[0].device
    }

    /// Run `admission`, `key`'s device's, before each admission against the
    /// stores of the placement's paths ([`StoreLedger::add_admission`]).
    pub(crate) fn add_store_admission(&self, key: AdmissionKey, admission: Admission) {
        for share in &self.shares {
            share.account.store.add_admission(key, admission.clone());
        }
    }

    /// The physical view of this placement: what the backend holds natively.
    pub(crate) fn physical(&self) -> Arc<Self> {
        Arc::new(Self { shares: self.shares.clone(), physical: true })
    }

    /// The device ledgers the placement charges, each once.
    fn devices(&self) -> Vec<&Arc<DeviceLedger>> {
        let mut devices: Vec<&Arc<DeviceLedger>> = Vec::new();
        for share in &self.shares {
            if !devices.iter().any(|device| Arc::ptr_eq(device, &share.device)) {
                devices.push(&share.device);
            }
        }
        devices
    }

    /// What this view can still charge: each path's growth within every
    /// level above it, a store or a device ledger reached by several paths
    /// counted once.
    pub(crate) fn available(&self) -> u64 {
        let mut stores = HashMap::new();
        let mut devices = HashMap::new();
        let mut available = 0u64;
        for share in &self.shares {
            let device = share.device.state();
            let views = share.state();
            let account = share.account.state();
            let store_key = Arc::as_ptr(&share.account.store);
            let store_room = *stores
                .entry(store_key)
                .or_insert_with(|| share.account.store.state().room(share.capacity));
            let device_key = Arc::as_ptr(&share.device);
            let device_room = *devices
                .entry(device_key)
                .or_insert_with(|| device.bound.unwrap_or(u64::MAX).saturating_sub(device.claim));
            let growth = Share::growth(device_room, &views, &account, share.grant, store_room, self.physical);
            // What the growth takes of each level's room: none of it within
            // that level's larger claim.
            let beyond = |claim: u64, view: u64| growth.saturating_sub(claim - view);
            *devices.get_mut(&device_key).expect("entered above") -= beyond(views.claim(), views.of(self.physical));
            *stores.get_mut(&store_key).expect("entered above") -=
                beyond(account.views.claim(), account.views.of(self.physical));
            available = available.saturating_add(growth);
        }
        available
    }

    /// Charge `bytes` along the placement, or refuse them whole, as one
    /// admission ([`Self::decide`]).
    pub(crate) fn reserve(self: &Arc<Self>, bytes: u64) -> Result<MemoryReservation, MemoryCapacity> {
        self.decide(|decision| decision.reserve(bytes))
    }

    /// Run `decide` as one admission against the stores the placement's
    /// paths charge: each store's decision lock taken (in address order, so
    /// admissions spanning stores never deadlock), every admission each
    /// store registered run, then `decide`, which charges through its
    /// [`Decision`]. No other admission against those stores runs between
    /// their admissions and the decision.
    pub(crate) fn decide<T>(self: &Arc<Self>, decide: impl FnOnce(&Decision<'_>) -> T) -> T {
        let mut stores = self.shares.iter().map(|share| &share.account.store).collect::<Vec<_>>();
        stores.sort_by_key(|store| Arc::as_ptr(store));
        stores.dedup_by(|a, b| Arc::ptr_eq(a, b));
        let _deciding = stores
            .iter()
            .map(|store| store.deciding.lock().expect("store decision lock poisoned"))
            .collect::<Vec<_>>();
        for store in &stores {
            for admission in store.admissions() {
                admission();
            }
        }
        decide(&Decision { placement: self })
    }

    /// Charge `bytes` along the placement, each path in order up to what
    /// every level above it can take, or refuse them whole: then nothing is
    /// charged, and each device ledger the placement reaches reads the
    /// demand it refused. Within an admission ([`Decision::reserve`]).
    fn charge(self: &Arc<Self>, bytes: u64) -> Result<MemoryReservation, MemoryCapacity> {
        let mut remaining = bytes;
        let mut portions = Vec::new();
        for share in &self.shares {
            let amount = share.reserve(remaining, self.physical);
            if amount != 0 {
                portions.push((share.clone(), amount));
                remaining -= amount;
            }
            if remaining == 0 {
                break;
            }
        }
        if remaining != 0 {
            for (share, amount) in portions {
                share.release(amount, self.physical);
            }
            for device in self.devices() {
                device.refused(bytes, self.physical);
            }
            return Err(MemoryCapacity {
                required: bytes,
                available: bytes - remaining,
            });
        }
        if !self.physical {
            for device in self.devices() {
                device.mark();
            }
        }
        Ok(MemoryReservation {
            placement: self.clone(),
            portions,
        })
    }

    fn release(&self, portions: &Portions) {
        for (share, bytes) in portions {
            share.release(*bytes, self.physical);
        }
    }

    /// Charge memory a driver already took. It is no reservation, so it is
    /// never refused: like a falling bound, it can leave a level above its
    /// bound.
    fn record(&self, portions: &Portions) {
        for (share, bytes) in portions {
            share.record(*bytes, self.physical);
        }
        if !self.physical {
            for device in self.devices() {
                device.mark();
            }
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

/// The bytes `portions` charge.
fn bytes_of(portions: &Portions) -> u64 {
    portions.iter().map(|(_, bytes)| bytes).sum()
}

/// Remove a prefix of `bytes` from `portions`, preserving the exact paths
/// charged at admission.
fn take(portions: &mut Portions, mut bytes: u64) -> Portions {
    let mut taken = Vec::new();
    while bytes != 0 {
        let (share, available) = portions
            .first_mut()
            .expect("charge partition covers its byte count");
        let amount = bytes.min(*available);
        taken.push((share.clone(), amount));
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
    for (share, bytes) in extra {
        if let Some((last, held)) = portions.last_mut() {
            if Arc::ptr_eq(last, &share) {
                *held += bytes;
                continue;
            }
        }
        portions.push((share, bytes));
    }
}

/// One atomic reservation for a complete invocation. It is split into the
/// allocation-owned charges as physical allocations are created; any unused
/// tail is released on drop. This prevents concurrent allocations from
/// invalidating a successful invocation-capacity check.
pub(crate) struct MemoryReservation {
    placement: Arc<Placement>,
    portions: Portions,
}

impl MemoryReservation {
    /// Reserved bytes not yet taken by an allocation.
    pub(crate) fn remaining(&self) -> u64 {
        bytes_of(&self.portions)
    }

    /// The charge of one allocation of `bytes`, taken from the reservation.
    pub(crate) fn take(&mut self, bytes: u64) -> MemoryCharge {
        MemoryCharge {
            placement: self.placement.clone(),
            portions: take(&mut self.portions, bytes),
        }
    }

    /// Transfer an already reserved increase into the charge for one
    /// physical allocation whose backing is being resized in place.
    pub(crate) fn extend(&mut self, charge: &mut MemoryCharge, bytes: u64) {
        assert!(Arc::ptr_eq(&self.placement, &charge.placement));
        append(&mut charge.portions, take(&mut self.portions, bytes));
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.placement.release(&self.portions);
    }
}

/// One allocation's charge, released from every level when the allocation
/// drops.
pub(crate) struct MemoryCharge {
    placement: Arc<Placement>,
    portions: Portions,
}

impl MemoryCharge {
    pub(crate) fn bytes(&self) -> u64 {
        bytes_of(&self.portions)
    }

    /// Keep only the first `bytes` of the charge, releasing the rest.
    pub(crate) fn shrink(&mut self, bytes: u64) {
        let retained = take(&mut self.portions, bytes);
        self.placement.release(&std::mem::replace(&mut self.portions, retained));
    }
}

impl Drop for MemoryCharge {
    fn drop(&mut self) {
        self.placement.release(&self.portions);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A share of a path of `grant` bytes on `store` (of `capacity`), for
    /// the device whose ledger on that store is `device`.
    fn share(store: &Arc<StoreLedger>, capacity: u64, grant: u64, device: &Arc<DeviceLedger>) -> Arc<Share> {
        Share::new(Account::new(store.clone()), device.clone(), capacity, Some(grant))
    }

    fn placement() -> Arc<Placement> {
        Placement::on(StoreLedger::new())
    }

    fn device_of(placement: &Placement) -> &Arc<DeviceLedger> {
        &placement.shares[0].device
    }

    /// A device on host RAM and its own memory reports its stores in one
    /// order however its charges arrive, so equal censuses compare equal.
    #[test]
    fn store_charges_come_in_store_order() {
        let topology = seismic_topology::fixtures::recorded::rtx3090();
        let charges: Vec<_> = topology
            .access_paths()
            .iter()
            .rev()
            .map(|path| (path.id(), 1))
            .collect();
        let mut stores: Vec<_> = topology.stores().iter().map(|store| store.id()).collect();
        stores.sort();
        assert!(stores.len() > 1);
        for _ in 0..16 {
            let reported: Vec<_> = store_charges(&topology, charges.clone())
                .into_keys()
                .collect();
            assert_eq!(reported, stores);
        }
    }

    /// A charge applies to every level at once and its drop releases every
    /// level: the share, its device ledger, its account and its store read
    /// the same bytes while it lives, and nothing after.
    #[test]
    fn a_charge_is_one_object_on_every_level() {
        let store = StoreLedger::new();
        let device = DeviceLedger::new(store.clone());
        let path = share(&store, 100, 100, &device);
        let placement = Placement::new(vec![path.clone()]);
        let held = placement.reserve(7).unwrap();
        assert_eq!((path.state().claim(), device.view().claim, path.account().claim(), store.claim()), (7, 7, 7, 7));
        drop(held);
        assert_eq!((path.state().claim(), device.view().claim, path.account().claim(), store.claim()), (0, 0, 0, 0));
    }

    /// Any level's refusal refuses the charge whole: nothing is charged at
    /// any level.
    #[test]
    fn a_refusal_at_any_level_charges_nothing() {
        for refusing in 0..3 {
            let store = StoreLedger::new();
            let device = DeviceLedger::new(store.clone());
            let path = share(&store, 100, 100, &device);
            match refusing {
                0 => device.set_bound(Some(5)),
                1 => path.account().set_bound(Some(5)),
                _ => store.set_bound(Some(5)),
            }
            let placement = Placement::new(vec![path.clone()]);
            assert!(placement.reserve(6).is_err());
            assert_eq!((path.state().claim(), device.view().claim, path.account().claim(), store.claim()), (0, 0, 0, 0));
        }
    }

    #[test]
    fn native_blocks_and_logical_ranges_share_capacity_without_double_charge() {
        let store = StoreLedger::new();
        store.set_bound(Some(16));
        let device = DeviceLedger::new(store.clone());
        let logical = Placement::new(vec![share(&store, 16, 16, &device)]);
        let physical = logical.physical();
        let range = logical.reserve(4).unwrap();
        let block = physical.reserve(12).unwrap();
        assert_eq!(store.claim(), 12);
        let second_range = logical.reserve(8).unwrap();
        assert_eq!(store.claim(), 12);
        let other = Placement::on(store.clone()).reserve(4).unwrap();
        assert!(physical.reserve(1).is_err());
        drop(range);
        drop(second_range);
        assert_eq!(store.claim(), 16);
        drop(block);
        assert_eq!(store.claim(), 4);
        drop(other);
        assert_eq!(store.claim(), 0);
    }

    #[test]
    fn measured_native_overhead_consumes_shared_admission_until_released() {
        use seismic_compiler::executable::MemoryAccount as _;
        let store = StoreLedger::new();
        store.set_bound(Some(10));
        let device = DeviceLedger::new(store.clone());
        let logical = Placement::new(vec![share(&store, 10, 10, &device)]);
        let physical = logical.physical();
        let block = physical.reserve(8).unwrap();
        let overhead = MemoryAccount(physical).record(4);
        assert_eq!(store.claim(), 12);
        assert!(Placement::on(store.clone()).reserve(1).is_err());
        drop(overhead);
        assert!(Placement::on(store.clone()).reserve(2).is_ok());
        drop(block);
        assert_eq!(store.claim(), 0);
    }

    /// The physical view charges the device's ledger as the logical one
    /// does: its refusals are in the device's reading, and the device's
    /// bound bounds what the backend holds.
    #[test]
    fn the_physical_view_shares_the_devices_bound_and_refusals() {
        use seismic_compiler::executable::MemoryAccount as _;
        let logical = placement();
        let device = device_of(&logical).clone();
        device.set_bound(Some(10));
        let physical = logical.physical();
        let block = physical.reserve(8).unwrap();
        assert_eq!(MemoryAccount(physical.clone()).available(), 2);
        assert!(physical.reserve(4).is_err());
        assert_eq!(device.view().refused, Some(12));
        assert_eq!(device.view().charged, 0);
        drop(block);
        assert_eq!(MemoryAccount(physical).available(), 10);
    }

    #[test]
    fn placements_share_grants_and_stores_and_release_the_paths_they_reserved() {
        let first_store = StoreLedger::new();
        let second_store = StoreLedger::new();
        let (first_device, second_device) = (DeviceLedger::new(first_store.clone()), DeviceLedger::new(second_store.clone()));
        let first = share(&first_store, 20, 6, &first_device);
        let second = share(&second_store, 20, 8, &second_device);
        let device = Placement::new(vec![first.clone(), second.clone()]);
        let upload = Placement::new(vec![first.clone(), second.clone()]);
        let mut reservation = device.reserve(10).unwrap();
        assert_eq!((first.state().claim(), second.state().claim()), (6, 4));
        let mut charge = reservation.take(8);
        drop(reservation);
        assert_eq!((first_store.claim(), second_store.claim()), (6, 2));
        let shared = upload.reserve(6).unwrap();
        assert!(upload.reserve(1).is_err());
        charge.shrink(4);
        assert_eq!((first_store.claim(), second_store.claim()), (4, 6));
        drop(shared);
        drop(charge);
        assert_eq!((first_store.claim(), second_store.claim()), (0, 0));
    }

    #[test]
    fn different_paths_cannot_spend_one_stores_capacity_twice() {
        let store = StoreLedger::new();
        let device = DeviceLedger::new(store.clone());
        let placement = Placement::new(vec![share(&store, 10, 8, &device), share(&store, 10, 8, &device)]);
        assert!(placement.reserve(11).is_err());
        assert_eq!(store.claim(), 0);
        let held = placement.reserve(10).unwrap();
        assert_eq!(device.view().claim, 10);
        drop(held);
        assert_eq!(store.claim(), 0);
    }

    #[test]
    fn live_bounds_are_shared_across_placements_and_executor_paths() {
        let store = StoreLedger::new();
        let device = DeviceLedger::new(store.clone());
        let a = share(&store, 100, 80, &device);
        let b = share(&store, 100, 80, &device);
        let first = Placement::new(vec![a.clone()]);
        let second = Placement::new(vec![b]);
        store.set_bound(Some(10));
        a.account().set_bound(Some(7));
        let held = first.reserve(6).unwrap();
        assert!(first.reserve(2).is_err());
        let other = second.reserve(4).unwrap();
        assert!(first.reserve(1).is_err());
        assert!(second.reserve(1).is_err());
        store.set_bound(Some(5));
        drop(other);
        assert!(second.reserve(1).is_err());
        drop(held);
        assert!(second.reserve(5).is_ok());
    }

    /// Two opens of one executor share its paths' accounts, each with its
    /// own device ledger: each reads its own charges, under its own bound,
    /// while the grant bounds both together.
    #[test]
    fn opens_share_an_account_under_their_own_device_bounds() {
        let store = StoreLedger::new();
        let account = Account::new(store.clone());
        let (first, second) = (DeviceLedger::new(store.clone()), DeviceLedger::new(store.clone()));
        let one = Placement::new(vec![Share::new(account.clone(), first.clone(), 100, Some(10))]);
        let two = Placement::new(vec![Share::new(account.clone(), second.clone(), 100, Some(10))]);
        first.set_bound(Some(4));
        let held = one.reserve(4).unwrap();
        assert!(one.reserve(1).is_err());
        let other = two.reserve(6).unwrap();
        assert!(two.reserve(1).is_err(), "the grant bounds both opens");
        assert_eq!((first.view().charged, second.view().charged, account.claim(), store.claim()), (4, 6, 10, 10));
        drop((held, other));
        assert_eq!(store.claim(), 0);
    }

    /// An account outlives a topology revision: an open is held to the grant
    /// and capacity its own revision reports, not the first open's.
    #[test]
    fn an_open_is_held_to_its_own_revisions_grant_and_capacity() {
        let store = StoreLedger::new();
        let account = Account::new(store.clone());
        let device = || DeviceLedger::new(store.clone());
        let earlier = Placement::new(vec![Share::new(account.clone(), device(), 100, Some(10))]);
        let later = Placement::new(vec![Share::new(account.clone(), device(), 100, Some(20))]);
        let smaller = Placement::new(vec![Share::new(account.clone(), device(), 15, None)]);
        let held = earlier.reserve(10).unwrap();
        assert!(earlier.reserve(1).is_err());
        assert_eq!(smaller.available(), 5);
        let more = later.reserve(10).unwrap();
        assert!(later.reserve(1).is_err());
        drop((held, more));
    }

    #[test]
    fn backend_staging_holds_share_the_host_ledger_until_native_release() {
        let store = StoreLedger::new();
        store.set_bound(Some(10));
        let tensors = Placement::on(store.clone());
        // Staging is built as the ledgers build it: what the backend holds
        // natively, never an allocation's charge.
        let staging = Placement::on(store.clone()).physical();
        let tensor = tensors.reserve(6).unwrap();
        let held = seismic_compiler::executable::MemoryAccount::reserve(&MemoryAccount(staging.clone()), 4)
            .unwrap();
        let view = staging.ledger().view();
        assert_eq!((view.charged, view.claim, view.high_water), (0, 4, 0), "staging is claimed, not charged");
        assert_eq!(store.claim(), 10);
        assert!(tensors.reserve(1).is_err());
        drop(held);
        assert_eq!(store.claim(), 6);
        assert!(tensors.reserve(4).is_ok());
        drop(tensor);
        assert_eq!(store.claim(), 0);
    }

    /// A placement over `ledgers`, one unlimited path on each, in order, of
    /// one device.
    fn spanning(ledgers: &[&Arc<StoreLedger>]) -> Arc<Placement> {
        Placement::new(
            ledgers
                .iter()
                .map(|ledger| Share::new(Account::new((*ledger).clone()), DeviceLedger::new((*ledger).clone()), u64::MAX, None))
                .collect(),
        )
    }

    /// Admissions on two threads against one store, then against two
    /// stores each spanning both in opposite orders, each decide on their
    /// own admission's reading: every store's admission (a driver's growth
    /// recorded, as Vulkan's reconciliation does) finds the store charged
    /// with exactly every earlier admission's growth and decision, never one
    /// between another's admission and its decision, and the address-ordered
    /// decision locks never deadlock.
    #[test]
    fn concurrent_admissions_decide_on_their_own_reading() {
        use std::sync::atomic::{AtomicU64, Ordering};
        const ADMISSIONS: u64 = 300;
        for stores in [1usize, 2] {
            let ledgers = (0..stores).map(|_| StoreLedger::new()).collect::<Vec<_>>();
            let grown = (0..stores).map(|_| Arc::new(AtomicU64::new(0))).collect::<Vec<_>>();
            let decided = (0..stores).map(|_| Arc::new(AtomicU64::new(0))).collect::<Vec<_>>();
            let holds = Arc::new(Mutex::new(Vec::new()));
            for (index, ledger) in ledgers.iter().enumerate() {
                let driver = MemoryAccount(spanning(&[ledger]).physical());
                let (ledger_seen, grown, decided, holds) =
                    (ledger.clone(), grown[index].clone(), decided[index].clone(), holds.clone());
                ledger.add_admission([index as u8; 16], Arc::new(move || {
                    let expected = grown.load(Ordering::SeqCst) + decided.load(Ordering::SeqCst);
                    assert_eq!(ledger_seen.claim(), expected, "an admission saw a charge another admission had not settled");
                    use seismic_compiler::executable::MemoryAccount as _;
                    holds.lock().unwrap().push(driver.record(1));
                    grown.fetch_add(1, Ordering::SeqCst);
                }));
            }
            let orders: Vec<Vec<usize>> = if stores == 1 { vec![vec![0], vec![0]] } else { vec![vec![0, 1], vec![1, 0]] };
            let threads = orders
                .into_iter()
                .map(|order| {
                    let placement = spanning(&order.iter().map(|&index| &ledgers[index]).collect::<Vec<_>>());
                    let decided = decided[order[0]].clone();
                    std::thread::spawn(move || {
                        let mut reservations = Vec::new();
                        for _ in 0..ADMISSIONS {
                            reservations.push(placement.decide(|decision| {
                                let reservation = decision.reserve(1).unwrap();
                                decided.fetch_add(1, Ordering::SeqCst);
                                reservation
                            }));
                        }
                        reservations
                    })
                })
                .collect::<Vec<_>>();
            let kept = threads.into_iter().map(|thread| thread.join().unwrap()).collect::<Vec<_>>();
            let total = ledgers.iter().map(|ledger| ledger.claim()).sum::<u64>();
            assert_eq!(total, 2 * ADMISSIONS * stores as u64 + 2 * ADMISSIONS);
            drop(kept);
        }
    }

    /// A store's admission runs before each charge against it, from any
    /// view of any placement on the store, before the charge is decided.
    #[test]
    fn a_stores_admission_runs_before_each_charge() {
        let placement = placement();
        let runs = Arc::new(std::sync::atomic::AtomicU64::new(0));
        placement.add_store_admission([0; 16], {
            let runs = runs.clone();
            Arc::new(move || {
                runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            })
        });
        let _logical = placement.reserve(1).unwrap();
        let _physical = placement.physical().reserve(1).unwrap();
        assert_eq!(runs.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn reservation_transfer_and_drop_conserve_all_charges() {
        for bound in 0..8 {
            for requested in 0..10 {
                let placement = placement();
                let device = device_of(&placement).clone();
                device.set_bound(Some(bound));
                let reservation = placement.reserve(requested);
                if requested > bound {
                    assert!(reservation.is_err());
                    assert_eq!(device.view().charged, 0);
                    assert_eq!(device.store.claim(), 0);
                    continue;
                }
                for transferred in 0..=requested {
                    let placement = self::placement();
                    let device = device_of(&placement).clone();
                    device.set_bound(Some(bound));
                    let mut reservation = placement.reserve(requested).unwrap();
                    let charge = reservation.take(transferred);
                    assert_eq!(device.view().charged, requested);
                    drop(reservation);
                    assert_eq!(device.view().charged, transferred);
                    assert_eq!(device.store.claim(), transferred);
                    drop(charge);
                    assert_eq!(device.view().charged, 0);
                    assert_eq!(device.store.claim(), 0);
                }
            }
        }
    }

    #[test]
    fn resized_charge_transfers_only_growth_and_releases_shrink() {
        let placement = placement();
        let device = device_of(&placement).clone();
        device.set_bound(Some(24));
        let mut initial = placement.reserve(16).unwrap();
        let mut charge = initial.take(16);
        drop(initial);
        assert_eq!(device.view().charged, 16);
        let mut growth = placement.reserve(8).unwrap();
        growth.extend(&mut charge, 8);
        drop(growth);
        assert_eq!(charge.bytes(), 24);
        assert_eq!(device.view().charged, 24);
        charge.shrink(8);
        assert_eq!(device.view().charged, 8);
        drop(charge);
        assert_eq!(device.view().charged, 0);
    }

    #[test]
    fn a_refused_reservation_is_read_as_the_demand_it_made_until_the_mark_restarts() {
        let placement = placement();
        let device = device_of(&placement).clone();
        device.set_bound(Some(10));
        let held = placement.reserve(6).unwrap();
        assert_eq!(device.view().refused, None);
        assert!(placement.reserve(7).is_err());
        assert!(placement.reserve(5).is_err());
        let view = device.view();
        assert_eq!((view.charged, view.high_water), (6, 6));
        // The largest refused demand: what was charged and what was asked.
        assert_eq!(view.refused, Some(13));
        device.restart_high_water();
        assert_eq!(device.view().refused, None);
        drop(held);
    }

    #[test]
    fn overflow_and_lower_bound_leave_existing_reservations_intact() {
        let placement = placement();
        let device = device_of(&placement).clone();
        let reservation = placement.reserve(u64::MAX).unwrap();
        assert!(placement.reserve(1).is_err());
        device.set_bound(Some(0));
        assert_eq!(device.view().charged, u64::MAX);
        assert_eq!(device.view().bound, Some(0));
        assert!(placement.reserve(1).is_err());
        drop(reservation);
        assert_eq!(device.view().charged, 0);
    }

    #[test]
    fn lower_bound_blocks_growth_until_charges_fall_below_it() {
        let placement = placement();
        let device = device_of(&placement).clone();
        let first = placement.reserve(4).unwrap();
        let second = placement.reserve(4).unwrap();
        device.set_bound(Some(5));
        assert_eq!(device.view().charged, 8);
        assert_eq!(placement.reserve(1).err().unwrap().available, 0);
        drop(second);
        assert_eq!(device.view().charged, 4);
        assert_eq!(placement.reserve(2).err().unwrap().available, 1);
        let growth = placement.reserve(1).unwrap();
        drop(growth);
        drop(first);
        assert_eq!(device.view().charged, 0);
    }

    #[test]
    fn devices_sharing_a_store_charge_it_once_each_under_their_own_bounds() {
        let store = StoreLedger::new();
        let metal = Placement::on(store.clone());
        let cpu = Placement::on(store.clone());
        device_of(&metal).set_bound(Some(8));
        let mut gpu = metal.reserve(6).unwrap();
        let _gpu = gpu.take(6);
        let host = cpu.reserve(5).unwrap();
        assert_eq!(device_of(&metal).view().charged, 6);
        assert_eq!(device_of(&cpu).view().charged, 5);
        assert_eq!(store.claim(), 11);
        // The device bound constrains only this device's own charges.
        assert!(metal.reserve(3).is_err());
        drop(host);
        assert_eq!(store.claim(), 6);
        // A separate store is a separate ledger.
        let dedicated = Placement::on(StoreLedger::new());
        let _dedicated = dedicated.reserve(4).unwrap();
        assert_eq!(store.claim(), 6);
    }

    #[test]
    fn concurrent_reservations_cannot_both_spend_the_same_capacity() {
        let placement = placement();
        device_of(&placement).set_bound(Some(10));
        let ready = std::sync::Barrier::new(3);
        let release = std::sync::Barrier::new(3);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        let reservation = placement.reserve(6);
                        ready.wait();
                        release.wait();
                        reservation.is_ok()
                    })
                })
                .collect();
            ready.wait();
            assert_eq!(device_of(&placement).view().charged, 6);
            release.wait();
            assert_eq!(
                workers
                    .into_iter()
                    .map(|worker| u32::from(worker.join().unwrap()))
                    .sum::<u32>(),
                1
            );
        });
        assert_eq!(device_of(&placement).view().charged, 0);
    }
}
