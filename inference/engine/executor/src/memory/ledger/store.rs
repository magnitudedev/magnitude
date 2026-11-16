//! Where a store's ledger is found and what a load allocated in it: the
//! scope's ledgers ([`Ledgers`]), a device's view of its stores
//! ([`DeviceLedgers`]), and the Seismic allocations a load holds
//! ([`Held`], [`Recorded`]).

use super::*;
use std::num::NonZeroU64;

/// Where ledgers are found: one store ledger per store identity, shared by
/// every load in the scope. The product uses the process's
/// ([`Self::process`]); a test may use one of its own ([`Self::new`]),
/// sharing nothing with other tests. A store has one ledger in a scope by
/// construction: there is no other way to make one.
pub struct Ledgers {
    pub(super) stores: Mutex<HashMap<seismic::StoreIdentity, Weak<Store>>>,
    pub(super) changes: Arc<Changes>,
}

impl Ledgers {
    fn of(changes: Changes) -> Self {
        Self { stores: Mutex::default(), changes: Arc::new(changes) }
    }

    /// A test's scope of its own, no OS behind its stores: a record its
    /// test makes outside a change is a change of its own.
    #[cfg(any(test, feature = "fixtures"))]
    pub fn new() -> Self {
        Self::of(Changes { alone: std::sync::atomic::AtomicBool::new(true), ..Changes::default() })
    }

    /// A test's scope of its own under the process's rule: a record
    /// outside a change panics.
    #[cfg(test)]
    pub(crate) fn strict() -> Self {
        Self::of(Changes::default())
    }

    /// The changes of this scope's stores ([`Changes`]).
    pub fn changes(&self) -> &Arc<Changes> {
        &self.changes
    }

    /// The process's scope.
    pub fn process() -> &'static Self {
        static PROCESS: OnceLock<Ledgers> = OnceLock::new();
        PROCESS.get_or_init(|| Ledgers::of(Changes::default()))
    }

    /// The ledger in this scope of each of `stores`, as Seismic charges
    /// it: what a device opens charging, the stores it reaches
    /// ([`seismic::Devices::reached_stores`], [`seismic::Devices::open_in`]).
    /// Through it Seismic charges only its device's own entries
    /// ([`seismic::memory::Store::own`]), which opening allocates before any
    /// reading of the store; every claim is a [`DeviceLedgers`]', which
    /// exists only once its stores are read ([`UnreadLedgers::read`]).
    pub fn stores(&self, stores: Vec<seismic::StoreIdentity>) -> Vec<Arc<dyn seismic::memory::Store>> {
        let shape = Arc::new(LoadShape { stores: Vec::new() });
        stores.into_iter().map(|identity| Arc::new(self.view(identity, shape.clone())) as Arc<dyn seismic::memory::Store>).collect()
    }

    /// `store`'s one ledger in this scope, as a load of `shape` sees it:
    /// a test's, which reads it before it claims in it.
    #[cfg(test)]
    pub(crate) fn ledger(&self, store: seismic::StoreIdentity, shape: Arc<LoadShape>) -> Ledger {
        self.view(store, shape)
    }

    /// `store`'s one ledger in this scope, as a load of `shape` sees it,
    /// read or not: given out only as Seismic's charging view
    /// ([`Self::stores`]) or inside ledgers not yet read
    /// ([`UnreadLedgers`]).
    fn view(&self, store: seismic::StoreIdentity, shape: Arc<LoadShape>) -> Ledger {
        let mut stores = self.stores.lock().expect("ledger scope lock poisoned");
        let found = stores.get(&store).and_then(Weak::upgrade);
        let store = found.unwrap_or_else(|| {
            let made = Arc::new(Store { identity: store.clone(), state: Mutex::new(State::default()), changes: self.changes.clone(), pressure: Arc::default() });
            stores.insert(store, Arc::downgrade(&made));
            made
        });
        Ledger { store, shape }
    }
}

/// The ledgers of the stores one opened device reaches before their first
/// reading ([`DeviceLedgers::open`]). A store's grant is read through its
/// opened device, so a device's own memory is charged as it opens, before
/// any reading of its stores ([`Ledgers::stores`]); a judgment needs a
/// reading. These hold no claim and judge nothing: their one use is the
/// first reading ([`Self::read`]), which gives the [`DeviceLedgers`] every
/// claim is made in. So no claim of a load is judged in a store never read.
#[must_use = "a device's ledgers claim only once read"]
pub struct UnreadLedgers(DeviceLedgers);

impl UnreadLedgers {
    /// The changes of the scope these ledgers are in, which the first
    /// sample is taken beside ([`Sample::settled`]).
    pub fn changes(&self) -> &Arc<Changes> {
        self.0.changes()
    }

    /// The ledgers read from `sample`, one instant's ([`DeviceLedgers::read`]),
    /// and the reading: the first a load holds.
    pub fn read(self, device: &seismic::Device, sample: &Sample) -> (DeviceLedgers, DeviceReading) {
        let reading = self.0.read(device, sample);
        (self.0, reading)
    }

    /// [`Self::read`] where the device's grant is `grant`
    /// ([`DeviceLedgers::read_granted`]): a test's, with no device opened.
    #[cfg(any(test, feature = "fixtures"))]
    pub fn read_granted(self, sample: &Sample, grant: DeviceGrant) -> (DeviceLedgers, DeviceReading) {
        let reading = self.0.read_granted(sample, grant);
        (self.0, reading)
    }

    /// The ledgers read now, each store at `reading`, a test's scripted
    /// figures ([`Ledger::read`]), and the reading.
    #[cfg(test)]
    pub(crate) fn read_scripted(self, reading: Reading) -> (DeviceLedgers, DeviceReading) {
        let at = self.0.device.sampled(Instant::now());
        let device = self.0.device.read(at, reading);
        let host = (!Arc::ptr_eq(&self.0.device.store, &self.0.host.store)).then(|| self.0.host.read(at, reading));
        (self.0, DeviceReading { device, host })
    }
}

/// The ledgers of the stores one opened device reaches, built when the
/// device opens and passed down its load: its own memory's (device
/// allocations) and host RAM's (shared and host-tier allocations). On
/// unified memory both are the one store, so both are one ledger.
#[derive(Clone)]
pub struct DeviceLedgers {
    pub(super) device: Ledger,
    pub(super) host: Ledger,
}

impl DeviceLedgers {
    /// The ledgers `device` reaches in the process's scope, as a load of
    /// `shape` sees them, before their first reading.
    pub fn open(device: &seismic::Device, shape: Arc<LoadShape>) -> UnreadLedgers {
        Self::open_in(Ledgers::process(), device, shape)
    }

    /// The ledgers `device` reaches in `scope`: a test's own, so tests
    /// running together share no store.
    ///
    /// # Panics
    /// Where `device`'s own memory is a store apart from host RAM and its
    /// backend reads that store's grant by no mechanism that shows the
    /// process's changes in step with what the ledger counts
    /// ([`GrantSource`]).
    pub fn open_in(scope: &Ledgers, device: &seismic::Device, shape: Arc<LoadShape>) -> UnreadLedgers {
        let view = |contract, shape| scope.view(placed_store(device, contract), shape);
        let ledgers = Self { device: view(seismic::Contract::DeviceOnly, shape.clone()), host: view(seismic::Contract::Shared, shape) };
        if !Arc::ptr_eq(&ledgers.device.store, &ledgers.host.store) {
            GrantSource::of(device);
        }
        UnreadLedgers(ledgers)
    }

    /// These ledgers as a load of `shape` sees them: the same stores, read
    /// as these are.
    pub fn shaped(&self, shape: Arc<LoadShape>) -> Self {
        let view = |ledger: &Ledger| Ledger { store: ledger.store.clone(), shape: shape.clone() };
        Self { device: view(&self.device), host: view(&self.host) }
    }

    /// Whether the stores `device` reaches hold `demand` beside every
    /// committed claim, at `reading`'s OS figures with every holding since
    /// counted: [`Self::claim_at`]'s check, holding nothing.
    fn fits(&self, reading: &DeviceReading, device: &seismic::Device, demand: crate::memory::ClaimDemand) -> Result<(), Short> {
        let (on_device, on_host) = self.placed(device, demand);
        for ((ledger, held), bytes) in self.judged(Some(reading)).into_iter().zip([on_device, on_host]) {
            if bytes > 0 {
                ledger.covers(held, bytes)?;
            }
        }
        Ok(())
    }

    /// Each store's ledger and the reading a decision there is judged at:
    /// `reading`'s, which a load holds, or the latest where none.
    fn judged<'a>(&'a self, reading: Option<&'a DeviceReading>) -> [(&'a Ledger, Option<&'a Estimate>); 2] {
        let host = reading.map(|reading| reading.host.as_ref().unwrap_or(&reading.device));
        [(&self.device, reading.map(|reading| &reading.device)), (&self.host, host)]
    }

    /// Claim `demand`, an allocation of `kind` about to be made, in the
    /// stores `device` reaches ([`Self::placed`]), each claim admitted by
    /// comparing its new bytes against its store's free room, cache
    /// yielding to it, by the store's latest reading: a domain's claim as
    /// it serves. A load claims at the reading it holds ([`Self::claim_at`]).
    /// Seismic charges the allocation's exact bytes to each claim's entry as
    /// it makes it ([`Held::allocate`]), and frees them as it frees it.
    /// Refused, nothing is held.
    ///
    /// # Panics
    /// For a sequence's state, which is claimed by its rows alone.
    pub fn claim(&self, device: &seismic::Device, kind: CommittedKind, demand: crate::memory::ClaimDemand) -> Result<Held, Short> {
        self.claim_judged(None, device, kind, demand)
    }

    /// [`Self::claim`] at `reading`, the one a load holds from its start
    /// ([`Self::read`]): its OS figures fixed, every holding since in the
    /// scope's stores counted now, so a load's claims read one reading and
    /// no concurrent reading, resize or pressure event moves what it was
    /// planned against (design §5).
    pub fn claim_at(&self, reading: &DeviceReading, device: &seismic::Device, kind: CommittedKind, demand: crate::memory::ClaimDemand) -> Result<Held, Short> {
        self.claim_judged(Some(reading), device, kind, demand)
    }

    fn claim_judged(&self, reading: Option<&DeviceReading>, device: &seismic::Device, kind: CommittedKind, demand: crate::memory::ClaimDemand) -> Result<Held, Short> {
        assert!(!matches!(kind, CommittedKind::State(_)), "a sequence's state is claimed by its rows");
        let (on_device, on_host) = self.placed(device, demand);
        let unified = Arc::ptr_eq(&self.device.store, &self.host.store);
        let mut claims = Vec::new();
        let mut unclaimed = Vec::new();
        for (index, ((ledger, held), bytes)) in self.judged(reading).into_iter().zip([on_device, on_host]).enumerate() {
            if bytes > 0 {
                claims.push(ledger.claim_at(held, kind, bytes)?);
            } else if !(unified && index == 1) && !Holding::Committed(kind).pooled() {
                unclaimed.push((ledger.clone(), kind));
            }
        }
        Ok(Held { claims, unclaimed })
    }

    /// The bytes `demand` places in the device's own memory's store and in
    /// host RAM's where it is a store apart: each contract's bytes in the
    /// store its storage lies in, a host tier's backing in host RAM, and
    /// the bytes the backend stages through host RAM there too. On unified
    /// memory every byte is the device's.
    pub(crate) fn placed(&self, device: &seismic::Device, demand: crate::memory::ClaimDemand) -> (u64, u64) {
        let (mut on_device, mut on_host) = (0u64, 0u64);
        let mut place = |contract, bytes: u64| {
            if bytes == 0 {
                return;
            }
            let into = match Arc::ptr_eq(&self.of_contract(device, contract).store, &self.device.store) {
                true => &mut on_device,
                false => &mut on_host,
            };
            *into = into.checked_add(bytes).expect("a load's demand fits u64");
        };
        place(seismic::Contract::DeviceOnly, demand.device);
        place(seismic::Contract::Upload, demand.upload);
        place(seismic::Contract::Shared, demand.shared.checked_add(demand.host).expect("a load's demand fits u64"));
        if !Arc::ptr_eq(&self.device.store, &self.host.store) {
            on_host = on_host.checked_add(demand.staged).expect("a load's demand fits u64");
        }
        (on_device, on_host)
    }

    /// The bytes `demand` places in the store `tier` lies in
    /// ([`Self::placed`]).
    pub fn placed_in(&self, device: &seismic::Device, demand: crate::memory::ClaimDemand, tier: BankTier) -> u64 {
        let (on_device, on_host) = self.placed(device, demand);
        match Arc::ptr_eq(&self.of(tier).store, &self.device.store) {
            true => on_device,
            false => on_host,
        }
    }

    /// The most of `least..=most` whose `demand` fits, by halving at
    /// `reading`, the one the load holds, claimed as `kind` there
    /// ([`Self::claim_at`]), nothing allocated: `demand` never demands less
    /// of a larger count, and an error of it ends the search as it is.
    /// Short as `least` is where not even it fits.
    pub fn admit_most<E>(
        &self,
        reading: &DeviceReading,
        device: &seismic::Device,
        kind: CommittedKind,
        least: u64,
        most: u64,
        demand: impl Fn(u64) -> Result<crate::memory::ClaimDemand, E>,
    ) -> Result<Result<(u64, Held), Short>, E> {
        assert!(least <= most, "invariant violated: an admission's least count exceeds its most");
        let fitting = match self.fits(reading, device, demand(most)?) {
            Ok(()) => most,
            Err(_) => {
                if let Err(short) = self.fits(reading, device, demand(least)?) {
                    return Ok(Err(short));
                }
                // `fitting` fits, `over` does not.
                let (mut fitting, mut over) = (least, most);
                while over - fitting > 1 {
                    let middle = fitting + (over - fitting) / 2;
                    match self.fits(reading, device, demand(middle)?) {
                        Ok(()) => fitting = middle,
                        Err(_) => over = middle,
                    }
                }
                fitting
            }
        };
        Ok(self.claim_at(reading, device, kind, demand(fitting)?).map(|held| (fitting, held)))
    }

    /// Read every store `device` reaches from `sample`, one instant's, and
    /// return each store's estimate as the reading leaves it
    /// ([`Ledger::read`]), the value a load prices against: the
    /// driver's own memory brought up to date in it before it was stamped
    /// ([`Sample::try_now`]): host RAM by the
    /// host's figures, on unified memory bounded by the device's working
    /// set under the wired ceiling (Metal's
    /// `recommendedMaxWorkingSetSize`, its grant); the device's own memory,
    /// where it is a store of its own, by what its grant leaves of the
    /// store's capacity, read in step with what the ledger counts
    /// ([`GrantSource`]).
    ///
    /// # Panics
    /// Where a device whose memory is its own reads no grant, or one it
    /// failed to read: every backend with such memory reads one for an
    /// opened device (CUDA's free figure, Vulkan's memory budget, which its
    /// floor requires).
    pub fn read(&self, device: &seismic::Device, sample: &Sample) -> DeviceReading {
        self.read_after(device, sample, false)
    }

    /// [`Self::read`], `after_idle` marking the first unpressured reading
    /// after an unmonitored idle sleep.
    pub fn read_after(&self, device: &seismic::Device, sample: &Sample, after_idle: bool) -> DeviceReading {
        let topology = device.topology();
        let path = device
            .info()
            .paths(seismic::Contract::DeviceOnly)
            .first()
            .and_then(|path| topology.access_path(*path))
            .expect("an opened device places device storage");
        let grant = match Arc::ptr_eq(&self.device.store, &self.host.store) {
            true => match path.grant() {
                None => DeviceGrant::Host,
                Some(grant) => match grant.by {
                    seismic::ReportedBy::MetalRecommendedWorkingSet
                    | seismic::ReportedBy::MetalWorkingSetBeyondDedicated => {
                        DeviceGrant::Wired { working_set: grant.value }
                    }
                    seismic::ReportedBy::CudaTotalMemory
                    | seismic::ReportedBy::CudaUsableMemory
                    | seismic::ReportedBy::CudaTotalBeyondDedicated => {
                        DeviceGrant::HostLimit { limit: grant.value }
                    }
                    seismic::ReportedBy::VulkanHeap
                    | seismic::ReportedBy::VulkanHeaps
                    | seismic::ReportedBy::VulkanHeapsBeyondDedicated => {
                        let left = sample
                            .grant
                            .unwrap_or_else(|| panic!("invariant violated: {} read no grant with its usage", device.info().selector()))
                            .left
                            .min(grant.value);
                        DeviceGrant::HostBudget { capacity: grant.value, left }
                    }
                    other => panic!(
                        "invariant violated: {} reports an unexpected host-memory grant {other:?}",
                        device.info().selector()
                    ),
                },
            },
            false => {
                let capacity = topology.store(path.store()).expect("a path's store is of its revision").capacity().value;
                let free = match (GrantSource::of(device), sample.grant) {
                    (GrantSource::SameQuery, Some(grant)) => grant.left,
                    (GrantSource::SameQuery, None) => panic!("invariant violated: {} read no grant with its usage", device.info().selector()),
                    (GrantSource::Synchronous, _) => match sample.status().grant_left(path) {
                        Some(Ok(left)) => left.value,
                        Some(Err(error)) => panic!("invariant violated: {} read no grant of its own memory: {error}", device.info().selector()),
                        None => panic!("invariant violated: an opened device whose memory is its own reports no grant"),
                    },
                };
                DeviceGrant::Own { capacity, free }
            }
        };
        self.read_granted_after(sample, grant, after_idle)
    }

    /// [`Self::read`] of `sample` where the device's grant is `grant`: each
    /// store's reading folded into its ledger. A first reading is
    /// [`UnreadLedgers::read_granted`]'s.
    ///
    /// # Panics
    /// Where `grant` is of the other kind of store than these ledgers keep:
    /// unified grants for one store, a store's own for two.
    pub(crate) fn read_granted(&self, sample: &Sample, grant: DeviceGrant) -> DeviceReading {
        self.read_granted_after(sample, grant, false)
    }

    pub(crate) fn read_granted_after(&self, sample: &Sample, grant: DeviceGrant, after_idle: bool) -> DeviceReading {
        let at = sample.at();
        match (Arc::ptr_eq(&self.device.store, &self.host.store), grant) {
            (true, DeviceGrant::Host) => {
                DeviceReading { device: self.host.read_after(at, Reading::host(sample.host()), after_idle), host: None }
            }
            (true, DeviceGrant::Wired { working_set }) => {
                DeviceReading { device: self.host.read_after(at, Reading::unified(sample.host(), working_set), after_idle), host: None }
            }
            (true, DeviceGrant::HostLimit { limit }) => {
                DeviceReading { device: self.host.read_after(at, Reading::host_limit(sample.host(), limit), after_idle), host: None }
            }
            (true, DeviceGrant::HostBudget { capacity, left }) => {
                DeviceReading { device: self.host.read_after(at, Reading::host_budget(sample.host(), capacity, left), after_idle), host: None }
            }
            (false, DeviceGrant::Own { capacity, free }) => {
                let host = self.host.read_after(at, Reading::host(sample.host()), after_idle);
                let lag = own_lag(&self.device.store.identity, sample.host());
                DeviceReading { device: self.device.read_after(at, Reading::device(capacity, free, lag), after_idle), host: Some(host) }
            }
            (unified, grant) => {
                let stores = if unified { "one store" } else { "two stores" };
                panic!("invariant violated: a device grant {grant:?} read into ledgers of {stores}")
            }
        }
    }

    /// The ledgers of host RAM's store in `scope`, as a unified device
    /// reaches it (both views one ledger), for a load of `shape`: a test's,
    /// read from a recorded sample ([`Sample::scripted`],
    /// [`UnreadLedgers::read_granted`]) with no device opened.
    #[cfg(any(test, feature = "fixtures"))]
    pub fn unified_in(scope: &Ledgers, shape: Arc<LoadShape>) -> UnreadLedgers {
        let host = scope.view(seismic::StoreIdentity::HostRam, shape);
        UnreadLedgers(Self { device: host.clone(), host })
    }

    /// The changes of the scope these ledgers are in, which a sample paired
    /// with them is taken beside ([`Sample::try_now`]).
    pub fn changes(&self) -> &Arc<Changes> {
        &self.device.store.changes
    }

    /// The device's own memory's ledger: host RAM's on unified memory.
    pub fn device(&self) -> &Ledger {
        &self.device
    }

    /// Host RAM's ledger.
    pub fn host(&self) -> &Ledger {
        &self.host
    }

    /// The granule pool of `class` of the store `device` places `contract`
    /// in, opened at `granule`: its blocks whole granules the backend's
    /// allocations waste least on ([`seismic::block_bytes`],
    /// [`seismic::Device::native_allocations`]), each charged once to that
    /// pool's entry, what the backend allocates for it
    /// ([`seismic::BlockSize::held`], [`Ledger::pool_block`]). A load opens
    /// each store's pool of a class once.
    pub fn open_pool(&self, device: &seismic::Device, contract: seismic::Contract, class: PoolClass, granule: u64) -> Result<seismic::Pool, String> {
        let sizing = seismic::BlockSizing { granule, allocations: device.native_allocations() };
        let block = seismic::block_bytes(sizing).map_err(|error| format!("the {contract:?} store's {class:?} pool blocks: {error:?}"))?;
        let entries = self.of_contract(device, contract).pool_block(class, NonZeroU64::new(block.held).expect("a block holds a granule"));
        device.pool(contract, granule, block.bytes, entries).map_err(|error| format!("the {contract:?} store's {class:?} pool: {error}"))
    }

    /// The ledger of the store `device` places `contract`'s fixed storage
    /// in: one of these, by construction of the device's ledgers.
    pub fn of_contract(&self, device: &seismic::Device, contract: seismic::Contract) -> &Ledger {
        let store = placed_store(device, contract);
        self.keeping(&store)
            .unwrap_or_else(|| panic!("invariant violated: {contract:?} storage lies in a store the device's ledgers do not keep"))
    }

    /// The ledger of `store`, where these keep one.
    pub fn keeping(&self, store: &seismic::StoreIdentity) -> Option<&Ledger> {
        [&self.device, &self.host].into_iter().find(|ledger| ledger.store.identity == *store)
    }

    /// The ledger of the store `tier` lies in.
    pub fn of(&self, tier: BankTier) -> &Ledger {
        match tier {
            BankTier::Bank => &self.device,
            BankTier::Host => &self.host,
        }
    }

    /// The claim `load`'s host tier backs its load-time slots in, split from
    /// the load's `weights` ([`Held::split_backing`]): exactly where `load`
    /// holds host slots, as its bank opens with them (`BankCache::opened`),
    /// of the tier's bytes above its floor
    /// ([`crate::planning::ModelLoadPlan::host_tier_bytes`]). Those may be
    /// none: a tier is charged at least its landing reserve, in whole
    /// granules, so its first slots may cost nothing more.
    pub fn host_backing(&self, weights: &mut Held, load: &crate::planning::ModelLoadPlan) -> Result<Option<Claim>, String> {
        if load.expert_bank().is_none_or(|bank| bank.host_slots == 0) {
            return Ok(None);
        }
        Ok(Some(weights.split_backing(&self.host, load.host_tier_bytes()?)))
    }

    /// Split the load-time pooled records of `load`'s expert bank and banked
    /// stored tables out of `weights` as [`CommittedKind::Backing`] claims
    /// before `weights` allocates its non-pooled tensors: the pool's entry
    /// allocates their granules (`Holding::Pool`), so non-pooled buffer
    /// rounding and temporary import workspaces never consume the bank's
    /// unallocated record reserve before `ServedBank::new` carves it.
    pub fn split_records(&self, weights: &mut Held, load: &crate::planning::ModelLoadPlan) -> Result<(), String> {
        let Some(bank) = load.expert_bank() else {
            return Ok(());
        };
        let on_bank = bank
            .resident_charge(bank.slots)?
            .checked_add(bank.tables_charge()?)
            .ok_or_else(|| "bank record charge overflows".to_owned())?;
        if on_bank > 0 {
            let backing = weights.split_backing(self.of(BankTier::Bank), on_bank);
            weights.claims.push(backing);
        }
        let on_host = bank.host_charge(0)?;
        if on_host > 0 {
            let backing = weights.split_backing(self.of(BankTier::Host), on_host);
            weights.claims.push(backing);
        }
        Ok(())
    }
}

/// What a device's own memory's store grants at a sample, beside host
/// RAM's figures ([`DeviceLedgers::read`]).
#[derive(Clone, Copy, Debug)]
pub enum DeviceGrant {
    /// Host RAM with no separate device ceiling (CPU).
    Host,
    /// macOS unified Metal: host RAM bounded by the kernel wired ceiling and
    /// `recommendedMaxWorkingSetSize`.
    Wired { working_set: u64 },
    /// Unified host RAM bounded by a static device ceiling (unified CUDA's
    /// total or usable memory on Linux or Windows).
    HostLimit { limit: u64 },
    /// Unified host RAM bounded by a Vulkan heap budget (`left` left of
    /// `capacity` beyond current heap usage).
    HostBudget { capacity: u64, left: u64 },
    /// A store of the device's own: `free` of its `capacity` that its grant
    /// leaves, read in step with what the ledger counts (`GrantSource`).
    Own { capacity: u64, free: u64 },
}

/// Each store's estimate as one reading of a device's stores left it
/// ([`DeviceLedgers::read`]): a value its holder keeps, so pricing it needs
/// no later state of the ledgers.
#[derive(Clone, Debug)]
pub struct DeviceReading {
    /// The device's own memory's store: host RAM's on unified memory.
    pub(super) device: Estimate,
    /// Host RAM's, where it is a store apart from the device's.
    pub(super) host: Option<Estimate>,
}

impl DeviceReading {
    /// A reading of the two stores' estimates, as a test or fixture sets
    /// them from scripted readings.
    #[cfg(any(test, feature = "fixtures"))]
    pub fn of(device: Estimate, host: Option<Estimate>) -> Self {
        Self { device, host }
    }

    /// The estimates a load whose in-process figure is `in_process`, not
    /// yet held in its store, prices at this reading once `ledgers` hold
    /// what they hold now ([`Ledger::room_after`] in each store): the
    /// reading's OS figures fixed, and every holding since it, the load's
    /// own claims and other loads' alike, counted; the figure in the
    /// device's own store, where the load's state lies. With nothing held
    /// since, each [`Estimate::room`] is the reading's own less the figure:
    /// the room the load's plan settles into (`PlanningRoom`), its search
    /// and its claims read, the one authority on room (design §5). Reads no
    /// OS figure.
    pub fn room_after(&self, ledgers: &DeviceLedgers, in_process: u64) -> LoadEstimate {
        let device = ledgers.device.room_after(&self.device, in_process);
        let host = self.host.as_ref().map(|host| ledgers.host.room_after(host, 0));
        LoadEstimate { device, host }
    }
}

impl LoadEstimate {
    /// The estimate of the store `tier` lies in: the device's own memory's
    /// for the bank, host RAM's for a host tier (the device's on unified
    /// memory, where they are one store).
    pub fn of(&self, tier: BankTier) -> &Estimate {
        match tier {
            BankTier::Bank => &self.device,
            BankTier::Host => self.host.as_ref().unwrap_or(&self.device),
        }
    }
}

/// The estimates a load's room is read from, one per store its device
/// reaches ([`DeviceReading::room_after`]).
#[derive(Clone, Debug)]
pub struct LoadEstimate {
    /// The device's own memory's store, where its state and bank lie: host
    /// RAM's on unified memory.
    pub device: Estimate,
    /// Host RAM's, where it is a store apart from the device's (a discrete
    /// device: its host tier, staging and shared storage); none on unified
    /// memory, where the two are one store and `device` is it.
    pub host: Option<Estimate>,
}

/// The committed claims of one allocation across the stores a device
/// reaches ([`DeviceLedgers::claim`]): at most one per store. Kept for the
/// life of the allocation; dropped, its bytes are free.
#[must_use]
pub struct Held {
    claims: Vec<Claim>,
    /// Stores the device reaches where this claim initially placed 0 bytes:
    /// opened on demand inside [`Self::allocate`] if an allocation lands there
    /// (such as a discrete GPU's `Contract::Shared` / `Contract::Words` buffer
    /// during program preparation, qualification, or resource allocation).
    unclaimed: Vec<(Ledger, CommittedKind)>,
}

impl Held {
    /// `make`, the allocation on the owner's thread, as one change
    /// ([`Changes::change`]): every allocation it makes in a store the device
    /// reaches is charged to this holding's non-pooled claim there, at its
    /// exact bytes, as Seismic makes it ([`seismic::memory::charging`]).
    ///
    /// This is the only way an allocation is charged to a holding: it names
    /// every store the device reaches, those this holding placed no bytes in
    /// included, so no allocation it makes is left outside a charging scope.
    pub fn allocate<T>(&mut self, make: impl FnOnce() -> T) -> T {
        let changes = self
            .claims
            .first()
            .map(|claim| claim.ledger().store.changes.clone())
            .or_else(|| self.unclaimed.first().map(|(ledger, _)| ledger.store.changes.clone()));
        let Some(changes) = changes else {
            return make();
        };
        changes.change(|_| {
            let mut extra: Vec<Claim> = self
                .unclaimed
                .iter()
                .map(|(ledger, kind)| {
                    let id = ledger.state().insert(Holding::Committed(*kind), 0);
                    Claim::held(ledger.clone(), id, *kind)
                })
                .collect();
            let entries = self
                .claims
                .iter()
                .filter(|claim| !claim.pooled())
                .chain(extra.iter())
                .map(Claim::entry)
                .collect::<Vec<_>>();
            let outcome = seismic::memory::charging(&entries, make);
            drop(entries);
            let mut retained = Vec::new();
            for (claim, slot) in extra.drain(..).zip(std::mem::take(&mut self.unclaimed)) {
                if claim.allocated_bytes() > 0 || claim.bytes() > 0 {
                    self.claims.push(claim);
                } else {
                    drop(claim);
                    retained.push(slot);
                }
            }
            self.unclaimed = retained;
            outcome
        })
    }

    /// The claims, one per store.
    pub fn claims(&self) -> &[Claim] {
        &self.claims
    }

    /// The claims, to keep for the life of the allocation.
    pub fn into_claims(self) -> Vec<Claim> {
        self.claims
    }

    /// The bytes claimed in `store`.
    pub fn bytes_in(&self, store: &seismic::StoreIdentity) -> u64 {
        self.claims.iter().filter(|claim| claim.ledger().store.identity == *store).map(Claim::bytes).sum()
    }

    /// Split off `bytes` claimed in `ledger`'s store, before any is
    /// allocated, as a [`CommittedKind::Backing`] claim: memory the
    /// allocation backs after the rest is made (a host tier backed to its
    /// load-time slots, which its bank allocates as each slot backs). An
    /// allocation claiming nothing there claims no bytes there, and splits
    /// off a backing of none.
    ///
    /// # Panics
    /// Where it claimed fewer unallocated bytes there.
    pub fn split_backing(&mut self, ledger: &Ledger, bytes: u64) -> Claim {
        match self.claims.iter_mut().find(|claim| Arc::ptr_eq(&claim.ledger().store, &ledger.store)) {
            Some(claim) => claim.split_backing(bytes),
            None => {
                assert_eq!(bytes, 0, "invariant violated: a backing split from an allocation claiming nothing in {:?}", ledger.store.identity);
                let kind = CommittedKind::Backing(PoolClass::Records);
                let id = ledger.state().insert(Holding::Committed(kind), 0);
                Claim::held(ledger.clone(), id, kind)
            }
        }
    }
}

/// `free`, on the owner's thread, the free of what `claims` hold, and their
/// release, as one change ([`Changes::change`]); with no claims there is no
/// record to make.
pub fn free<R>(claims: Vec<Claim>, free: impl FnOnce() -> R) -> R {
    let Some(changes) = claims.first().map(|claim| claim.ledger().store.changes.clone()) else {
        return free();
    };
    changes.change(|_| {
        let freed = free();
        drop(claims);
        freed
    })
}

/// What a load allocated as it started: each allocation's claim, its exact
/// bytes charged to it by Seismic as it was made ([`Held::allocate`]), held
/// for the domain's life. The arena's allocations are claimed as
/// [`CommittedKind::Arena`] and gathered into its claim
/// ([`Self::claim_arena`]); the pooled holdings the load's granules back are
/// carved from the load's headroom ([`Self::carve`]), their granules the
/// pool entry's allocation. Nothing is recorded at first.
#[derive(Default)]
pub struct Recorded {
    claims: Vec<Claim>,
    /// The load's minimum request state, in each store that holds state
    /// ([`Self::record_state_floor`]): held for the domain's life.
    state_floors: Vec<Claim>,
}

impl Recorded {
    /// `floor`, the load's minimum request state claimed as
    /// [`CommittedKind::StateFloor`], kept for the domain's life.
    pub fn record_state_floor(&mut self, floor: Held) {
        for claim in floor.claims {
            assert_eq!(claim.kind(), CommittedKind::StateFloor, "invariant violated: a state floor recorded of another kind");
            self.state_floors.push(claim);
        }
    }

    /// `held`, an allocation the load made as it started, joins the record.
    ///
    /// # Panics
    /// Where it is not a load's, the arena's or the state tables'
    /// allocation.
    pub fn record(&mut self, held: Held) {
        for claim in held.claims {
            assert!(
                matches!(
                    claim.kind(),
                    CommittedKind::Load
                        | CommittedKind::Arena
                        | CommittedKind::StateTables
                        | CommittedKind::Backing(PoolClass::Records)
                ),
                "invariant violated: a load records only its own, its arena's, its state tables' and its record backings' allocations, not {:?}",
                claim.kind()
            );
            self.claims.push(claim);
        }
    }

    /// `bytes` of the load's claims in `store` as a pooled holding of
    /// `kind`: bytes the load was admitted for that its claims never
    /// allocated, which the store's pool backs as granules of its own
    /// allocation. Never admitted again.
    ///
    /// # Panics
    /// For a kind the pool does not back, whose bytes are charged where they
    /// are allocated; or where the load's claims there leave fewer bytes
    /// unallocated.
    pub fn carve(&mut self, store: &seismic::StoreIdentity, kind: CommittedKind, bytes: u64) -> Claim {
        assert!(Holding::Committed(kind).pooled(), "invariant violated: {kind:?} is charged where it is allocated, never carved");
        let mut left = bytes;
        let mut ledger = self
            .claims
            .iter()
            .chain(&self.state_floors)
            .find(|claim| claim.ledger().store.identity == *store)
            .map(|claim| claim.ledger().clone());
        for source in [CommittedKind::Backing(PoolClass::Records), CommittedKind::Load] {
            for claim in self
                .claims
                .iter()
                .filter(|claim| claim.kind() == source && claim.ledger().store.identity == *store)
            {
                let mut state = claim.ledger().state();
                let entry = *state.entry(claim.id());
                let taken = left.min(entry.bytes - entry.allocated);
                state.resize(claim.id(), entry.bytes - taken);
                left -= taken;
                ledger.get_or_insert_with(|| claim.ledger().clone());
            }
        }
        assert_eq!(left, 0, "invariant violated: a holding carves {bytes} bytes of a load's unallocated bytes in {store:?}");
        self.claims
            .retain(|claim| !(claim.kind() == CommittedKind::Backing(PoolClass::Records) && claim.bytes() == 0));
        let ledger = ledger.unwrap_or_else(|| panic!("invariant violated: a load records nothing in {store:?}"));
        let id = ledger.state().insert(Holding::Committed(kind), bytes);
        Claim::held(ledger, id, kind)
    }

    /// The arena's claim: the [`CommittedKind::Arena`] allocations the load
    /// recorded in `ledger`'s store, its graph scratch and bound constants,
    /// each holding exactly what its allocations charged
    /// ([`Claim::fit_allocated`]). A group's class beyond one place is
    /// claimed as the group opens, never here.
    pub fn claim_arena(&mut self, ledger: &Ledger) -> ArenaClaim {
        let store = ledger.store();
        let (mut arena, rest): (Vec<Claim>, Vec<Claim>) = std::mem::take(&mut self.claims)
            .into_iter()
            .partition(|claim| claim.kind() == CommittedKind::Arena && claim.ledger().store.identity == store);
        self.claims = rest;
        arena.iter_mut().for_each(Claim::fit_allocated);
        ArenaClaim::of(arena)
    }

    /// The record's claims, held for the domain's life.
    pub fn into_claims(self) -> Vec<Claim> {
        self.claims.into_iter().chain(self.state_floors).collect()
    }
}

/// Whether `device`'s own memory and host RAM are one store: unified memory,
/// where the device's room and host RAM share one ceiling. A discrete
/// device's memory is apart from host RAM's. The one rule a search's room
/// states it by ([`crate::RoomNow::host_shared`]).
pub(crate) fn shares_host(device: &seismic::Device) -> bool {
    placed_store(device, seismic::Contract::DeviceOnly) == placed_store(device, seismic::Contract::Shared)
}

/// The store `device` places `contract`'s fixed storage in.
fn placed_store(device: &seismic::Device, contract: seismic::Contract) -> seismic::StoreIdentity {
    let topology = device.topology();
    device
        .info()
        .paths(contract)
        .first()
        .and_then(|path| topology.access_path(*path))
        .and_then(|path| topology.store_identity(path.store()))
        .expect("an opened device places device, upload and shared storage")
}

/// How a device store whose memory is its own shows the process's changes
/// in the grant a reading takes, in step with what the ledger counts of it:
/// so no change the ledger counts is read again as others' (`IN_STEP`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GrantSource {
    /// Vulkan: the grant (budget less usage) comes from the one
    /// `VK_EXT_memory_budget` query the driver's own memory is settled from
    /// ([`seismic::Device::reconcile_driver_memory`]). A driver that shows a
    /// free late shows it late in both, so the freed bytes stay counted as
    /// the driver's own until it shows them free: the visibility is
    /// observed, never assumed.
    SameQuery,
    /// CUDA: `cuMemGetInfo`'s free figure and `cuMemAlloc`/`cuMemFree` are
    /// one synchronous driver count, so every change shows in the next
    /// reading.
    Synchronous,
}

impl GrantSource {
    /// `device`'s source.
    ///
    /// # Panics
    /// For a backend with memory of its own and neither mechanism.
    fn of(device: &seismic::Device) -> Self {
        match device.backend() {
            seismic::BackendName::Vulkan => Self::SameQuery,
            seismic::BackendName::Cuda => Self::Synchronous,
            backend => panic!("invariant violated: {backend:?} has memory of its own and reads no grant in step with its usage"),
        }
    }
}

/// The lag of a device store's grant behind what the ledger counts of it,
/// by either [`GrantSource`]: none.
const IN_STEP: seismic::Lag = seismic::Lag {
    release: std::time::Duration::ZERO,
    growth: std::time::Duration::ZERO,
};

/// How late a change of the process's shows in a reading of `store`, one
/// a device's ledgers keep, the lag [`DeviceLedgers::read`] reads it at:
/// host RAM's (unified memory's included) by the host's own mechanisms,
/// `host`'s lag ([`Reading::host`]); a device store whose memory is its own
/// in the next reading, its grant read in step with the change
/// (`IN_STEP`). A change made at `t` shows in a reading taken at or past
/// `t + λ` of its direction (`memory::journal`).
///
/// # Panics
/// For a volume: no ledger keeps one.
pub fn own_lag(store: &seismic::StoreIdentity, host: &seismic::HostSample) -> seismic::Lag {
    match store {
        seismic::StoreIdentity::HostRam => host.lag,
        seismic::StoreIdentity::DeviceMemory(_) => IN_STEP,
        seismic::StoreIdentity::Volume(_) => panic!("invariant violated: a volume's changes are read into no ledger"),
    }
}
