//! The ledger: the one record of what the process holds in each store.
//!
//! - **One ledger per store.** Unified (or device) memory, and host RAM
//!   where it is separate. Every byte the process holds there is one
//!   [`Entry`]: an owner kind, its bytes, its [`Holding`] (committed or
//!   cache), and how many of its bytes its holder has allocated.
//! - **Claimed, allocated, released.** Every allocation takes its entry
//!   first, admitted by comparing only its new bytes against free room;
//!   its holder marks the bytes allocated as it allocates them
//!   ([`Claim::allocated`]), and dropping the entry is the free. A reading
//!   attributes to others only what the process's allocated bytes do not
//!   account for, and free room excludes every entry, allocated or not: an
//!   entry taken but not yet allocated never reads as free, and memory
//!   allocated under an entry never reads as others'.
//! - **The only record.** An allocation is backed by an entry or it cannot
//!   be made: committed memory exists only as a [`Claim`], taken by
//!   [`Ledger::claim`] from a [`Need`], and cache only as a [`Cached`],
//!   taken by [`Ledger::cache`]. The allocation paths (the granule pool's
//!   blocks, the arena, weights) take one of them as their right to
//!   allocate, and dropping it is the free.
//! - **Every free shows at once.** Every store's frees are synchronous:
//!   host blocks are mapped directly (`munmap`, `VirtualFree`), and plain
//!   device allocations free at once on every backend (`cuMemFree`,
//!   `vkFreeMemory`; a released shared Metal buffer unwires within one
//!   reading interval, 66 ms for 4 GiB on an M4 Pro). Bytes a GPU fence
//!   still reads stay held in their own entry until the fence completes
//!   and the free returns: the holder drops its handle then. So what the
//!   ledger holds covers them.
//! - **Allocated is what exists.** An entry's allocated bytes are memory
//!   that exists for it: an allocation backed after it begins (a bank's
//!   growth) is allocated as it backs ([`Ledger::backing`]), and cache a
//!   claim takes in process stays allocated, releasing, until its holder
//!   frees it ([`Cached::released`]): the claim allocates those bytes only
//!   then ([`Snapshot::allocatable`]).
//! - **One OS input.** The ledger reads the OS only through [`Available`],
//!   which one function per OS (and per device backend) produces. "Used" is
//!   never read: it is `total − available` of that same reading, so the two
//!   cannot disagree.
//! - **One reading per decision.** Every figure a decision reads comes from
//!   one [`Snapshot`]: the entries and the reading at one instant, so no
//!   decision mixes readings.
//!
//! Sizes are never computed here. A [`Need`] names what is needed and with
//! what inputs; the assessment lane's formulas size it inside
//! [`Ledger::claim`] and [`Ledger::admit`] through the one private `size`,
//! the same functions plan and assessment predict with.
//!
//! Effect-free: no clocks are read and no OS is queried here. Readings are
//! handed in.

use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::Instant;

use crate::operation::RequestId;
use crate::planning::BankTier;

// ---------------------------------------------------------------------------
// The one OS input.
// ---------------------------------------------------------------------------

/// A store's available memory at one instant: the only OS memory figure the
/// process reads. Available means pages the OS can hand out without
/// swapping or compressing anyone (`seismic::HostStatus::allocatable`, which
/// says why the OSes differ). One function per OS (and per device backend)
/// produces it:
///
/// | Store | Reading |
/// |---|---|
/// | macOS | `free_count − speculative_count + purgeable_count` less `vm.vm_page_free_target` (`HOST_VM_INFO64`); `external_page_count` (file-backed) beside it, [`Available::reclaimable`] |
/// | Linux | `MemAvailable`, bounded by cgroup v2 `memory.max − memory.current` |
/// | Windows | `ullAvailPhys`, bounded by the job object's limit |
/// | CUDA device | `cuMemGetInfo` free |
/// | Vulkan device | `VK_EXT_memory_budget` budget less usage (the backend's floor requires the extension) |
///
/// Fields are private: `available ≤ total` holds by construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Available {
    total: u64,
    available: u64,
    /// Clean file-backed pages beyond `available` that a load may count once
    /// ([`seismic::Reclaim::CleanFileCache`]): `available + reclaimable ≤
    /// total` holds by construction. Zero where `available` already counts
    /// the file cache the OS reclaims first.
    reclaimable: u64,
    /// What a driver's grant on the process's allocations in the store
    /// leaves (Metal's working set less what it has allocated); none where
    /// no grant bounds the store.
    grant_left: Option<u64>,
    at: Instant,
}

/// A reading that does not satisfy `available ≤ total`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImpossibleReading;

impl Available {
    /// The reading, refused where it is not one.
    pub fn new(
        total: u64,
        available: u64,
        grant_left: Option<u64>,
        at: Instant,
    ) -> Result<Self, ImpossibleReading> {
        if available > total {
            return Err(ImpossibleReading);
        }
        Ok(Self { total, available, reclaimable: 0, grant_left, at })
    }

    /// This reading with `bytes` of clean file-backed pages a load may count
    /// once, no more than `total − available`.
    pub fn with_reclaimable(self, bytes: u64) -> Self {
        Self { reclaimable: bytes.min(self.total - self.available), ..self }
    }

    /// Host RAM's reading on this OS. The one place host memory figures
    /// are read.
    ///
    /// # Panics
    /// Where the OS reports more available than total: no reading.
    pub fn host(at: Instant) -> Result<Self, Blind> {
        let reading = seismic::host_available().map_err(|error| Blind(error.to_string()))?;
        let available = Self::new(reading.total, reading.available, None, at).unwrap_or_else(|ImpossibleReading| {
            panic!("invariant violated: host RAM reads {} available of {} total", reading.available, reading.total)
        });
        Ok(available.with_reclaimable(reading.reclaimable))
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn available(&self) -> u64 {
        self.available
    }

    /// The clean file-backed pages beyond [`Self::available`] a load may
    /// count once.
    pub fn reclaimable(&self) -> u64 {
        self.reclaimable
    }

    pub fn at(&self) -> Instant {
        self.at
    }
}

/// The available reading failed: the ledger cannot see the store.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Blind(pub String);

// ---------------------------------------------------------------------------
// Entries.
// ---------------------------------------------------------------------------

/// What committed memory is held for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CommittedKind {
    /// A sequence's state (KV), as it grows. The only committed kind a
    /// claim may take ([`Ledger::claim`]): an earlier sequence's growth
    /// preempts a later one's.
    State(RequestId),
    /// A streamed tier's landing rows.
    Landing,
    /// An optional component's memory (a draft head, a vision encoder):
    /// its weights, imported after the load started, and what binding it
    /// and its pools hold, for as long as it is resident.
    Component,
    /// A program set's qualification fixtures, held while its preparation
    /// runs.
    Qualification,
    /// An import's upload window, staged through host RAM where that is a
    /// store apart: held while the import runs.
    Upload,
    /// Retained launch outputs' slots, held while their pool keeps them.
    Outputs,
    /// The arena at load: workspace, pass buffers, graph slots, group
    /// classes.
    Arena,
    /// The expert bank's floor: one step's routed experts per layer.
    BankFloor,
    /// A measurement job's room, in another process.
    Measurement,
    /// A prompt group's scratch beyond the decode set the floor holds, from
    /// its open to its close: optional, taken only from what yields at once
    /// ([`Ledger::claim_units`]).
    Scratch,
    /// A prompt group's class beyond one place, from its open to its
    /// close: optional, as [`Self::Scratch`].
    GroupClass,
    /// What a load allocated as it started ([`Recorded`]), until the
    /// holdings that name it carve it out.
    Load,
    /// An allocation backed after it begins, allocated as its memory is
    /// backed and cache once it ends ([`Ledger::backing`]): a bank
    /// holding's growth, a host tier's backing to its load-time slots.
    Backing,
    /// One request's minimum state: what a load needs to serve beside its
    /// weights and its bank's floor. Held for the domain's life; each
    /// sequence's state is credited against it ([`State::raise`]), so a
    /// request's first rows draw from room the load already holds and no
    /// window opens in which it is lost.
    StateFloor,
}

impl CommittedKind {
    /// What a claim of this kind may count as room beyond the free memory
    /// above the OS's floor, stated by its kind: what a load allocates as it
    /// starts is the user's own action and takes the clean file cache once
    /// ([`seismic::Reclaim::CleanFileCache`]); every claim made while serving, a
    /// growth of the bank and a backing among them, takes only memory the OS
    /// hands out without displacing anyone ([`seismic::Reclaim::Nothing`]).
    pub fn reclaim(self) -> seismic::Reclaim {
        match self {
            Self::Load
            | Self::Arena
            | Self::BankFloor
            | Self::Landing
            | Self::Upload
            | Self::Outputs
            | Self::Component
            | Self::Qualification
            | Self::StateFloor => {
                seismic::Reclaim::CleanFileCache
            }
            Self::State(_) | Self::Measurement | Self::Scratch | Self::GroupClass | Self::Backing => seismic::Reclaim::Nothing,
        }
    }
}

/// What cache memory is held for. Cache fills all room no committed claim
/// holds, and yields to one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacheKind {
    /// Idle graph slots beyond the floor's, a closed prompt group's class
    /// until its pools release them. They yield just before the bank.
    GraphSlots,
    /// Expert bank slots above the floor.
    Bank,
    /// Free granules of blocks the granule pool holds.
    Spare,
    /// Retained prefixes.
    Prefix,
    /// An idle drafter.
    Drafter,
    /// An idle vision encoder.
    Vision,
}

/// An entry's state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holding {
    Committed(CommittedKind),
    Cache(CacheKind),
}

/// One holding of the process in one store: `bytes` held, `allocated` of
/// them allocated by its holder ([`Claim::allocated`]); cache is allocated
/// whole. `allocated ≤ bytes` always. `releasing` is memory a cache holding
/// yielded to a claim in process that its holder has not freed yet
/// ([`Cached::released`]): allocated still, held no more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Entry {
    pub holding: Holding,
    pub bytes: u64,
    pub allocated: u64,
    pub releasing: u64,
}

// ---------------------------------------------------------------------------
// Needs, and the formulas that size them.
// ---------------------------------------------------------------------------

/// What committed memory is needed for, with the inputs its formula sizes
/// it from, or its bytes where its caller holds them. The one input to
/// [`Ledger::claim`] and [`Ledger::admit`]: there are no kind-specific
/// claims. The arena is carved from what the load recorded allocating
/// ([`Recorded::claim_arena`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Need {
    /// The state `sequence`'s history takes as it grows from `from` rows
    /// to `to`: the pages `to` rows occupy beyond those `from` rows occupy
    /// ([`LoadShape::state_bytes`]), so its claim holds what its stores
    /// allocate for it.
    StateRows { sequence: RequestId, from: u64, to: u64 },
    /// A streamed tier's landing rows, at load, of `slot_bytes` each.
    LandingRows { tier: BankTier, rows: u64, slot_bytes: u64 },
    /// A bank holding's floor, at load: the bank plan's charge at it, which
    /// plan and assessment size the bank by too.
    BankFloor { bytes: u64 },
    /// A measurement job's room, as the job declares it.
    Measurement { bytes: u64 },
    /// One request's minimum state, at load: the state `rows` rows take, held
    /// for the domain's life ([`CommittedKind::StateFloor`]).
    StateFloor { rows: u64 },
}

impl Need {
    /// The kind a claim for this need holds.
    pub fn kind(&self) -> CommittedKind {
        match self {
            Self::StateRows { sequence, .. } => CommittedKind::State(*sequence),
            Self::LandingRows { .. } => CommittedKind::Landing,
            Self::BankFloor { .. } => CommittedKind::BankFloor,
            Self::Measurement { .. } => CommittedKind::Measurement,
            Self::StateFloor { .. } => CommittedKind::StateFloor,
        }
    }
}

/// The header-derived inputs a load's needs are sized from: each of its
/// state stores' shape. Never planner objects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadShape {
    pub stores: Vec<StoreShape>,
}

/// One state store's history domains, and the context and largest advance
/// the store lays their pages out at
/// ([`magnitude_state::HistoryDomainLayout::geometry`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreShape {
    pub domains: Vec<magnitude_state::HistoryDomainLayout>,
    pub context: usize,
    pub max_advance: usize,
}

impl StoreShape {
    fn of(state: &crate::StateStorePlan) -> Self {
        Self {
            domains: state.history.iter().map(|domain| domain.layout.clone()).collect(),
            context: state.context_rows,
            max_advance: state.max_advance,
        }
    }
}

impl LoadShape {
    /// The shape of the load `plan` places ([`Self::of_states`]). The one
    /// way a load's shape is built.
    pub fn of(plan: &crate::planning::ExecutionPlan) -> Self {
        Self::of_states(plan.resources().target_state(), plan.resources().head_state())
    }

    /// One request's minimum state in the load `plan` places: what its first
    /// step claims, its one row and the most speculative rows its drafter adds
    /// past it ([`crate::domain::claimed_through`]), so a load that holds it
    /// serves a request whatever the method.
    pub fn minimum_state(plan: &crate::planning::ExecutionPlan) -> u64 {
        Self::of(plan).state_bytes(crate::domain::claimed_through(0, 1, plan.policy().method()))
    }

    /// The history domains of every state store a load holds: the
    /// target's, and its drafter's own where it keeps one (an MTP or draft
    /// head of its own history). A sequence's rows advance both, so its
    /// state claim holds both; state no claim holds is memory the ledger
    /// cannot see.
    pub(crate) fn of_states(target: &crate::StateStorePlan, head: Option<&crate::StateStorePlan>) -> Self {
        Self { stores: std::iter::once(target).chain(head).map(StoreShape::of).collect() }
    }

    /// The state a history of `rows` rows occupies across the load's
    /// stores: in each, the whole pages its rows take
    /// ([`magnitude_state::batch_state_bytes`]), which is what the store
    /// allocates for them. A history grows by a page as its rows cross a
    /// page boundary.
    pub fn state_bytes(&self, rows: u64) -> u64 {
        self.stores
            .iter()
            .try_fold(0u64, |total, store| {
                let bytes = magnitude_state::batch_state_bytes(&store.domains, store.context, store.max_advance, rows)
                    .expect("a load's history domains size every history it admits");
                total.checked_add(bytes)
            })
            .expect("a load's state bytes fit u64")
    }
}

/// The bytes `need` takes: the one place any need's bytes come from. State
/// rows by the pages they occupy ([`LoadShape::state_bytes`]); the rest as
/// their callers hold them.
fn size(shape: &LoadShape, need: &Need) -> u64 {
    match need {
        Need::StateRows { from, to, .. } => {
            assert!(from <= to, "a sequence's state need grows its history, never shrinks it");
            shape.state_bytes(*to) - shape.state_bytes(*from)
        }
        Need::LandingRows { rows, slot_bytes, .. } => rows.checked_mul(*slot_bytes).expect("landing rows' bytes fit u64"),
        Need::Measurement { bytes } | Need::BankFloor { bytes } => *bytes,
        Need::StateFloor { rows } => shape.state_bytes(*rows),
    }
}

// ---------------------------------------------------------------------------
// The ledger.
// ---------------------------------------------------------------------------

/// Where ledgers are found: one store ledger per store identity, shared by
/// every load in the scope. The product uses the process's
/// ([`Self::process`]); a test may use one of its own ([`Self::new`]),
/// sharing nothing with other tests. A store has one ledger in a scope by
/// construction: there is no other way to make one.
#[derive(Default)]
pub struct Ledgers {
    stores: Mutex<HashMap<seismic::StoreIdentity, Weak<Store>>>,
}

impl Ledgers {
    /// A scope of its own.
    pub fn new() -> Self {
        Self::default()
    }

    /// The process's scope.
    pub fn process() -> &'static Self {
        static PROCESS: OnceLock<Ledgers> = OnceLock::new();
        PROCESS.get_or_init(Ledgers::new)
    }

    /// `store`'s one ledger in this scope, as a load of `shape` sees it.
    pub fn ledger(&self, store: seismic::StoreIdentity, shape: Arc<LoadShape>) -> Ledger {
        let mut stores = self.stores.lock().expect("ledger scope lock poisoned");
        let found = stores.get(&store).and_then(Weak::upgrade);
        let store = found.unwrap_or_else(|| {
            let made = Arc::new(Store { identity: store.clone(), state: Mutex::new(State::default()) });
            stores.insert(store, Arc::downgrade(&made));
            made
        });
        Ledger { store, shape }
    }
}

/// The ledgers of the stores one opened device reaches, built when the
/// device opens and passed down its load: its own memory's (device
/// allocations) and host RAM's (shared and host-tier allocations). On
/// unified memory both are the one store, so both are one ledger.
#[derive(Clone)]
pub struct DeviceLedgers {
    device: Ledger,
    host: Ledger,
}

impl DeviceLedgers {
    /// The ledgers `device` reaches in the process's scope, as a load of
    /// `shape` sees them.
    pub fn open(device: &seismic::Device, shape: Arc<LoadShape>) -> Self {
        Self::open_in(Ledgers::process(), device, shape)
    }

    /// The ledgers `device` reaches in `scope`: a test's own, so tests
    /// running together share no store.
    pub fn open_in(scope: &Ledgers, device: &seismic::Device, shape: Arc<LoadShape>) -> Self {
        Self {
            device: scope.ledger(placed_store(device, seismic::Contract::DeviceOnly), shape.clone()),
            host: scope.ledger(placed_store(device, seismic::Contract::Shared), shape),
        }
    }

    /// Whether the stores `device` reaches hold `demand` beside every
    /// committed claim, by their last readings: [`Self::claim`]'s check,
    /// holding nothing.
    fn fits(&self, device: &seismic::Device, kind: CommittedKind, demand: crate::memory::ClaimDemand) -> Result<(), Short> {
        let (on_device, on_host) = self.placed(device, demand);
        for (ledger, bytes) in [(&self.device, on_device), (&self.host, on_host)] {
            if bytes == 0 {
                continue;
            }
            ledger.covers(bytes, kind.reclaim())?;
        }
        Ok(())
    }

    /// Claim `demand`, an allocation of `kind` about to be made, in the
    /// stores `device` reaches ([`Self::placed`]), each claim admitted by
    /// comparing its new bytes against its store's free room, cache
    /// yielding to it, by the store's last reading; a store whose last
    /// reading failed admits nothing ([`Short::Blind`]). Its holder marks it
    /// allocated as the allocation is made ([`Held::allocated`]) and drops
    /// it as the allocation is freed. Refused, nothing is held.
    ///
    /// # Panics
    /// For a sequence's state, which is claimed by its rows alone.
    pub fn claim(&self, device: &seismic::Device, kind: CommittedKind, demand: crate::memory::ClaimDemand) -> Result<Held, Short> {
        assert!(!matches!(kind, CommittedKind::State(_)), "a sequence's state is claimed by its rows");
        let (on_device, on_host) = self.placed(device, demand);
        let mut claims = Vec::new();
        for (ledger, bytes) in [(&self.device, on_device), (&self.host, on_host)] {
            if bytes == 0 {
                continue;
            }
            match ledger.state().take_in(&ledger.store.identity, kind, bytes)? {
                Taken::Held(id) => claims.push(Claim { ledger: ledger.clone(), id, kind }),
                Taken::Preempting { .. } => unreachable!("only a sequence's growth preempts"),
            }
        }
        Ok(Held { claims })
    }

    /// The bytes `demand` places in the device's own memory's store and in
    /// host RAM's where it is a store apart: each contract's bytes in the
    /// store its storage lies in, a host tier's backing in host RAM, and
    /// the bytes the backend stages through host RAM there too. On unified
    /// memory every byte is the device's.
    fn placed(&self, device: &seismic::Device, demand: crate::memory::ClaimDemand) -> (u64, u64) {
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
        place(seismic::Contract::DeviceOnly, demand.device.checked_add(demand.reserved).expect("a load's demand fits u64"));
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

    /// The most of `least..=most` whose `demand` fits, by halving on the
    /// ledgers' last readings, claimed as `kind` ([`Self::claim`]), nothing
    /// allocated: `demand` never demands less of a larger count, and an
    /// error of it ends the search as it is. Short as `least` is where not
    /// even it fits.
    pub fn admit_most<E>(
        &self,
        device: &seismic::Device,
        kind: CommittedKind,
        least: u64,
        most: u64,
        demand: impl Fn(u64) -> Result<crate::memory::ClaimDemand, E>,
    ) -> Result<Result<(u64, Held), Short>, E> {
        assert!(least <= most, "invariant violated: an admission's least count exceeds its most");
        let fitting = match self.fits(device, kind, demand(most)?) {
            Ok(()) => most,
            Err(_) => {
                if let Err(short) = self.fits(device, kind, demand(least)?) {
                    return Ok(Err(short));
                }
                // `fitting` fits, `over` does not.
                let (mut fitting, mut over) = (least, most);
                while over - fitting > 1 {
                    let middle = fitting + (over - fitting) / 2;
                    match self.fits(device, kind, demand(middle)?) {
                        Ok(()) => fitting = middle,
                        Err(_) => over = middle,
                    }
                }
                fitting
            }
        };
        Ok(self.claim(device, kind, demand(fitting)?).map(|held| (fitting, held)))
    }

    /// Read every store `device` reaches, each by its one reading, at `at`:
    /// host RAM by `host`, its reading then (the live machine's
    /// [`Available::host`]), bounded on unified memory by the device's grant
    /// (Metal's working set); the device's own memory, where it is a store
    /// of its own, by what its grant leaves (`cuMemGetInfo`'s free, Vulkan's
    /// budget less usage), of the store's capacity. `status` is the device's
    /// status now, the one source of its grants.
    ///
    /// # Panics
    /// Where a device whose memory is its own reports no grant: every
    /// backend with such memory reads one for an opened device (CUDA's free
    /// figure, Vulkan's memory budget, which its floor requires).
    pub fn read(&self, device: &seismic::Device, status: &seismic::Status, host: Result<Available, Blind>, at: Instant) {
        let topology = device.topology();
        let path = device
            .info()
            .paths(seismic::Placing::Fixed(seismic::Contract::DeviceOnly))
            .first()
            .and_then(|path| topology.access_path(*path))
            .expect("an opened device places device storage");
        let left = || match status.grant_left(path) {
            None => Ok(None),
            Some(Ok(left)) => Ok(Some(left.value)),
            Some(Err(error)) => Err(Blind(error.to_string())),
        };
        if Arc::ptr_eq(&self.device.store, &self.host.store) {
            self.host.read(|| {
                let host = host?;
                Ok(Available { grant_left: left()?, ..host })
            });
            return;
        }
        self.host.read(|| host);
        let capacity = topology.store(path.store()).expect("a path's store is of its revision").capacity().value;
        self.device.read(|| {
            let available = left()?
                .unwrap_or_else(|| panic!("invariant violated: an opened device whose memory is its own reports no grant"));
            Ok(Available::new(capacity, available.min(capacity), None, at).expect("a grant bounded by the store's capacity is a reading"))
        });
    }

    /// The device's own memory's ledger: host RAM's on unified memory.
    pub fn device(&self) -> &Ledger {
        &self.device
    }

    /// Host RAM's ledger.
    pub fn host(&self) -> &Ledger {
        &self.host
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
}

/// The committed claims of one allocation across the stores a device
/// reaches ([`DeviceLedgers::claim`]): at most one per store. Kept for the
/// life of the allocation; dropped, its bytes are free.
#[must_use]
pub struct Held {
    claims: Vec<Claim>,
}

impl Held {
    /// Every claimed byte allocated: the allocation was made.
    pub fn allocated(&mut self) {
        for claim in &mut self.claims {
            let left = claim.bytes() - claim.allocated_bytes();
            claim.allocated(left);
        }
    }

    /// The claims, to keep for the life of the allocation.
    pub fn into_claims(self) -> Vec<Claim> {
        self.claims
    }

    /// The bytes claimed in `store`.
    pub fn bytes_in(&self, store: &seismic::StoreIdentity) -> u64 {
        self.claims.iter().filter(|claim| claim.ledger.store.identity == *store).map(Claim::bytes).sum()
    }

    /// Split off `bytes` claimed in `store`, before any is allocated, as a
    /// [`CommittedKind::Backing`] claim: memory the allocation backs after
    /// the rest is made (a host tier backed to its load-time slots after
    /// its import), which [`Recorded::record_backing`] keeps apart.
    ///
    /// # Panics
    /// Where it claimed fewer unallocated bytes there.
    pub fn split_backing(&mut self, store: &seismic::StoreIdentity, bytes: u64) -> Claim {
        self.claims
            .iter_mut()
            .find(|claim| claim.ledger.store.identity == *store)
            .unwrap_or_else(|| panic!("invariant violated: a backing split from an allocation claiming nothing in {store:?}"))
            .split_backing(bytes)
    }
}

/// What a load allocated as it started: one [`CommittedKind::Load`] claim
/// per store its device reaches, allocated whole. The holdings that name
/// those bytes carve them out ([`Self::carve`]), so no allocation is
/// admitted twice; the rest is held for the domain's life.
pub struct Recorded {
    claims: Vec<Claim>,
    /// The load's minimum request state, in each store that holds state
    /// ([`Self::record_state_floor`]): held for the domain's life.
    state_floors: Vec<Claim>,
    /// Allocations the load began that back after it started, each store's
    /// claim to them allocated as they back ([`Self::record_backing`]).
    backings: Vec<Claim>,
}

impl Recorded {
    /// Nothing recorded yet, in each store `ledgers` keep.
    pub fn new(ledgers: &DeviceLedgers) -> Self {
        let mut claims: Vec<Claim> = Vec::new();
        for ledger in [&ledgers.device, &ledgers.host] {
            if claims.iter().any(|claim| Arc::ptr_eq(&claim.ledger.store, &ledger.store)) {
                continue;
            }
            let id = ledger.state().insert(Holding::Committed(CommittedKind::Load), 0);
            claims.push(Claim { ledger: ledger.clone(), id, kind: CommittedKind::Load });
        }
        Self { claims, state_floors: Vec::new(), backings: Vec::new() }
    }

    /// `floor`, the load's minimum request state claimed as
    /// [`CommittedKind::StateFloor`], kept for the domain's life.
    pub fn record_state_floor(&mut self, floor: Held) {
        for claim in floor.claims {
            assert_eq!(claim.kind, CommittedKind::StateFloor, "invariant violated: a state floor recorded of another kind");
            self.state_floors.push(claim);
        }
    }

    /// `backing`, a [`CommittedKind::Backing`] claim split from an
    /// allocation the load made as it started ([`Held::split_backing`]),
    /// kept until its holder takes it ([`Self::backing`]): one per store.
    pub fn record_backing(&mut self, backing: Claim) {
        assert_eq!(backing.kind, CommittedKind::Backing, "invariant violated: a backing recorded of another kind");
        let store = &backing.ledger.store.identity;
        assert!(self.backings.iter().all(|held| held.ledger.store.identity != *store), "invariant violated: two backings recorded in {store:?}");
        self.backings.push(backing);
    }

    /// The backing recorded in `store`, to its holder.
    pub fn backing(&mut self, store: &seismic::StoreIdentity) -> Option<Claim> {
        let at = self.backings.iter().position(|backing| backing.ledger.store.identity == *store)?;
        Some(self.backings.remove(at))
    }

    /// The record of `store`.
    fn of(&mut self, store: &seismic::StoreIdentity) -> &mut Claim {
        self.claims
            .iter_mut()
            .find(|claim| claim.ledger.store.identity == *store)
            .unwrap_or_else(|| panic!("invariant violated: a load records nothing in {store:?}"))
    }

    /// `held`, an allocation the load made as it started, joins the record.
    ///
    /// # Panics
    /// Where it is not a load's allocation, allocated whole.
    pub fn record(&mut self, held: Held) {
        for claim in held.claims {
            assert_eq!(claim.kind, CommittedKind::Load, "invariant violated: a load records only its own allocations");
            assert_eq!(claim.allocated_bytes(), claim.bytes(), "invariant violated: a load records an allocation it made");
            let into = self.of(&claim.ledger.store.identity).id;
            let (ledger, id) = claim.into_parts();
            ledger.state().merge(id, into);
        }
    }

    /// `bytes` of `store`'s record as a claim of `kind`, allocated whole:
    /// a holding of memory the load allocated as it started, never
    /// admitted again.
    ///
    /// # Panics
    /// Where the record lacks the bytes.
    pub fn carve(&mut self, store: &seismic::StoreIdentity, kind: CommittedKind, bytes: u64) -> Claim {
        let record = self.of(store);
        let ledger = record.ledger.clone();
        let mut state = ledger.state();
        let has = state.entry(record.id).bytes;
        let rest = has
            .checked_sub(bytes)
            .unwrap_or_else(|| panic!("invariant violated: a holding carves {bytes} bytes of a load's {has} recorded in {store:?}"));
        state.resize(record.id, rest);
        let id = state.insert(Holding::Committed(kind), bytes);
        state.allocate(id, bytes);
        drop(state);
        Claim { ledger, id, kind }
    }

    /// The arena's claim: the `allocated` bytes of its graph scratch and
    /// bound constants, carved from the device's record. A group's class
    /// beyond one place is claimed as the group opens, never here.
    pub fn claim_arena(&mut self, ledger: &Ledger, allocated: u64) -> ArenaClaim {
        ArenaClaim { claim: self.carve(&ledger.store(), CommittedKind::Arena, allocated) }
    }

    /// The record's claims, held for the domain's life.
    ///
    /// # Panics
    /// Where a backing recorded was never taken by its holder.
    pub fn into_claims(self) -> Vec<Claim> {
        assert!(self.backings.is_empty(), "invariant violated: a backing recorded that no holding took");
        self.claims.into_iter().chain(self.state_floors).collect()
    }

}

/// The store `device` places `contract`'s fixed storage in.
fn placed_store(device: &seismic::Device, contract: seismic::Contract) -> seismic::StoreIdentity {
    let topology = device.topology();
    device
        .info()
        .paths(seismic::Placing::Fixed(contract))
        .first()
        .and_then(|path| topology.access_path(*path))
        .and_then(|path| topology.store_identity(path.store()))
        .expect("an opened device places device, upload and shared storage")
}

/// Why a claim is refused outright, of any kind: a load's allocation
/// ([`DeviceLedgers::claim`]) or a need ([`Ledger::claim`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Short {
    /// `store`'s last reading failed (`reason`): no figure of it admits
    /// any bytes there, and none is short of them.
    Blind { store: seismic::StoreIdentity, reason: String },
    /// `store` is `over` bytes short of the `required` the allocation adds
    /// there.
    Over { store: seismic::StoreIdentity, required: u64, over: u64 },
}

/// One store's record.
struct Store {
    identity: seismic::StoreIdentity,
    state: Mutex<State>,
}

/// Which entry of a store.
type EntryId = u64;

/// A store's entries and its latest reading. The totals are the entries'
/// sums, kept with them.
#[derive(Default)]
struct State {
    held: BTreeMap<EntryId, Entry>,
    next: EntryId,
    committed: u64,
    cache: u64,
    /// What entries' holders allocated of them, and what cache that
    /// yielded still releases: the process's memory a reading shows.
    allocated: u64,
    /// What cache yielded to claims in process and its holders have not
    /// freed yet ([`Entry::releasing`]).
    releasing: u64,
    /// Cache by kind ([`CacheKind::ALL`]'s order).
    cache_of: [u64; CacheKind::ALL.len()],
    /// Each sequence's committed state, by admission.
    state_of: BTreeMap<RequestId, u64>,
    /// What the sequences' state holds in all: the sum of `state_of`.
    state_total: u64,
    /// What [`CommittedKind::StateFloor`] claims hold: committed state up to
    /// it is held once, by the floor ([`Self::raise`]).
    floor: u64,
    /// Sequences' state entries a preemption reserved, and the entry of the
    /// preemption each one's bytes go to as its sequence is evicted.
    reserved: HashMap<EntryId, EntryId>,
    /// The entries of preemptions not yet completed: never preempted
    /// themselves.
    pending: std::collections::HashSet<EntryId>,
    /// Which cache yields first to a claim.
    order: Vec<CacheKind>,
    reading: Option<Result<Available, Blind>>,
    others: Option<u64>,
    others_before: Option<u64>,
    /// What the grant lets the process hold, as of the reading: what it
    /// allocated then and what the grant left.
    grant: Option<u64>,
}

impl State {
    /// Refused as `store`'s reading failed where `bytes` more are asked of
    /// it: no figure of a blind store admits any bytes there, and none is
    /// short of them. The one place a refusal is told blind.
    fn seen(&self, store: &seismic::StoreIdentity, bytes: u64) -> Result<(), Short> {
        match &self.reading {
            Some(Ok(_)) => Ok(()),
            _ if bytes == 0 => Ok(()),
            Some(Err(Blind(reason))) => Err(Short::Blind { store: store.clone(), reason: reason.clone() }),
            None => Err(Short::Blind { store: store.clone(), reason: "not read yet".into() }),
        }
    }

    /// [`Self::take`] in `store`, refused as [`Short`]: blind where the
    /// store's reading failed ([`Self::seen`]).
    fn take_in(&mut self, store: &seismic::StoreIdentity, kind: CommittedKind, bytes: u64) -> Result<Taken, Short> {
        self.seen(store, bytes)?;
        self.take(kind, bytes).map_err(|over| over.of(store, bytes))
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            committed: self.committed - self.floor.min(self.state_total),
            preemptible: self.state_of.values().skip(1).sum(),
            cache: self.cache,
            allocated: self.allocated,
            releasing: self.releasing,
            cache_of: self.cache_of,
            reading: self.reading.clone().unwrap_or_else(|| Err(Blind("not read yet".into()))),
            others: self.others,
            others_before: self.others_before,
            grant: self.grant,
        }
    }

    /// Add `bytes` to the totals of `holding`, or take them away.
    fn count(&mut self, holding: Holding, bytes: u64, add: bool) {
        let apply = |total: &mut u64| {
            *total = if add {
                total.checked_add(bytes).expect("a store's bytes fit u64")
            } else {
                total.checked_sub(bytes).expect("a total holds its entries")
            }
        };
        match holding {
            Holding::Committed(kind) => {
                apply(&mut self.committed);
                if kind == CommittedKind::StateFloor {
                    apply(&mut self.floor);
                }
                if let CommittedKind::State(sequence) = kind {
                    apply(&mut self.state_total);
                    let held = self.state_of.entry(sequence).or_default();
                    apply(held);
                    if *held == 0 {
                        self.state_of.remove(&sequence);
                    }
                }
            }
            Holding::Cache(kind) => {
                apply(&mut self.cache);
                apply(&mut self.cache_of[kind.index()]);
            }
        }
    }

    /// A new entry of `bytes`, none of them allocated yet.
    fn insert(&mut self, holding: Holding, bytes: u64) -> EntryId {
        let id = self.next;
        self.next += 1;
        self.held.insert(id, Entry { holding, bytes, allocated: 0, releasing: 0 });
        self.count(holding, bytes, true);
        id
    }

    /// Entry `id`'s holder has `allocated` of its bytes allocated now, the
    /// store's total with it.
    fn allocate(&mut self, id: EntryId, allocated: u64) {
        let entry = *self.entry(id);
        assert!(allocated <= entry.bytes, "invariant violated: an entry's holder allocates {allocated} of its {} bytes", entry.bytes);
        self.allocated = (self.allocated - entry.allocated).checked_add(allocated).expect("a store's bytes fit u64");
        self.entry(id).allocated = allocated;
    }

    fn entry(&mut self, id: EntryId) -> &mut Entry {
        self.held.get_mut(&id).expect("a holder's entry is in its ledger")
    }

    /// Cache entry `id` yields `bytes` to a claim in process: held no
    /// more, they stay allocated, releasing, until its holder frees them
    /// ([`Cached::released`]). So the reading counts them the process's,
    /// and nothing allocates them twice: the claim allocates its bytes only
    /// as the holder frees these ([`Snapshot::allocatable`]).
    fn yields(&mut self, id: EntryId, bytes: u64) {
        let entry = self.entry(id);
        assert!(matches!(entry.holding, Holding::Cache(_)), "only cache yields to a claim");
        entry.bytes -= bytes;
        entry.allocated -= bytes;
        entry.releasing += bytes;
        let holding = entry.holding;
        self.count(holding, bytes, false);
        self.releasing += bytes;
    }

    /// Entry `id`'s holder freed `bytes` of what it yielded.
    fn released(&mut self, id: EntryId, bytes: u64) {
        let entry = self.entry(id);
        let left = entry.releasing.checked_sub(bytes);
        let has = entry.releasing;
        entry.releasing = left.unwrap_or_else(|| panic!("invariant violated: a holding frees {bytes} of the {has} bytes it yielded"));
        self.releasing -= bytes;
        self.allocated -= bytes;
    }

    /// Resize entry `id` to `bytes`, its totals with it, what its holder
    /// allocated past them freed.
    fn resize(&mut self, id: EntryId, bytes: u64) {
        let entry = *self.entry(id);
        if entry.allocated > bytes {
            self.allocate(id, bytes);
        }
        self.count(entry.holding, entry.bytes, false);
        self.count(entry.holding, bytes, true);
        self.entry(id).bytes = bytes;
    }

    /// Entry `from`'s bytes, and what of them is allocated, join committed
    /// entry `into` of the same kind.
    fn merge(&mut self, from: EntryId, into: EntryId) {
        let from = self.held.remove(&from).expect("a taken entry is in its ledger");
        let into = self.entry(into);
        into.bytes += from.bytes;
        into.allocated += from.allocated;
    }

    /// Entry `id` turns `holding`, its bytes kept.
    fn turn(&mut self, id: EntryId, holding: Holding) {
        let entry = *self.entry(id);
        self.count(entry.holding, entry.bytes, false);
        self.count(holding, entry.bytes, true);
        self.entry(id).holding = holding;
    }

    /// Entry `id` was let go: a preempted sequence's bytes go to the
    /// preemption that reserved them; any other's are free.
    fn release(&mut self, id: EntryId) {
        let entry = self.held.remove(&id).expect("a holder's entry is in its ledger");
        self.allocated -= entry.allocated + entry.releasing;
        self.releasing -= entry.releasing;
        match self.reserved.remove(&id) {
            // The preempted sequence's bytes become the preemption's.
            Some(to) => {
                self.count(entry.holding, entry.bytes, false);
                let holding = self.entry(to).holding;
                self.count(holding, entry.bytes, true);
                self.entry(to).bytes += entry.bytes;
            }
            None => self.count(entry.holding, entry.bytes, false),
        }
    }

    /// The state entries a claim of `kind` may take, the latest admitted
    /// sequence's first: a sequence's growth preempts every sequence
    /// admitted after it, and nothing else preempts.
    fn preemptible(&self, kind: CommittedKind) -> Vec<EntryId> {
        let CommittedKind::State(claimant) = kind else {
            return Vec::new();
        };
        let mut victims: Vec<(RequestId, EntryId)> = self
            .held
            .iter()
            .filter_map(|(id, entry)| match entry.holding {
                Holding::Committed(CommittedKind::State(sequence))
                    if sequence > claimant
                        && entry.bytes > 0
                        && !self.reserved.contains_key(id)
                        && !self.pending.contains(id) =>
                {
                    Some((sequence, *id))
                }
                _ => None,
            })
            .collect();
        victims.sort_by(|a, b| b.cmp(a));
        victims.into_iter().map(|(_, id)| id).collect()
    }

    /// Whether `bytes` more committed memory of `kind` fits the budget
    /// beside every committed holding it may not preempt: whether free
    /// room, cache and preemptible state cover it.
    fn admit(&self, kind: CommittedKind, bytes: u64) -> Admit {
        let takable: u64 = self.preemptible(kind).iter().map(|id| self.held[id].bytes).sum();
        self.covers(self.raise(kind, bytes), takable, kind.reclaim())
    }

    /// What `bytes` more committed memory of `kind` raise what is held: a
    /// sequence's state is held first by the state floor, which is already
    /// held, so only what it adds beyond the floor's unused bytes raises it.
    fn raise(&self, kind: CommittedKind, bytes: u64) -> u64 {
        match kind {
            CommittedKind::State(_) => bytes - bytes.min(self.floor.saturating_sub(self.state_total)),
            _ => bytes,
        }
    }

    /// Whether free room under `reclaim`, the cache and `takable` committed
    /// bytes cover `bytes`.
    fn covers(&self, bytes: u64, takable: u64, reclaim: seismic::Reclaim) -> Admit {
        let covered = self.snapshot().free_for(reclaim) + self.cache + takable;
        match bytes.checked_sub(covered) {
            Some(over) if over > 0 => Admit::Refuse(Over { bytes: over }),
            _ => Admit::Admit,
        }
    }

    /// Hold `bytes` more committed memory of `kind` in a new committed
    /// entry, in the one order the holding kinds give: free room, then
    /// cache in the yield order, then the state of sequences admitted after
    /// the claimant's, the latest first and just enough. Where it needs
    /// state, the entry holds what free room and cache gave and the
    /// sequences' entries are reserved to it: their bytes join it as they
    /// are evicted, and no other claim can take them.
    fn take(&mut self, kind: CommittedKind, bytes: u64) -> Result<Taken, Over> {
        if let Admit::Refuse(over) = self.admit(kind, bytes) {
            return Err(over);
        }
        let free = self.snapshot().free_for(kind.reclaim());
        let mut owed = self.raise(kind, bytes).saturating_sub(free);
        let mut short = owed.saturating_sub(self.cache);
        let mut victims = Vec::new();
        for id in self.preemptible(kind) {
            if short == 0 {
                break;
            }
            short = short.saturating_sub(self.held[&id].bytes);
            victims.push(id);
        }
        assert_eq!(short, 0, "admission counts preemptible state only where it covers the rest");
        for cache in self.yield_order() {
            let ids: Vec<EntryId> = self
                .held
                .iter()
                .filter(|(_, entry)| entry.holding == Holding::Cache(cache))
                .map(|(id, _)| *id)
                .collect();
            for id in ids {
                let yields = self.held[&id].bytes.min(owed);
                self.yields(id, yields);
                owed -= yields;
            }
        }
        let entry = self.insert(Holding::Committed(kind), bytes - owed);
        if victims.is_empty() {
            return Ok(Taken::Held(entry));
        }
        let sequences = victims
            .iter()
            .map(|id| match self.held[id].holding {
                Holding::Committed(CommittedKind::State(sequence)) => sequence,
                _ => unreachable!("only state is preemptible"),
            })
            .collect();
        for victim in victims {
            self.reserved.insert(victim, entry);
        }
        self.pending.insert(entry);
        Ok(Taken::Preempting { entry, sequences })
    }

    /// The preemption whose entry is `entry` ends with `bytes` held for its
    /// need: every sequence it reserved was evicted. What their eviction
    /// freed beyond the need turns spare.
    fn completed(&mut self, entry: EntryId, bytes: u64) {
        assert!(
            !self.reserved.values().any(|to| *to == entry),
            "a preemption completes only once every sequence it named is evicted"
        );
        self.pending.remove(&entry);
        let has = self.held[&entry].bytes;
        let spare = has.checked_sub(bytes).expect("an evicted sequence frees what it held");
        self.resize(entry, bytes);
        if spare > 0 {
            let id = self.insert(Holding::Cache(CacheKind::Spare), spare);
            self.allocate(id, spare);
        }
    }

    /// The preemption whose entry is `entry` was abandoned: its sequences
    /// stand, and what it held is let go.
    fn abandoned(&mut self, entry: EntryId) {
        self.reserved.retain(|_, to| *to != entry);
        self.pending.remove(&entry);
        self.release(entry);
    }

    /// Hold up to `most` whole `unit`s of committed memory of `kind`: as
    /// many as free room, then spare granules, then idle graph slots, then
    /// bank slots above the floor cover now, that cache yielding as
    /// [`Self::take`] yields it.
    /// Never preempts and never refuses. The entry and its units.
    fn take_units(&mut self, kind: CommittedKind, unit: NonZeroU64, most: u64) -> (EntryId, u64) {
        assert!(!matches!(kind, CommittedKind::State(_)), "a sequence's state is claimed by its rows, never in units");
        let yielding = [CacheKind::Spare, CacheKind::GraphSlots, CacheKind::Bank];
        let snapshot = self.snapshot();
        let cover = yielding.iter().fold(snapshot.free(), |cover, kind| cover.saturating_add(snapshot.cache_of(*kind)));
        let units = most.min(cover / unit.get());
        let bytes = units * unit.get();
        let mut owed = bytes.saturating_sub(snapshot.free());
        for cache in yielding {
            let ids: Vec<EntryId> = self
                .held
                .iter()
                .filter(|(_, entry)| entry.holding == Holding::Cache(cache))
                .map(|(id, _)| *id)
                .collect();
            for id in ids {
                let yields = self.held[&id].bytes.min(owed);
                self.yields(id, yields);
                owed -= yields;
            }
        }
        assert_eq!(owed, 0, "free room, spare and bank cache cover the units counted from them");
        (self.insert(Holding::Committed(kind), bytes), units)
    }

    /// The ranked order, spare first, then the kinds it leaves out, idle
    /// graph slots just before the bank.
    fn yield_order(&self) -> Vec<CacheKind> {
        let mut order = vec![CacheKind::Spare];
        order.extend(self.order.iter().copied().filter(|kind| !matches!(kind, CacheKind::Spare | CacheKind::GraphSlots)));
        for kind in CacheKind::ALL {
            if kind != CacheKind::GraphSlots && !order.contains(&kind) {
                order.push(kind);
            }
        }
        let bank = order.iter().position(|kind| *kind == CacheKind::Bank).expect("every kind yields");
        order.insert(bank, CacheKind::GraphSlots);
        order
    }
}

impl CacheKind {
    /// In declaration order, which [`Self::index`] counts.
    const ALL: [Self; 6] = [Self::GraphSlots, Self::Bank, Self::Spare, Self::Prefix, Self::Drafter, Self::Vision];

    const fn index(self) -> usize {
        self as usize
    }
}

/// What a take came to.
enum Taken {
    /// The entry holds the whole need.
    Held(EntryId),
    /// The entry holds what free room and cache gave; these sequences'
    /// eviction gives the rest.
    Preempting { entry: EntryId, sequences: Vec<RequestId> },
}

/// One load's view of one store's ledger: the store's one record, and the
/// load's shape its needs are sized from ([`size`]). Clones share both.
#[derive(Clone)]
pub struct Ledger {
    store: Arc<Store>,
    shape: Arc<LoadShape>,
}

/// A committed claim: the right to allocate its bytes, held until dropped.
/// Dropping it frees its bytes; its holder drops it once no GPU fence reads
/// them. Not `Clone`: one holder.
#[must_use]
pub struct Claim {
    ledger: Ledger,
    id: EntryId,
    kind: CommittedKind,
}

/// The arena's claim, carved once at load from what the load recorded
/// allocating and living for the load: its graph scratch, the held pass's
/// buffers and the floor's group of one place, the bytes of the load's own
/// graph placement. A group's class beyond one place is never the arena's:
/// it is claimed as the group opens and freed as it closes
/// ([`CommittedKind::GroupClass`]).
#[must_use]
pub struct ArenaClaim {
    claim: Claim,
}

impl ArenaClaim {
    pub fn bytes(&self) -> u64 {
        self.claim.bytes()
    }
}

/// A cache holding: the right to allocate its bytes, held until dropped.
/// Dropping it frees its bytes, as a claim's, and what it yielded. A claim
/// may take its bytes in process ([`Ledger::claim`]): its holder reads what
/// it still holds from [`Self::bytes`], evicts the rest and reports the
/// memory freed ([`Self::released`]), which the claim allocates only then.
#[must_use]
pub struct Cached {
    ledger: Ledger,
    id: EntryId,
    kind: CacheKind,
}

/// Why a claim was not taken whole.
#[derive(Debug)]
pub enum Refused {
    /// Its store is blind, or committed memory it may not preempt, with the
    /// need, exceeds the budget. Nothing was taken.
    Short(Short),
    /// Free room and cache cannot cover it: the preemption holds what they
    /// gave, and the bytes of the sequences it names are reserved to it.
    Preempt(Preemption),
}

/// A claim waiting on preemption: what free room and cache gave its need,
/// and the sequences, the latest admitted first and just enough, whose
/// state is reserved to it. Their holders evict them (fenced) and drop
/// their claims; each one's bytes join the preemption as it drops, and no
/// other claim can take them. Then [`Ledger::complete`] (or
/// [`Claim::complete`] for a growth) holds the need, which cannot fail.
/// Dropped before that, it lets go of what it holds and the sequences
/// stand.
#[must_use]
pub struct Preemption {
    ledger: Ledger,
    entry: EntryId,
    kind: CommittedKind,
    bytes: u64,
    sequences: Vec<RequestId>,
}

impl Preemption {
    /// The sequences to evict, the latest admitted first.
    pub fn sequences(&self) -> &[RequestId] {
        &self.sequences
    }

    /// The refusal it comes to where its holder cannot evict the sequences
    /// it names: its need, over by what their state would give.
    pub fn short(&self) -> Short {
        let held = self.ledger.state().held[&self.entry].bytes;
        Over { bytes: self.bytes - held }.of(&self.ledger.store.identity, self.bytes)
    }

    /// Its parts, without letting go of what it holds.
    fn into_parts(self) -> (Ledger, EntryId, CommittedKind, u64) {
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so its ledger moves out once.
        let ledger = unsafe { std::ptr::read(&this.ledger) };
        // SAFETY: as the ledger; the sequences' allocation is let go here.
        drop(unsafe { std::ptr::read(&this.sequences) });
        (ledger, this.entry, this.kind, this.bytes)
    }
}

impl std::fmt::Debug for Preemption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Preemption").field("kind", &self.kind).field("bytes", &self.bytes).field("sequences", &self.sequences).finish()
    }
}

impl Drop for Preemption {
    /// Nothing is recorded while unwinding ([`Claim`]'s drop).
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        self.ledger.state().abandoned(self.entry);
    }
}

/// A need refused: committed memory it may not preempt, with the need,
/// exceeds the budget by `bytes`. Cache never refuses a claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Over {
    pub bytes: u64,
}

impl Over {
    /// The refusal of `required` bytes in `store` this many short.
    fn of(self, store: &seismic::StoreIdentity, required: u64) -> Short {
        Short::Over { store: store.clone(), required, over: self.bytes }
    }
}

/// Admission by the budget's figures alone: the check inside every take,
/// and the arena's ([`Ledger::admit_arena`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    /// The need fits beside every committed claim (cache yields to it).
    Admit,
    Refuse(Over),
}

/// The ledger and its reading at one instant: every figure one decision
/// reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    committed: u64,
    /// Committed state of every sequence but the earliest admitted: what
    /// preemption can still take.
    preemptible: u64,
    cache: u64,
    /// What entries' holders allocated of them, and cache still releasing.
    allocated: u64,
    /// What cache yielded to claims and its holders have not freed yet.
    releasing: u64,
    cache_of: [u64; CacheKind::ALL.len()],
    reading: Result<Available, Blind>,
    /// Others' memory at the reading, beside what the process allocated
    /// then: a fact of that instant, which no claim or release since moves.
    others: Option<u64>,
    /// Others' memory at the reading before, for the guarantee.
    others_before: Option<u64>,
    /// What the grant lets the process hold, as of the reading.
    grant: Option<u64>,
}

impl Ledger {
    fn state(&self) -> MutexGuard<'_, State> {
        self.store.state.lock().expect("ledger lock poisoned")
    }

    pub fn store(&self) -> seismic::StoreIdentity {
        self.store.identity.clone()
    }

    /// Take the store's reading with `read` (its OS's [`Available`]
    /// function) under the ledger's lock, so no claim or free falls between
    /// the OS's figure and the entries it is paired with. A reading older
    /// than the latest is ignored, so readings never mix.
    pub fn read(&self, read: impl FnOnce() -> Result<Available, Blind>) {
        let mut state = self.state();
        let reading = read();
        if let (Ok(now), Some(Ok(latest))) = (&reading, &state.reading) {
            if now.at < latest.at {
                return;
            }
        }
        // What the process allocated is its memory of the reading; what it
        // holds beyond that is not in the reading yet, so never others'.
        let allocated = state.allocated;
        let others = reading.as_ref().ok().map(|reading| (reading.total - reading.available).saturating_sub(allocated));
        state.grant = reading.as_ref().ok().and_then(|reading| reading.grant_left).map(|left| allocated.saturating_add(left));
        state.reading = Some(reading);
        state.others_before = std::mem::replace(&mut state.others, others);
    }

    /// The ledger and its latest reading, together.
    pub fn snapshot(&self) -> Snapshot {
        self.state().snapshot()
    }

    /// Every entry, for telemetry.
    pub fn entries(&self) -> Vec<Entry> {
        self.state().held.values().copied().collect()
    }

    /// Cache yields to a claim in `order`, the measured value per byte of
    /// each kind, least first; kinds it leaves out follow.
    pub fn rank(&self, order: Vec<CacheKind>) {
        self.state().order = order;
    }

    /// The bytes `need` takes, by the formulas a claim of it is sized by.
    pub fn bytes(&self, need: &Need) -> u64 {
        size(&self.shape, need)
    }

    /// Whether `need` could be held beside every committed claim: refused
    /// as [`Self::claim`] refuses it.
    pub fn admit(&self, need: &Need) -> Result<(), Short> {
        let bytes = size(&self.shape, need);
        let state = self.state();
        state.seen(&self.store.identity, bytes)?;
        match state.admit(need.kind(), bytes) {
            Admit::Admit => Ok(()),
            Admit::Refuse(over) => Err(over.of(&self.store.identity, bytes)),
        }
    }

    /// Whether `bytes` more, preempting no sequence, could be held beside
    /// every committed claim counting `reclaim` as room: refused as a claim
    /// of them is.
    pub fn covers(&self, bytes: u64, reclaim: seismic::Reclaim) -> Result<(), Short> {
        let state = self.state();
        state.seen(&self.store.identity, bytes)?;
        match state.covers(bytes, 0, reclaim) {
            Admit::Admit => Ok(()),
            Admit::Refuse(over) => Err(over.of(&self.store.identity, bytes)),
        }
    }

    /// The one way committed memory is taken: `need` sized by the
    /// formulas, synchronously, in the one order the holding kinds give:
    /// free room, then cache in the yield order (spare granules at once,
    /// bank slots within two decode steps or one prompt chunk, which the
    /// guarantee covers). Each cache holder reads what it still holds from
    /// its handle's `bytes`. Where these cannot cover it, the state of
    /// sequences admitted after the claimant's covers the rest
    /// ([`Refused::Preempt`]); refused outright only where committed memory
    /// it may not preempt, with the need, exceeds the budget, or where the
    /// store is blind.
    pub fn claim(&self, need: Need) -> Result<Claim, Refused> {
        let bytes = size(&self.shape, &need);
        let kind = need.kind();
        let taken = self.state().take_in(&self.store.identity, kind, bytes).map_err(Refused::Short)?;
        match taken {
            Taken::Held(id) => Ok(Claim { ledger: self.clone(), id, kind }),
            Taken::Preempting { entry, sequences } => {
                Err(Refused::Preempt(Preemption { ledger: self.clone(), entry, kind, bytes, sequences }))
            }
        }
    }

    /// Committed memory of `kind` in whole `unit`s, at most `most` of them:
    /// as many as free room, then spare granules, then idle graph slots,
    /// then bank slots above the
    /// floor cover now, yielding that cache as [`Self::claim`] does. It
    /// never preempts a sequence and is never refused: it may hold none
    /// (0 bytes). Its holder reads the units it got from the second value.
    ///
    /// # Panics
    /// For a sequence's state, which is claimed by its rows alone.
    pub fn claim_units(&self, kind: CommittedKind, unit: NonZeroU64, most: u64) -> (Claim, u64) {
        let (id, units) = self.state().take_units(kind, unit, most);
        (Claim { ledger: self.clone(), id, kind }, units)
    }

    /// The claim `preemption` waited for, every sequence it named evicted.
    /// A sequence not yet evicted, or a preemption for a growth, is a
    /// broken invariant.
    pub fn complete(&self, preemption: Preemption) -> Claim {
        let (ledger, entry, kind, bytes) = preemption.into_parts();
        assert!(Arc::ptr_eq(&ledger.store, &self.store), "a preemption completes in its own ledger");
        ledger.state().completed(entry, bytes);
        Claim { ledger, id: entry, kind }
    }

    /// Hold `bytes` of free room as cache of `kind`; refused where free
    /// room is short of them or the store is blind. Free room is the budget
    /// less everything held; the bank's target subtracts the guarantee
    /// before it asks.
    pub fn cache(&self, kind: CacheKind, bytes: u64) -> Result<Cached, Short> {
        let mut state = self.state();
        state.seen(&self.store.identity, bytes)?;
        let free = state.snapshot().free();
        if free < bytes {
            return Err(Over { bytes: bytes - free }.of(&self.store.identity, bytes));
        }
        let id = state.insert(Holding::Cache(kind), bytes);
        state.allocate(id, bytes);
        Ok(Cached { ledger: self.clone(), id, kind })
    }

    /// Hold `bytes` of free room as a [`CommittedKind::Backing`] claim, none
    /// of it allocated: an allocation backed after it begins (a bank's
    /// growth), allocated as its memory is backed ([`Claim::backed`]) and
    /// cache once it ends ([`Claim::into_backed_cache`]). Refused as
    /// [`Self::cache`] refuses.
    pub fn backing(&self, bytes: u64) -> Result<Claim, Short> {
        let mut state = self.state();
        state.seen(&self.store.identity, bytes)?;
        let free = state.snapshot().free();
        if free < bytes {
            return Err(Over { bytes: bytes - free }.of(&self.store.identity, bytes));
        }
        let id = state.insert(Holding::Committed(CommittedKind::Backing), bytes);
        drop(state);
        Ok(Claim { ledger: self.clone(), id, kind: CommittedKind::Backing })
    }
}

impl Claim {
    pub fn kind(&self) -> CommittedKind {
        self.kind
    }

    pub fn bytes(&self) -> u64 {
        self.ledger.state().entry(self.id).bytes
    }

    /// Its holder made `bytes` more of it.
    ///
    /// # Panics
    /// Past [`Self::bytes`].
    pub fn allocated(&mut self, bytes: u64) {
        let mut state = self.ledger.state();
        let has = state.entry(self.id).allocated;
        state.allocate(self.id, has.checked_add(bytes).expect("a claim's bytes fit u64"));
    }

    /// Its holder freed `bytes` of what it allocated, keeping the claim.
    ///
    /// # Panics
    /// Past what it allocated.
    pub fn deallocated(&mut self, bytes: u64) {
        let mut state = self.ledger.state();
        let has = state.entry(self.id).allocated;
        let left = has.checked_sub(bytes).unwrap_or_else(|| panic!("invariant violated: a claim frees {bytes} of its {has} allocated bytes"));
        state.allocate(self.id, left);
    }

    /// Its holder freed `bytes` it allocated and holds no claim to them
    /// any more: its bytes and what it allocated both fall by them.
    ///
    /// # Panics
    /// Past what it allocated.
    pub fn shrink(&mut self, bytes: u64) {
        let mut state = self.ledger.state();
        let entry = *state.entry(self.id);
        let allocated = entry
            .allocated
            .checked_sub(bytes)
            .unwrap_or_else(|| panic!("invariant violated: a claim shrinks by {bytes} of its {} allocated bytes", entry.allocated));
        state.allocate(self.id, allocated);
        state.resize(self.id, entry.bytes - bytes);
    }

    /// What its holder allocated of it.
    pub fn allocated_bytes(&self) -> u64 {
        self.ledger.state().entry(self.id).allocated
    }

    /// `backed` of its bytes exist now: what its holder allocated is that,
    /// at most its bytes (Seismic's charge of a backing may run past the
    /// plan's figure it was claimed by).
    pub fn backed(&mut self, backed: u64) {
        let mut state = self.ledger.state();
        let bytes = state.entry(self.id).bytes;
        state.allocate(self.id, backed.min(bytes));
    }

    /// The backing ended at `backed` bytes: what it did not back is free,
    /// and the rest is cache of `kind` ([`Self::into_cache`]).
    pub fn into_backed_cache(mut self, backed: u64, kind: CacheKind) -> Cached {
        self.backed(backed);
        let allocated = self.allocated_bytes();
        self.ledger.state().resize(self.id, allocated);
        self.into_cache(kind)
    }

    /// Split off `bytes` of it, none allocated, as a
    /// [`CommittedKind::Backing`] claim of their own: memory it covers that
    /// backs after the rest is made (a load's host tier, after its import).
    ///
    /// # Panics
    /// Past what its holder has not allocated.
    fn split_backing(&mut self, bytes: u64) -> Claim {
        let mut state = self.ledger.state();
        let entry = *state.entry(self.id);
        let unallocated = entry.bytes - entry.allocated;
        assert!(bytes <= unallocated, "invariant violated: a claim splits off {bytes} bytes of its {unallocated} unallocated");
        state.resize(self.id, entry.bytes - bytes);
        let id = state.insert(Holding::Committed(CommittedKind::Backing), bytes);
        drop(state);
        Claim { ledger: self.ledger.clone(), id, kind: CommittedKind::Backing }
    }

    /// Its ledger and entry, without letting go of what it holds.
    fn into_parts(self) -> (Ledger, EntryId) {
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped, so its ledger moves out once.
        let ledger = unsafe { std::ptr::read(&this.ledger) };
        (ledger, this.id)
    }

    /// Grow by `need`, of this claim's kind (a sequence's rows), as
    /// [`Ledger::claim`] takes.
    pub fn grow(&mut self, need: Need) -> Result<(), Refused> {
        assert_eq!(need.kind(), self.kind, "a claim grows by a need of its own kind");
        let bytes = size(&self.ledger.shape, &need);
        let mut state = self.ledger.state();
        match state.take_in(&self.ledger.store.identity, self.kind, bytes).map_err(Refused::Short)? {
            Taken::Held(entry) => {
                state.merge(entry, self.id);
                Ok(())
            }
            Taken::Preempting { entry, sequences } => {
                drop(state);
                Err(Refused::Preempt(Preemption { ledger: self.ledger.clone(), entry, kind: self.kind, bytes, sequences }))
            }
        }
    }

    /// Grow by the need `preemption` waited for, every sequence it named
    /// evicted ([`Ledger::complete`]).
    pub fn complete(&mut self, preemption: Preemption) {
        let (ledger, entry, kind, bytes) = preemption.into_parts();
        assert!(Arc::ptr_eq(&ledger.store, &self.ledger.store) && kind == self.kind, "a growth completes into its own claim");
        let mut state = ledger.state();
        state.completed(entry, bytes);
        state.merge(entry, self.id);
    }

    /// Turn the claim into cache of `kind` in place, in its own entry, its
    /// bytes and their allocation kept throughout (a closed group's class,
    /// an active drafter going idle): claims in process take it as any
    /// cache.
    ///
    /// # Panics
    /// Where it is not allocated whole: cache is memory the process holds.
    pub fn into_cache(self, kind: CacheKind) -> Cached {
        let (ledger, id) = self.into_parts();
        let mut state = ledger.state();
        assert!(!state.reserved.contains_key(&id), "a preempted sequence's state is dropped, its bytes the preemption's");
        let entry = *state.entry(id);
        assert_eq!(entry.allocated, entry.bytes, "invariant violated: a claim turns cache only once allocated whole");
        state.turn(id, Holding::Cache(kind));
        drop(state);
        Cached { ledger, id, kind }
    }
}

impl Cached {
    pub fn kind(&self) -> CacheKind {
        self.kind
    }

    /// What it still holds: claims may have taken some in process.
    pub fn bytes(&self) -> u64 {
        self.ledger.state().entry(self.id).bytes
    }

    /// What claims took of it in process that its holder has not freed yet.
    pub fn releasing(&self) -> u64 {
        self.ledger.state().entry(self.id).releasing
    }

    /// Its holder freed `bytes` of what claims took of it: their claims may
    /// allocate them now.
    ///
    /// # Panics
    /// Past [`Self::releasing`].
    pub fn released(&mut self, bytes: u64) {
        self.ledger.state().released(self.id, bytes);
    }

    /// The store it is held in.
    pub fn store(&self) -> seismic::StoreIdentity {
        self.ledger.store()
    }

    /// Split off `bytes` as a holding of its own, of `kind`, in place (a
    /// granule handed from the bank to spare).
    pub fn split(&mut self, bytes: u64, kind: CacheKind) -> Cached {
        let mut state = self.ledger.state();
        let has = state.entry(self.id).bytes;
        let rest = has.checked_sub(bytes).expect("a split takes no more than the holding has");
        state.resize(self.id, rest);
        let id = state.insert(Holding::Cache(kind), bytes);
        state.allocate(id, bytes);
        drop(state);
        Cached { ledger: self.ledger.clone(), id, kind }
    }
}

impl Drop for Claim {
    /// The claim's bytes are free. Unwinding from a panic, nothing is
    /// recorded: the panic is the report, and the ledger it may have
    /// poisoned is never locked again on the way out.
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        self.ledger.state().release(self.id);
    }
}

impl Drop for Cached {
    /// The holding's bytes are free; nothing is recorded while unwinding
    /// ([`Claim`]'s drop).
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        self.ledger.state().release(self.id);
    }
}

impl Snapshot {
    /// Committed bytes held.
    pub fn committed(&self) -> u64 {
        self.committed
    }

    /// Cache bytes held.
    pub fn cache(&self) -> u64 {
        self.cache
    }

    /// Cache bytes of `kind` held.
    pub fn cache_of(&self, kind: CacheKind) -> u64 {
        self.cache_of[kind.index()]
    }

    fn held(&self) -> u64 {
        self.committed + self.cache
    }

    /// Bytes the process allocated: of what it holds, and what cache
    /// yielded to claims that its holders have not freed yet.
    pub fn allocated(&self) -> u64 {
        self.allocated
    }

    /// What cache yielded to claims in process that its holders have not
    /// freed yet ([`Cached::released`]).
    pub fn releasing(&self) -> u64 {
        self.releasing
    }

    /// What the process may hold in the store:
    /// `min(grant, total − others)`, which is
    /// `min(grant, allocated + available)`. Never counts others'
    /// compressible pages. Zero when blind.
    pub fn budget(&self) -> u64 {
        self.budget_for(seismic::Reclaim::Nothing)
    }

    /// What the process may hold counting `reclaim` as room: with
    /// [`seismic::Reclaim::CleanFileCache`] the reading's clean file-backed
    /// pages too, which a load's claim takes once (a claim holds them, so
    /// the next claim finds them held), still within the grant.
    pub fn budget_for(&self, reclaim: seismic::Reclaim) -> u64 {
        let more = match (&self.reading, reclaim) {
            (Ok(reading), seismic::Reclaim::CleanFileCache) => reading.reclaimable,
            _ => 0,
        };
        match (&self.reading, self.others()) {
            (Ok(reading), Some(others)) => (reading.total - others).saturating_add(more).min(self.grant.unwrap_or(u64::MAX)),
            _ => 0,
        }
    }

    /// Other programs' memory: `(total − available) − allocated`, all as
    /// of the reading: only the process's allocated bytes are its own in a
    /// reading, so a claim not yet allocated stays out of free room and is
    /// never read as others'. The single definition; never read
    /// separately. None when blind.
    pub fn others(&self) -> Option<u64> {
        self.others.filter(|_| self.reading.is_ok())
    }

    /// What the process may allocate beside what it has: what its claims
    /// hold unallocated and the room it leaves free, less what cache
    /// yielded to claims and still releases. Seismic admits this much more
    /// (`DeviceReader::read_ledgers`): a claim that took cache in process
    /// allocates those bytes only as their holder frees them.
    pub fn allocatable(&self) -> u64 {
        self.held().max(self.budget()).saturating_sub(self.allocated)
    }


    /// Room no entry holds: the budget less everything held.
    pub fn free(&self) -> u64 {
        self.free_for(seismic::Reclaim::Nothing)
    }

    /// Room no entry holds counting `reclaim` as room ([`Self::budget_for`]).
    pub fn free_for(&self, reclaim: seismic::Reclaim) -> u64 {
        self.budget_for(reclaim).saturating_sub(self.held())
    }

    /// The speed guarantee: free room the bank's target leaves for claims
    /// that arrive faster than eviction delivers. `in_process` is one prompt
    /// chunk's mandatory growth (from the formulas); outside the process it
    /// is the rise in others' memory between the last two readings. Both can
    /// arrive within one interval, so it is their sum.
    pub fn guarantee(&self, in_process: u64) -> u64 {
        let rise = self.others().zip(self.others_before).map_or(0, |(now, before)| now.saturating_sub(before));
        in_process.saturating_add(rise)
    }

    /// What free room lacks of the guarantee of `in_process`
    /// ([`Self::guarantee`]): what cache must give back now. Zero where
    /// free room keeps it; blind, the budget and so free room are none.
    pub fn short(&self, in_process: u64) -> u64 {
        self.guarantee(in_process).saturating_sub(self.free())
    }

    /// Whether the model unloads: the guarantee cannot be kept, free room
    /// below it ([`Self::guarantee`] of `in_process`) with the cache at its
    /// floor and preemption left nothing to take; or the store is blind.
    /// Committed memory never exceeds the budget (a claim past it is
    /// refused, but for a load's own, which counts the clean file cache
    /// once, and the OS drops that cache as the load allocates), so only
    /// the guarantee can fail; its outside term is others'
    /// rise between the last two readings, which falls to zero once they
    /// stop rising.
    pub fn unload(&self, in_process: u64) -> bool {
        self.reading.is_err() || self.free() < self.guarantee(in_process) && self.cache == 0 && self.preemptible == 0
    }
}

#[cfg(test)]
mod tests;
