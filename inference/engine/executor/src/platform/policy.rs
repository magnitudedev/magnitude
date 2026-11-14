//! The engine's sole fit policy over the topology's stores, grants and
//! placements ([`Fit`]).
//!
//! Seismic's topology reports stores, their capacities, the access paths
//! executors place storage along and the grants on them, and a status sample
//! reads each store's headroom and what is left of each grant. No store
//! keeps a reserve here: the speed guarantee of the store's ledger is the
//! only reserve (`memory::ledger::Estimate::guarantee`), and this module
//! applies the topology to every store a load reaches:
//!
//! - Stable fit compares a workload with each store's capacity, bounded by
//!   the applicable process limits for host RAM and by the grants of the
//!   paths it is placed along. It uses no current free memory. An
//!   executor's stable fit for a storage kind is the sum along its
//!   placement of each grant within its store's fit, a store reached by
//!   several paths or executors counted once.
//! - A live claim takes at most each store's observed headroom, and along
//!   a placement each grant's remaining reading bounds it too.
//!
//! Host RAM is read as the OS gives it ([`seismic::HostSample`]): where a
//! cgroup or job limits the process, its total and supply are the limit's.
//! No figure here counts what the process holds: that is the memory
//! ledger's alone.

use seismic::{
    AccessPathId, DeviceTopology, Executor, ExecutorId, Contract, HostSample, ObservationError, Status,
    Store, StoreId, StoreKind,
};
use std::{
    collections::BTreeMap,
    fmt,
    time::{Duration, Instant},
};

/// Host RAM's fit as `host` reads it: its capacity within the binding
/// scope's total, and the OS's supply ([`seismic::Supply`]) as headroom.
fn host_ram_fit(store: &Store, host: &HostSample) -> StoreFit {
    store_fit(store, store.capacity().value.min(host.total), host.supply.total(), MemoryConstraint::HostRam)
}

/// `store`'s fit from its capacity within process limits and its headroom.
fn store_fit(store: &Store, capacity_bytes: u64, headroom_bytes: u64, constraint: MemoryConstraint) -> StoreFit {
    StoreFit { store: store.id(), capacity_bytes, headroom_bytes, constraint }
}

/// The bytes one round of the page-release probe takes and frees, at most
/// ([`page_release`]).
const PAGE_RELEASE_ROUND_BYTES: usize = 256 << 20;

/// The page-release probe's rounds a sample times back to back, and its
/// samples: one round alone varies twofold between runs on an idle
/// machine, the median over five samples of four does not
/// ([`page_release`]).
const PAGE_RELEASE_ROUNDS_PER_SAMPLE: usize = 4;
const PAGE_RELEASE_SAMPLES: usize = 5;

/// The rate a load releases memory at, measured and positive: a finished
/// release's bytes over the span they took to release (its wait for queued
/// steps apart, `ResizeTimes::releasing`), the slowest
/// measured; before any, the machine's page-release rate
/// ([`Self::of_machine`]).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "f64", into = "f64")]
pub struct ReleaseRate(f64);

/// A release rate is finite by construction ([`ReleaseRate::new`]).
impl Eq for ReleaseRate {}

impl TryFrom<f64> for ReleaseRate {
    type Error = String;

    fn try_from(rate: f64) -> Result<Self, String> {
        match rate.is_finite() && rate > 0.0 {
            true => Ok(Self(rate)),
            false => Err(format!("a release rate is measured and positive, not {rate}")),
        }
    }
}

impl From<ReleaseRate> for f64 {
    fn from(rate: ReleaseRate) -> f64 {
        rate.0
    }
}

impl ReleaseRate {
    /// # Panics
    /// Where `bytes_per_second` is not a positive finite rate.
    pub fn new(bytes_per_second: f64) -> Self {
        Self::try_from(bytes_per_second).unwrap_or_else(|error| panic!("{error}"))
    }

    /// The rate the machine returns pages to the OS at, as a release is
    /// counted on to ([`page_release`]): a generic figure of the machine,
    /// measured once and kept ([`machine_rate`]).
    pub fn of_machine(machine: Option<(&std::path::Path, &DeviceTopology)>) -> Self {
        Self::new(page_release_rate(machine))
    }

    pub fn bytes_per_second(&self) -> f64 {
        self.0
    }
}

/// A rate the machine's figures keep: host RAM's page-release rate, a
/// volume's sequential write rate.
#[derive(serde::Serialize, serde::Deserialize)]
struct MachineRate {
    bytes_per_second: f64,
}

impl seismic::measure::Stored for MachineRate {
    const SCHEMA: &'static str = "machine-rate-v1";
}

/// The most bytes a stored [`MachineRate`] is read as.
const MACHINE_RATE_RECORD_BYTES: u64 = 4096;

/// A generic figure of the machine, measured once: where `figure` names a
/// kernel cache and the figure's key there, the rate its machine figures
/// (`KernelCache::machine_figures`) keep under that key, `measure`d and
/// stored there when they keep none; elsewhere, `measure`d.
pub(super) fn machine_rate<E>(
    figure: Option<(&std::path::Path, String)>,
    measure: impl FnOnce() -> Result<f64, E>,
) -> Result<f64, E> {
    let Some((cache, key)) = figure else { return measure() };
    let figures = crate::kernel_cache::KernelCache::machine_figures(cache);
    match figures.load::<MachineRate>(&key, MACHINE_RATE_RECORD_BYTES) {
        Ok(Some(MachineRate { bytes_per_second })) if bytes_per_second.is_finite() && bytes_per_second > 0.0 => {
            return Ok(bytes_per_second);
        }
        Ok(_) => {}
        Err(error) => tracing::warn!("machine figures: the {key} rate stored is unreadable: {error}"),
    }
    let bytes_per_second = measure()?;
    if let Err(error) = figures.store(&key, &MachineRate { bytes_per_second }, MACHINE_RATE_RECORD_BYTES) {
        tracing::warn!("machine figures: the {key} rate could not be stored: {error}");
    }
    Ok(bytes_per_second)
}

/// The host's identity in `topology` (`PhysicalIdentity::Host`): its CPU
/// as the OS reports it and its RAM, what a machine's figures are keyed by.
fn host_identity(topology: &DeviceTopology) -> Option<String> {
    topology.physical_devices().iter().find_map(|device| match device.identity() {
        seismic::PhysicalIdentity::Host { machine } => Some(machine.clone()),
        _ => None,
    })
}

/// The machine's page-release rate, kept under
/// `page-release-median:{host}` (named by its method, so a figure another
/// method stored is never read as its; [`machine_rate`]); the probe runs at most once
/// per process, bounded by half of host RAM's supply
/// (`seismic::host_sample`), so it never causes the pressure a release
/// relieves.
fn page_release_rate(machine: Option<(&std::path::Path, &DeviceTopology)>) -> f64 {
    static PROBED: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    let key = machine.and_then(|(cache, topology)| Some((cache, format!("page-release-median:{}", host_identity(topology)?))));
    let probe = || {
        let room = seismic::host_sample().supply.total();
        let bound = usize::try_from(room / 2).unwrap_or(usize::MAX).max(4096);
        page_release(bound.min(PAGE_RELEASE_ROUND_BYTES))
    };
    machine_rate(key, || Ok::<_, std::convert::Infallible>(*PROBED.get_or_init(probe))).unwrap_or_else(|never| match never {})
}

/// The rate the machine returns pages to the OS at, as a release is
/// counted on to: one round discarded, then [`PAGE_RELEASE_SAMPLES`]
/// samples of [`PAGE_RELEASE_ROUNDS_PER_SAMPLE`] rounds back to back of
/// `bytes` each ([`page_round`]), the median sample's rate. A release is
/// help counted on, so a smaller rate is the safe side: the median, never
/// the fastest.
fn page_release(bytes: usize) -> f64 {
    page_round(bytes);
    let mut samples = (0..PAGE_RELEASE_SAMPLES)
        .map(|_| {
            let released = (0..PAGE_RELEASE_ROUNDS_PER_SAMPLE).map(|_| page_round(bytes)).sum::<Duration>();
            (PAGE_RELEASE_ROUNDS_PER_SAMPLE * bytes) as f64 / released.as_secs_f64().max(f64::MIN_POSITIVE)
        })
        .collect::<Vec<_>>();
    samples.sort_by(f64::total_cmp);
    samples[PAGE_RELEASE_SAMPLES / 2]
}

/// One round of the page-release probe: a byte written to every page of
/// `bytes` of new memory, then the memory freed, the free timed.
fn page_round(bytes: usize) -> Duration {
    let page = 4096;
    let layout = std::alloc::Layout::from_size_align(bytes, page).expect("the probe's layout");
    // SAFETY: a non-zero layout; every write lies within it, and it is
    // freed with the same layout.
    unsafe {
        let memory = std::alloc::alloc(layout);
        assert!(!memory.is_null(), "the page-release probe could not take {bytes} bytes");
        for offset in (0..bytes).step_by(page) {
            std::ptr::write_volatile(memory.add(offset), 1);
        }
        let started = Instant::now();
        std::alloc::dealloc(memory, layout);
        started.elapsed()
    }
}

/// Why fit could not answer: never a verdict that a demand does not fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryPolicyError {
    /// A required observation is unavailable; missing observations never
    /// authorize allocation.
    Observation(ObservationError),
    /// The fit subject belongs to another topology revision.
    ForeignIdentifier,
    /// Storage demanded of an executor that places none of that kind: an
    /// unavailable executor.
    Unplaced {
        executor: ExecutorId,
        kind: Contract,
    },
}

impl fmt::Display for MemoryPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Observation(error) => error.fmt(f),
            Self::ForeignIdentifier => {
                f.write_str("fit subject belongs to another topology revision")
            }
            Self::Unplaced { executor, kind } => {
                write!(f, "executor {executor:?} places no {kind:?} storage")
            }
        }
    }
}

impl std::error::Error for MemoryPolicyError {}

/// What a store holds for an executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainRole {
    /// The executor's device storage is placed on it.
    Allocation,
    /// Host-side work (staging, host tables) and nothing of device storage.
    Staging,
}

impl DomainRole {
    /// A store `executor`'s device storage is placed on holds its
    /// allocations; any other store it charges holds host-side work.
    pub fn of(topology: &DeviceTopology, executor: &Executor, store: StoreId) -> Self {
        let allocates = executor.paths(Contract::DeviceOnly).iter().any(|id| {
            topology
                .access_path(*id)
                .expect("placement belongs to its topology")
                .store()
                == store
        });
        if allocates {
            Self::Allocation
        } else {
            Self::Staging
        }
    }
}

/// What bounds a fit: a store's own room or an access path's grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryConstraint {
    /// Host RAM and its applicable process limits.
    HostRam,
    /// A device memory store: memory the OS does not manage.
    DeviceLocal,
    /// What the executor's driver lets this process place along an access
    /// path (CUDA's usable memory, a Vulkan heap's budget, Metal's working
    /// set), as reported and, where a reading exists, what is left of it.
    Grant,
    /// A volume's space available to this process.
    Volume,
}

impl fmt::Display for MemoryConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::HostRam => "host RAM",
            Self::DeviceLocal => "device memory",
            Self::Grant => "executor placement grant",
            Self::Volume => "volume space",
        })
    }
}

/// One store's fit: its capacity within process limits, what
/// metadata-only fit plans on, and its live headroom, what a live claim
/// may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreFit {
    pub store: StoreId,
    /// The store's capacity, host RAM's within its binding scope.
    pub capacity_bytes: u64,
    /// What a live claim may take: host RAM's supply, a volume's space, a
    /// device memory's capacity, each within what is left of its grants.
    pub headroom_bytes: u64,
    /// What bounds the store's room.
    pub constraint: MemoryConstraint,
}

/// What an executor's placement of one storage kind holds, stable and live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlacementFit {
    /// Along the placement, each grant within its store's stable fit.
    pub stable_bytes: u64,
    /// Along the placement, what is left of each grant within its store's
    /// live ceiling: what a live claim may take.
    pub ceiling_bytes: u64,
}

/// Whether fit plans on stable capacity or on live headroom.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitMode {
    /// Capacity within process limits and grants: no current free memory.
    Stable,
    /// Observed headroom and what is left of each grant.
    Live,
}

/// Bytes of one storage kind on one executor. Demands are placed in order,
/// sharing store and path budgets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageDemand {
    /// The executor whose placement the bytes go along.
    pub executor: ExecutorId,
    /// The storage kind, whose placement it is.
    pub kind: Contract,
    pub bytes: u64,
}

/// A demand that does not fit, by the store where it ran out of room.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FitRefusal {
    /// The last store the demand reached.
    pub store: StoreId,
    /// The demand's bytes.
    pub required_bytes: u64,
    /// Its bytes no store or grant had room for.
    pub missing_bytes: u64,
    /// What ran out: the store's room or a grant.
    pub constraint: MemoryConstraint,
}

/// Where a demand's bytes landed, and what of it did not fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemandFit {
    /// The bytes placed in each store, in store order.
    pub placed: Vec<(StoreId, u64)>,
    /// Each demand some of whose bytes did not fit.
    pub refused: Vec<FitRefusal>,
}

impl DemandFit {
    /// Whether every byte of the demand was placed.
    pub fn fits(&self) -> bool {
        self.refused.is_empty()
    }
}


/// The sole interpretation of stores and grants for engine fit questions.
pub struct Fit<'a> {
    topology: &'a DeviceTopology,
    status: &'a Status,
    /// The reserves of the volumes this process writes, where it reads them.
    volumes: Option<&'a super::VolumeReserves>,
}

impl<'a> Fit<'a> {
    /// Fit over `topology`'s stores and grants as `status` reads them.
    pub fn new(topology: &'a DeviceTopology, status: &'a Status) -> Self {
        Self { topology, status, volumes: None }
    }

    /// This fit, reading volumes against `volumes`' reserves: a volume
    /// this process does not write keeps nothing free for it.
    pub fn with_volumes(self, volumes: &'a super::VolumeReserves) -> Self {
        Self { volumes: Some(volumes), ..self }
    }

    /// Store `id`'s fit: its capacity, host RAM's within its binding scope,
    /// and its headroom within what is left of every grant on it: host
    /// RAM's supply, a volume's space less the margin its writer keeps, a
    /// device memory's capacity (what the process holds there is its
    /// ledger's).
    pub fn store(&self, id: StoreId) -> Result<StoreFit, MemoryPolicyError> {
        let store = self
            .topology
            .store(id)
            .ok_or(MemoryPolicyError::ForeignIdentifier)?;
        let (constraint, margin, headroom_bytes) = match store.kind() {
            StoreKind::HostRam => return Ok(host_ram_fit(store, &self.status.sample.host)),
            StoreKind::DeviceMemory(_) => (MemoryConstraint::DeviceLocal, 0, store.capacity().value),
            // A volume keeps the margin its writes need (`super::volume`),
            // and none where this process writes nothing.
            StoreKind::Volume(volume) => (
                MemoryConstraint::Volume,
                self.volumes.and_then(|volumes| volumes.margin(volume)).unwrap_or(0),
                self.status.volume_space(id).map_err(MemoryPolicyError::Observation)?,
            ),
        };
        // A store's own figure knows only this process; what is left of a
        // grant on it is the driver's, which other processes' memory lowers.
        let headroom_bytes = self
            .topology
            .access_paths()
            .iter()
            .filter(|path| path.store() == id && self.status.sample.executors.contains(&path.executor()))
            .filter_map(|path| self.status.grant_left(path))
            .try_fold(headroom_bytes, |least, left| match left {
                Ok(left) => Ok(least.min(left.value)),
                Err(error) => Err(MemoryPolicyError::Observation(error.clone())),
            })?;
        Ok(store_fit(store, store.capacity().value.saturating_sub(margin), headroom_bytes.saturating_sub(margin), constraint))
    }

    /// Host RAM's fit from a host sample alone, as [`Fit::store`] gives it
    /// from a whole status: for a reader that opens no executor, which has
    /// only `host`.
    pub fn host_ram(topology: &DeviceTopology, host: &HostSample) -> StoreFit {
        host_ram_fit(topology.host_ram(), host)
    }

    /// What access path `id`'s grant leaves a claim: the reported grant,
    /// within what a live reading says is left of it. An ungranted path is
    /// bounded by its store alone.
    fn grant(&self, id: AccessPathId, mode: FitMode) -> Result<u64, MemoryPolicyError> {
        let path = self
            .topology
            .access_path(id)
            .ok_or(MemoryPolicyError::ForeignIdentifier)?;
        let Some(grant) = path.grant() else {
            return Ok(u64::MAX);
        };
        let left = match mode {
            FitMode::Stable => grant.value,
            FitMode::Live => match self.status.grant_left(path) {
                Some(Ok(left)) => grant.value.min(left.value),
                Some(Err(error)) => return Err(MemoryPolicyError::Observation(error.clone())),
                // An unopened executor has only its reported stable grant.
                None => grant.value,
            },
        };
        Ok(left)
    }

    /// What `executor`'s placement of `kind` storage holds: the sum along
    /// its paths of each grant within its store's fit.
    pub fn placement(
        &self,
        executor: &Executor,
        kind: Contract,
    ) -> Result<PlacementFit, MemoryPolicyError> {
        self.storage_beside(executor, &[kind], &[])
    }

    /// What `executor`'s placements of `kinds` hold together beside the
    /// direct demands `direct`, which take their stores' room first. A
    /// store or grant several kinds reach is counted once.
    pub fn storage_beside(
        &self,
        executor: &Executor,
        kinds: &[Contract],
        direct: &[(StoreId, u64)],
    ) -> Result<PlacementFit, MemoryPolicyError> {
        let demands: Vec<_> = kinds
            .iter()
            .map(|&kind| StorageDemand {
                executor: executor.id(),
                kind,
                bytes: u64::MAX,
            })
            .collect();
        let available = |mode| -> Result<u64, MemoryPolicyError> {
            let (_, storage) = self.place(&demands, direct, mode)?;
            Ok(storage.into_iter().sum())
        };
        Ok(PlacementFit {
            stable_bytes: available(FitMode::Stable)?,
            ceiling_bytes: available(FitMode::Live)?,
        })
    }

    /// The host RAM demand of `staged` bytes that `executor`'s backend
    /// stages while it fills device storage. Transfer staging is in host
    /// RAM on every backend (§2.3), and every claim's staged bytes are part
    /// of its device storage demand. Where host RAM is a store of the
    /// executor's device placement ([`DomainRole::Allocation`]), that demand
    /// already holds them in host RAM; only where host RAM is the
    /// executor's staging store are they a host demand of their own.
    pub fn host_staging(&self, executor: &Executor, staged: u64) -> u64 {
        match DomainRole::of(self.topology, executor, self.topology.host_ram().id()) {
            DomainRole::Allocation => 0,
            DomainRole::Staging => staged,
        }
    }

    /// Direct demands (host heap, staging, volume downloads) and executor
    /// storage share one budget per store and one budget per access grant.
    pub fn demand(
        &self,
        storage: &[StorageDemand],
        direct: &[(StoreId, u64)],
        mode: FitMode,
    ) -> Result<DemandFit, MemoryPolicyError> {
        self.place(storage, direct, mode).map(|(fit, _)| fit)
    }

    /// [`Self::demand`], with the bytes placed of each storage demand in
    /// order.
    fn place(
        &self,
        storage: &[StorageDemand],
        direct: &[(StoreId, u64)],
        mode: FitMode,
    ) -> Result<(DemandFit, Vec<u64>), MemoryPolicyError> {
        let room_of = |fit: &StoreFit| match mode {
            FitMode::Stable => fit.capacity_bytes,
            FitMode::Live => fit.headroom_bytes,
        };
        let mut stores = BTreeMap::<StoreId, u64>::new();
        let mut grants = BTreeMap::<AccessPathId, u64>::new();
        let mut placed = BTreeMap::<StoreId, u64>::new();
        let mut placed_storage = Vec::with_capacity(storage.len());
        let mut refused = Vec::new();
        for &(store, bytes) in direct {
            let fit = self.store(store)?;
            let room = stores.entry(store).or_insert(room_of(&fit));
            let take = bytes.min(*room);
            *room -= take;
            *placed.entry(store).or_default() += take;
            if take < bytes {
                refused.push(FitRefusal {
                    store,
                    required_bytes: bytes,
                    missing_bytes: bytes - take,
                    constraint: fit.constraint,
                });
            }
        }
        for demand in storage {
            let executor = self
                .topology
                .executor(demand.executor)
                .ok_or(MemoryPolicyError::ForeignIdentifier)?;
            let mut left = demand.bytes;
            let mut last = None;
            for &id in executor.paths(demand.kind) {
                let path = self
                    .topology
                    .access_path(id)
                    .expect("placement belongs to its topology");
                let fit = self.store(path.store())?;
                let room = stores.entry(path.store()).or_insert(room_of(&fit));
                let grant = grants.entry(id).or_insert(self.grant(id, mode)?);
                let constraint = if *room <= *grant {
                    fit.constraint
                } else {
                    MemoryConstraint::Grant
                };
                let take = left.min(*room).min(*grant);
                *room -= take;
                *grant -= take;
                left -= take;
                *placed.entry(path.store()).or_default() += take;
                last = Some((path.store(), constraint));
                if left == 0 {
                    break;
                }
            }
            placed_storage.push(demand.bytes - left);
            if left != 0 {
                let (store, constraint) = last.ok_or(MemoryPolicyError::Unplaced {
                    executor: demand.executor,
                    kind: demand.kind,
                })?;
                refused.push(FitRefusal {
                    store,
                    required_bytes: demand.bytes,
                    missing_bytes: left,
                    constraint,
                });
            }
        }
        let fit = DemandFit {
            placed: placed.into_iter().collect(),
            refused,
        };
        Ok((fit, placed_storage))
    }
}

/// One fresh observation of a store an opened executor uses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomainReading {
    /// What the store holds for the executor.
    pub role: DomainRole,
    pub domain: StoreId,
    /// What a live claim may take: host RAM's supply, a volume's space, a
    /// device memory's capacity, each within what is left of its grants.
    pub headroom_bytes: u64,
    /// The store across revisions, as other processes name it.
    pub identity: seismic::StoreIdentity,
    /// What bounds `headroom_bytes`.
    pub constraint: MemoryConstraint,
}

impl fmt::Display for DomainReading {
    /// The reading a claim's room was judged on.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} headroom {} bytes", self.constraint, self.headroom_bytes)
    }
}

/// Fresh store readings for the executor's placements and host-side work.
/// Allocation roles identify stores reached by device storage, not a
/// promise that device storage occupies only one store.
pub fn observe_domains(devices: &seismic::Devices, device: &seismic::Device) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let status = devices.device_status(device);
    domain_readings(device, &status)
}

/// The readings of every store `device` places storage on, and host RAM.
pub(crate) fn domain_readings(device: &seismic::Device, status: &Status) -> Result<Vec<DomainReading>, MemoryPolicyError> {
    let topology = device.topology();
    let fit = Fit::new(topology, status);
    let stores: std::collections::BTreeSet<_> = crate::memory::CLAIMED
        .iter()
        .flat_map(|kind| device.info().paths(*kind))
        .map(|id| {
            topology
                .access_path(*id)
                .expect("placement belongs to its topology")
                .store()
        })
        .chain(std::iter::once(topology.host_ram().id()))
        .collect();
    let readings: Result<Vec<_>, MemoryPolicyError> = stores
        .into_iter()
        .map(|store| {
            let reading = fit.store(store)?;
            Ok(DomainReading {
                role: DomainRole::of(topology, device.info(), store),
                domain: store,
                headroom_bytes: reading.headroom_bytes,
                identity: topology
                    .store_identity(store)
                    .expect("observed store belongs to topology"),
                constraint: reading.constraint,
            })
        })
        .collect();
    // Opt-in investigation trace: the exact sample and readings that decide
    // the ceiling, rather than a sample taken again after a failed worker
    // has exited.
    if std::env::var_os("MAGNITUDE_TRACE_MEMORY_BUDGET").is_some() {
        eprintln!(
            "magnitude-engine: memory budget pid={} device={} status={status:?} readings={readings:?}",
            std::process::id(),
            device.info().selector(),
        );
    }
    readings
}

#[cfg(test)]
mod tests {
    //! Fit over the recorded machines' topologies
    //! (`seismic::fixtures::recorded`). Status
    //! values (headroom, what is left of a grant) are each test's
    //! input; every structural value is the machine's.
    use super::*;
    use seismic::fixtures::recorded;
    use seismic::{BackendName, Reported, ReportedBy, RuntimeStatus, Sample};
    use std::time::SystemTime;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// Host RAM whose binding scope is the whole host, with `supply` bytes
    /// the OS can give.
    fn host(supply: u64) -> HostSample {
        crate::memory::ledger::tests::host_sample(u64::MAX, supply)
    }

    /// Host RAM within a limit of `total` bytes that leaves `supply`.
    fn limited(total: u64, supply: u64) -> HostSample {
        crate::memory::ledger::tests::host_sample(total, supply)
    }

    /// Status of an unopened machine: `host` alone, no grant read and no
    /// charge.
    fn status(host: HostSample) -> Status {
        Status {
            sample: Sample {
                taken: SystemTime::now(),
                host,
                volumes: Vec::new(),
                executors: Vec::new(),
                grants: Vec::new(),
            },
            runtime: RuntimeStatus::default(),
        }
    }

    /// Status of an unopened machine sampled for every executor of
    /// `topology`: `host` alone, no grant read and no charge.
    fn sampled(topology: &DeviceTopology, host: HostSample) -> Status {
        let mut status = status(host);
        status.sample.executors = topology
            .executors()
            .iter()
            .map(|executor| executor.id())
            .collect();
        status
    }

    /// The first executor of `backend`.
    fn executor(topology: &DeviceTopology, backend: BackendName) -> &Executor {
        topology
            .executors()
            .iter()
            .find(|executor| executor.backend() == backend)
            .unwrap()
    }

    /// The machine's one device memory store.
    fn device_memory(topology: &DeviceTopology) -> StoreId {
        let stores = topology
            .stores()
            .iter()
            .filter(|store| matches!(store.kind(), StoreKind::DeviceMemory(_)))
            .map(|store| store.id())
            .collect::<Vec<_>>();
        let [store] = stores[..] else {
            panic!("the machine has one device memory store");
        };
        store
    }

    /// What the reported grant of the `index`th path of `executor`'s
    /// placement of `kind` lets a claim take.
    fn grant_room(
        topology: &DeviceTopology,
        executor: &Executor,
        kind: Contract,
        index: usize,
    ) -> u64 {
        let path = topology
            .access_path(executor.paths(kind)[index])
            .unwrap();
        path.grant().unwrap().value
    }

    fn demand(executor: &Executor, kind: Contract, bytes: u64) -> StorageDemand {
        StorageDemand {
            executor: executor.id(),
            kind,
            bytes,
        }
    }

    /// The bytes `fit` placed on `store`.
    fn placed(fit: &DemandFit, store: StoreId) -> u64 {
        fit.placed
            .iter()
            .filter(|(on, _)| *on == store)
            .map(|(_, bytes)| bytes)
            .sum()
    }

    /// Another process's memory on a discrete device reaches its store's
    /// fit through what the driver says is left of the grant, which the
    /// process's own charges cannot show.
    #[test]
    fn a_stores_headroom_follows_what_is_left_of_its_grants() {
        let rtx3090 = recorded::rtx3090();
        let cuda = executor(&rtx3090, BackendName::Cuda);
        let path = rtx3090.access_path(cuda.paths(Contract::DeviceOnly)[0]).unwrap();
        let mut opened = sampled(&rtx3090, host(40 * GIB));
        let store = path.store();
        let room = |status: &Status| Fit::new(&rtx3090, status).store(store).unwrap().headroom_bytes;
        let capacity = room(&opened);
        opened.runtime.grants = vec![(path.id(), Ok(Reported::new(GIB, ReportedBy::CudaFree)))];
        assert!(capacity > GIB);
        assert_eq!(room(&opened), GIB);
    }

    /// Host RAM's stable fit is its capacity and its live fit its headroom:
    /// stable fit does not follow headroom.
    #[test]
    fn host_ram_fit_is_its_capacity_and_headroom() {
        let topology = recorded::rtx3090();
        let id = topology.host_ram().id();
        let capacity = topology.host_ram().capacity().value;
        let reading = |headroom| Fit::new(&topology, &status(host(headroom))).store(id).unwrap();
        let above = reading(16 * GIB);
        assert_eq!(above.capacity_bytes, capacity);
        assert_eq!(above.headroom_bytes, 16 * GIB);
        assert_eq!(above.constraint, MemoryConstraint::HostRam);
        assert_eq!(reading(GIB).capacity_bytes, capacity);
        assert_eq!(reading(0).headroom_bytes, 0);
    }

    /// A store an executor's device storage is placed on holds its
    /// allocations; any other store it charges holds host-side work.
    #[test]
    fn domain_roles_follow_the_device_storage_placement() {
        let rtx3090 = recorded::rtx3090();
        let (gpu, host_ram) = (device_memory(&rtx3090), rtx3090.host_ram().id());
        for backend in [BackendName::Cuda, BackendName::Vulkan] {
            let executor = executor(&rtx3090, backend);
            assert_eq!(
                DomainRole::of(&rtx3090, executor, gpu),
                DomainRole::Allocation
            );
            assert_eq!(
                DomainRole::of(&rtx3090, executor, host_ram),
                DomainRole::Staging
            );
        }
        let cpu = executor(&rtx3090, BackendName::Cpu);
        assert_eq!(
            DomainRole::of(&rtx3090, cpu, host_ram),
            DomainRole::Allocation
        );
        // GB10's CUDA allocates in host RAM.
        let gb10 = recorded::gb10();
        let cuda = executor(&gb10, BackendName::Cuda);
        assert_eq!(
            DomainRole::of(&gb10, cuda, gb10.host_ram().id()),
            DomainRole::Allocation
        );
        // Device storage over two paths allocates in both stores.
        let strix = recorded::strix_halo_windows();
        let vulkan = executor(&strix, BackendName::Vulkan);
        for store in [device_memory(&strix), strix.host_ram().id()] {
            assert_eq!(
                DomainRole::of(&strix, vulkan, store),
                DomainRole::Allocation
            );
        }
    }

    /// On the M4 Pro, Metal's stable fit is the smaller of its recommended
    /// working set and host RAM's capacity: the working set's while host RAM
    /// is larger, host RAM's fit under a process limit below it.
    #[test]
    fn metal_fit_respects_the_recommended_working_set() {
        let topology = recorded::m4_pro();
        let metal = executor(&topology, BackendName::Metal);
        let working_set = grant_room(&topology, metal, Contract::DeviceOnly, 0);
        let unlimited = sampled(&topology, host(16 * GIB));
        let fit = Fit::new(&topology, &unlimited);
        let host_fit = fit.store(topology.host_ram().id()).unwrap().capacity_bytes;
        assert!(working_set < host_fit);
        for kind in crate::memory::CLAIMED {
            assert_eq!(
                fit.placement(metal, kind).unwrap().stable_bytes,
                working_set
            );
        }
        // The CPU's host RAM is ungranted: host RAM's fit.
        let cpu = executor(&topology, BackendName::Cpu);
        assert_eq!(
            fit.placement(cpu, Contract::DeviceOnly)
                .unwrap()
                .stable_bytes,
            host_fit
        );

        let limited = sampled(&topology, limited(32 * GIB, 16 * GIB));
        let fit = Fit::new(&topology, &limited);
        let host_fit = fit.store(topology.host_ram().id()).unwrap().capacity_bytes;
        assert!(host_fit < working_set);
        assert_eq!(
            fit.placement(metal, Contract::DeviceOnly)
                .unwrap()
                .stable_bytes,
            host_fit
        );
    }

    /// Metal's live ceiling is the smaller of the working set left (the
    /// sample's grant status) and host RAM's headroom. The working set bounds
    /// what Metal may claim, never host RAM's own headroom.
    #[test]
    fn metal_working_set_bounds_metals_ceiling_not_host_rams() {
        let topology = recorded::m4_pro();
        let metal = executor(&topology, BackendName::Metal);
        let path = metal.paths(Contract::DeviceOnly)[0];
        let host_ram = topology.host_ram().id();
        let mut sampled = sampled(&topology, host(20 * GIB));
        let with_left = |sampled: &mut Status, left| {
            sampled.sample.grants = vec![(
                path,
                Ok(Reported::new(left, ReportedBy::MetalWorkingSetUnallocated)),
            )];
        };
        // Working set left 4 GiB is tighter than host RAM's headroom.
        with_left(&mut sampled, 4 * GIB);
        let fit = Fit::new(&topology, &sampled);
        let host_ceiling = fit.store(host_ram).unwrap().headroom_bytes;
        assert!(4 * GIB < host_ceiling);
        let room = fit.placement(metal, Contract::DeviceOnly).unwrap();
        assert_eq!(room.ceiling_bytes, 4 * GIB);
        // The sample bounds the ceiling, not stable fit.
        assert_eq!(
            room.stable_bytes,
            grant_room(&topology, metal, Contract::DeviceOnly, 0)
        );
        let over = fit
            .demand(
                &[demand(metal, Contract::DeviceOnly, 5 * GIB)],
                &[],
                FitMode::Live,
            )
            .unwrap();
        assert_eq!(
            over.refused,
            [FitRefusal {
                store: host_ram,
                required_bytes: 5 * GIB,
                missing_bytes: GIB,
                constraint: MemoryConstraint::Grant,
            }]
        );
        // Host RAM's ceiling is tighter than the working set left.
        with_left(&mut sampled, 30 * GIB);
        let fit = Fit::new(&topology, &sampled);
        let room = fit.placement(metal, Contract::DeviceOnly).unwrap();
        assert_eq!(room.ceiling_bytes, host_ceiling);
        let over = fit
            .demand(
                &[demand(metal, Contract::DeviceOnly, host_ceiling + 1)],
                &[],
                FitMode::Live,
            )
            .unwrap();
        assert_eq!(over.refused[0].constraint, MemoryConstraint::HostRam);
        // A spent working set grants nothing but leaves host RAM's ceiling.
        with_left(&mut sampled, 0);
        let fit = Fit::new(&topology, &sampled);
        assert_eq!(
            fit.placement(metal, Contract::DeviceOnly)
                .unwrap()
                .ceiling_bytes,
            0
        );
        assert_eq!(fit.store(host_ram).unwrap().headroom_bytes, host_ceiling);
    }

    /// An integrated GPU (GB10's CUDA and Vulkan, Strix Halo's and Lunar
    /// Lake's Vulkan) fits the smaller of its grant's room and host RAM's
    /// fit, and a demand beyond it is refused by whichever is smaller.
    #[test]
    fn an_integrated_gpu_is_bounded_by_the_smaller_of_its_grant_and_host_ram() {
        for (topology, backend) in [
            (recorded::gb10(), BackendName::Cuda),
            (recorded::gb10(), BackendName::Vulkan),
            (recorded::strix_halo_linux(), BackendName::Vulkan),
            (recorded::lunar_lake(), BackendName::Vulkan),
        ] {
            let unopened = sampled(&topology, host(8 * GIB));
            let fit = Fit::new(&topology, &unopened);
            let host_fit = fit.store(topology.host_ram().id()).unwrap().capacity_bytes;
            let executor = executor(&topology, backend);
            let grant = grant_room(&topology, executor, Contract::DeviceOnly, 0);
            let room = fit.placement(executor, Contract::DeviceOnly).unwrap();
            let label = format!("{backend:?} on {}", executor.name());
            assert_eq!(room.stable_bytes, grant.min(host_fit), "{label}");
            let over = fit
                .demand(
                    &[demand(executor, Contract::DeviceOnly, room.stable_bytes + 1)],
                    &[],
                    FitMode::Stable,
                )
                .unwrap();
            let bound = if host_fit <= grant {
                MemoryConstraint::HostRam
            } else {
                MemoryConstraint::Grant
            };
            assert_eq!(over.refused[0].constraint, bound, "{label}");
        }
    }

    /// On GB10 every charge is host RAM's: CUDA's and Vulkan's device
    /// storage and direct host work spend host RAM's one fit.
    #[test]
    fn gb10_executors_and_host_work_share_host_ram() {
        let topology = recorded::gb10();
        let unopened = status(host(8 * GIB));
        let fit = Fit::new(&topology, &unopened);
        let host_ram = topology.host_ram().id();
        let host_fit = fit.store(host_ram).unwrap().capacity_bytes;
        let cuda = executor(&topology, BackendName::Cuda);
        let vulkan = executor(&topology, BackendName::Vulkan);
        let (work, half) = (GIB, (host_fit - GIB) / 2);
        let result = fit
            .demand(
                &[
                    demand(cuda, Contract::DeviceOnly, half),
                    demand(vulkan, Contract::DeviceOnly, host_fit - work - half + 1),
                ],
                &[(host_ram, work)],
                FitMode::Stable,
            )
            .unwrap();
        assert_eq!(result.placed, [(host_ram, host_fit)]);
        assert_eq!(
            result.refused,
            [FitRefusal {
                store: host_ram,
                required_bytes: host_fit - work - half + 1,
                missing_bytes: 1,
                constraint: MemoryConstraint::HostRam,
            }]
        );
    }

    /// CUDA's free figure for an integrated device is no bound (§2.5): its
    /// live grant is its total less what it holds, and host RAM's headroom
    /// bounds it below that.
    #[test]
    fn integrated_cuda_is_bounded_by_host_ram_and_what_it_holds() {
        let topology = recorded::gb10();
        let cuda = executor(&topology, BackendName::Cuda);
        let path = cuda.paths(Contract::DeviceOnly)[0];
        let total = topology.access_path(path).unwrap().grant().unwrap().value;
        let mut opened = sampled(&topology, host(40 * GIB));
        let holding = |opened: &mut Status, held| {
            opened.runtime.grants = vec![(
                path,
                Ok(Reported::new(total - held, ReportedBy::CudaTotalMemory)),
            )];
        };
        holding(&mut opened, 0);
        let fit = Fit::new(&topology, &opened);
        let host_ceiling = fit.store(topology.host_ram().id()).unwrap().headroom_bytes;
        assert_eq!(
            fit.placement(cuda, Contract::DeviceOnly)
                .unwrap()
                .ceiling_bytes,
            host_ceiling
        );
        // Holding all but 8 GiB of its total leaves it 8 GiB, below host
        // RAM's headroom.
        holding(&mut opened, total - 8 * GIB);
        let fit = Fit::new(&topology, &opened);
        assert!(8 * GIB < host_ceiling);
        assert_eq!(
            fit.placement(cuda, Contract::DeviceOnly)
                .unwrap()
                .ceiling_bytes,
            8 * GIB
        );
    }

    /// Two executors on one store: each of the RTX 3090's CUDA and Vulkan
    /// fits the smaller of its grant's room and its memory's fit, and a
    /// demand split across both cannot exceed that fit once.
    #[test]
    fn cuda_and_vulkan_on_one_device_share_its_fit() {
        let topology = recorded::rtx3090();
        let gpu = device_memory(&topology);
        let capacity = topology.store(gpu).unwrap().capacity().value;
        let sampled = sampled(&topology, host(16 * GIB));
        let fit = Fit::new(&topology, &sampled);
        let own = fit.store(gpu).unwrap();
        assert_eq!(own.constraint, MemoryConstraint::DeviceLocal);
        assert_eq!(own.capacity_bytes, capacity);
        let cuda = executor(&topology, BackendName::Cuda);
        let vulkan = executor(&topology, BackendName::Vulkan);
        for executor in [cuda, vulkan] {
            let grant = grant_room(&topology, executor, Contract::DeviceOnly, 0);
            assert_eq!(
                fit.placement(executor, Contract::DeviceOnly)
                    .unwrap()
                    .stable_bytes,
                grant.min(own.capacity_bytes)
            );
        }
        let half = own.capacity_bytes / 2;
        let split = fit
            .demand(
                &[
                    demand(cuda, Contract::DeviceOnly, half),
                    demand(vulkan, Contract::DeviceOnly, own.capacity_bytes - half + 1),
                ],
                &[],
                FitMode::Stable,
            )
            .unwrap();
        assert_eq!(split.placed, [(gpu, own.capacity_bytes)]);
        assert_eq!(
            split.refused,
            [FitRefusal {
                store: gpu,
                required_bytes: own.capacity_bytes - half + 1,
                missing_bytes: 1,
                constraint: MemoryConstraint::DeviceLocal,
            }]
        );
    }

    /// The A10-8Q slice's stable fit is its capacity: its driver's own
    /// reserve is all it keeps.
    #[test]
    fn a_small_dedicated_device_fits_its_capacity() {
        let topology = recorded::a10_8q();
        let slice = device_memory(&topology);
        let capacity = topology.store(slice).unwrap().capacity().value;
        let sampled = sampled(&topology, host(16 * GIB));
        let fit = Fit::new(&topology, &sampled);
        let own = fit.store(slice).unwrap();
        assert_eq!(own.capacity_bytes, capacity);
        // CUDA's usable memory is below the slice's fit; Vulkan's heap, the
        // slice's capacity, is above it.
        let cuda = executor(&topology, BackendName::Cuda);
        let usable = grant_room(&topology, cuda, Contract::DeviceOnly, 0);
        assert!(usable < own.capacity_bytes);
        assert_eq!(
            fit.placement(cuda, Contract::DeviceOnly)
                .unwrap()
                .stable_bytes,
            usable
        );
        let vulkan = executor(&topology, BackendName::Vulkan);
        assert_eq!(
            fit.placement(vulkan, Contract::DeviceOnly)
                .unwrap()
                .stable_bytes,
            own.capacity_bytes
        );
    }

    /// A shared path's grant is one budget for every storage kind placed on
    /// it: the RTX 3090's Vulkan places device and upload storage (ReBAR)
    /// on one path, whose budget left bounds both together.
    #[test]
    fn remaining_path_grant_is_shared_by_device_and_upload() {
        let topology = recorded::rtx3090();
        let vulkan = executor(&topology, BackendName::Vulkan);
        let path = vulkan.paths(Contract::DeviceOnly)[0];
        assert_eq!(vulkan.paths(Contract::Upload), [path]);
        let mut sampled = sampled(&topology, host(16 * GIB));
        let gpu = device_memory(&topology);
        // A budget left 10 GiB.
        sampled.sample.grants = vec![(
            path,
            Ok(Reported::new(10 * GIB, ReportedBy::VulkanHeapBudget)),
        )];
        let fit = Fit::new(&topology, &sampled);
        assert_eq!(fit.store(gpu).unwrap().headroom_bytes, 10 * GIB);
        let result = fit
            .demand(
                &[
                    demand(vulkan, Contract::DeviceOnly, 7 * GIB),
                    demand(vulkan, Contract::Upload, 4 * GIB),
                ],
                &[],
                FitMode::Live,
            )
            .unwrap();
        assert_eq!(result.placed, [(gpu, 10 * GIB)]);
        assert_eq!(
            result.refused,
            [FitRefusal {
                store: gpu,
                required_bytes: 4 * GIB,
                missing_bytes: GIB,
                constraint: MemoryConstraint::DeviceLocal,
            }]
        );
    }

    /// A placement over two paths: Strix Halo on Windows (#207) places
    /// device storage on its own memory up to that store's fit, then on
    /// host RAM under the heaps' total beyond it. A demand larger than its
    /// own memory continues onto host RAM; beyond both, the refusal names
    /// the path that ran out: its grant, or host RAM when other work has
    /// taken host RAM's fit.
    #[test]
    fn a_placement_over_two_paths_continues_onto_host_ram() {
        let topology = recorded::strix_halo_windows();
        let unopened = sampled(&topology, host(16 * GIB));
        let fit = Fit::new(&topology, &unopened);
        let (radeon, host_ram) = (device_memory(&topology), topology.host_ram().id());
        let own_fit = fit.store(radeon).unwrap().capacity_bytes;
        let host_fit = fit.store(host_ram).unwrap().capacity_bytes;
        let vulkan = executor(&topology, BackendName::Vulkan);
        // The own-memory path's grant is its store's capacity, whose room is
        // its fit; the host path's room is below host RAM's fit.
        assert_eq!(grant_room(&topology, vulkan, Contract::DeviceOnly, 0), own_fit);
        let beyond = grant_room(&topology, vulkan, Contract::DeviceOnly, 1);
        assert!(beyond < host_fit);
        let room = fit.placement(vulkan, Contract::DeviceOnly).unwrap();
        assert_eq!(room.stable_bytes, own_fit + beyond);

        let spilled = fit
            .demand(
                &[demand(vulkan, Contract::DeviceOnly, own_fit + GIB)],
                &[],
                FitMode::Stable,
            )
            .unwrap();
        assert!(spilled.fits());
        assert_eq!(
            (placed(&spilled, radeon), placed(&spilled, host_ram)),
            (own_fit, GIB)
        );

        let over = fit
            .demand(
                &[demand(vulkan, Contract::DeviceOnly, own_fit + beyond + 1)],
                &[],
                FitMode::Stable,
            )
            .unwrap();
        assert_eq!(
            over.refused,
            [FitRefusal {
                store: host_ram,
                required_bytes: own_fit + beyond + 1,
                missing_bytes: 1,
                constraint: MemoryConstraint::Grant,
            }]
        );

        // Host work leaving 1 GiB of host RAM's fit.
        let crowded = fit
            .demand(
                &[demand(vulkan, Contract::DeviceOnly, own_fit + 2 * GIB)],
                &[(host_ram, host_fit - GIB)],
                FitMode::Stable,
            )
            .unwrap();
        assert_eq!(placed(&crowded, radeon), own_fit);
        assert_eq!(placed(&crowded, host_ram), host_fit);
        assert_eq!(
            crowded.refused,
            [FitRefusal {
                store: host_ram,
                required_bytes: own_fit + 2 * GIB,
                missing_bytes: GIB,
                constraint: MemoryConstraint::HostRam,
            }]
        );
    }

    /// A binding limit is a grant on host RAM that every path into it
    /// shares: its total bounds host RAM's stable fit and its supply the
    /// headroom, for the CPU and for CUDA's pinned upload alike.
    #[test]
    fn a_binding_limit_bounds_host_ram_for_every_path() {
        let topology = recorded::rtx3090();
        let host_ram = topology.host_ram().id();
        let limited = status(limited(32 * GIB, 16 * GIB));
        let fit = Fit::new(&topology, &limited);
        let reading = fit.store(host_ram).unwrap();
        assert_eq!(reading.capacity_bytes, 32 * GIB);
        assert_eq!(reading.headroom_bytes, 16 * GIB);
        let cpu = executor(&topology, BackendName::Cpu);
        let cuda = executor(&topology, BackendName::Cuda);
        for (executor, kind) in [(cpu, Contract::DeviceOnly), (cuda, Contract::Upload)] {
            let room = fit.placement(executor, kind).unwrap();
            assert_eq!((room.stable_bytes, room.ceiling_bytes), (reading.capacity_bytes, reading.headroom_bytes));
        }
    }

    /// Direct host work and every storage kind of an executor on host RAM
    /// spend host RAM's one fit.
    #[test]
    fn storage_kinds_and_direct_work_spend_one_host_budget() {
        let topology = recorded::m4_pro();
        let unopened = status(host(16 * GIB));
        let fit = Fit::new(&topology, &unopened);
        let host_ram = topology.host_ram().id();
        let room = fit.store(host_ram).unwrap().capacity_bytes;
        let cpu = executor(&topology, BackendName::Cpu);
        let result = fit
            .demand(
                &[
                    demand(cpu, Contract::DeviceOnly, room - 20),
                    demand(cpu, Contract::Upload, 20),
                ],
                &[(host_ram, 7)],
                FitMode::Stable,
            )
            .unwrap();
        assert_eq!(result.placed, [(host_ram, room)]);
        assert_eq!(
            result.refused,
            [FitRefusal {
                store: host_ram,
                required_bytes: 20,
                missing_bytes: 7,
                constraint: MemoryConstraint::HostRam,
            }]
        );
    }

    /// Storage demanded of an executor that places none (Metal on a device
    /// whose memory is not the host's) is refused as unplaced, never fitted
    /// as zero.
    #[test]
    fn demand_on_an_unavailable_executor_is_unplaced() {
        let topology = recorded::m4_pro_without_unified_memory();
        let unopened = status(host(16 * GIB));
        let fit = Fit::new(&topology, &unopened);
        let metal = executor(&topology, BackendName::Metal);
        let unplaced = MemoryPolicyError::Unplaced {
            executor: metal.id(),
            kind: Contract::DeviceOnly,
        };
        assert_eq!(
            fit.demand(
                &[demand(metal, Contract::DeviceOnly, 1)],
                &[],
                FitMode::Stable
            ),
            Err(unplaced.clone())
        );
        assert_eq!(fit.placement(metal, Contract::DeviceOnly), Err(unplaced));
    }

    /// Fit answers only for nodes of its own topology revision.
    #[test]
    fn a_subject_of_another_revision_is_refused() {
        let (topology, other) = (recorded::m4_pro(), recorded::m4_pro());
        let unopened = status(host(16 * GIB));
        let fit = Fit::new(&topology, &unopened);
        assert_eq!(
            fit.store(other.host_ram().id()),
            Err(MemoryPolicyError::ForeignIdentifier)
        );
        let cpu = executor(&other, BackendName::Cpu);
        assert_eq!(
            fit.demand(&[demand(cpu, Contract::DeviceOnly, 1)], &[], FitMode::Stable),
            Err(MemoryPolicyError::ForeignIdentifier)
        );
    }
}
