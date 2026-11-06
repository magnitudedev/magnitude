//! Point requests: what a
//! consumer asks the measurement job to measure, keyed by what its points
//! depend on, kept beside the records in the kernel cache, and measured by
//! the job in the order of who waits on them ([`measure`]).
//!
//! - **A request** ([`PointRequest`]) names points by their dependencies
//!   beside the executor. Its points are measured once their records exist;
//!   a request whose points all have records is satisfied, and is forgotten
//!   as it is next read ([`PointRequests::outstanding`]).
//! - **Assessment's standing requests** ([`standing`]): the figures every
//!   estimate is made from, fixed per backend and of no model: what its
//!   graphs hold ([`PointRequest::GraphCost`]), the model store's read line ([`PointRequest::Storage`]), the host memory's line
//!   and a CPU site's lines at its reference form ([`PointRequest::CpuSite`])
//!   where a bank can have a host tier ([`PointRequest::HostMemory`]), the
//!   device's convert at its reference form ([`PointRequest::Convert`]), and
//!   the executor's costs ([`PointRequest::Costs`]). They are measured first,
//!   whatever a load asks, so one job's time is allocated across them alone
//!   and no load moves them later or earlier.
//! - **Who waits** ([`Waiter`]): a load serving with a runtime decision
//!   pending on a request (its sites, moves, converts, tier copies and
//!   volumes) touches it when it loads and each time it prices again, so
//!   the load that touched last (the resident one) is measured for next,
//!   its requests in the order it lists them. A load's requests are its
//!   runtime's: none enters assessment, which waits on its standing
//!   requests alone.
//! - **The job** ([`measure`]) measures the outstanding requests in that
//!   order within its time, each point sampling for an equal part of what
//!   is left, and stops where its time ends: the next job resumes from the
//!   records stored. No load measures, and no load waits on the job.

use super::executor_costs::{executor_identity, walk_points, ExecutorCostsError, Measurement, Visit, Walks};
use super::streaming_costs::{CpuCosts, CpuForm, CeilingError, ConvertCosts, ConvertForm, MemoryCosts, MemoryError, MoveCosts, MoveError, MoveForm, StorageCosts, TierCopyCosts, TierCopyError, GraphMemory, StoredVolume, VolumeCosts};
use super::Priced;
use crate::native::SiteForm;
use crate::planning::BankBinding;
use magnitude_kernels::constants::DECODE_ROWS;
use seismic::measure::{Attempt, JsonFileStore, Reason, Record, RecordStore, StoreError, Timing};
use seismic::{Device, DeviceTopology, Executor, Identity, Measurer, PointKey};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// What a consumer can ask the measurement job for, beside the executor
/// every request is keyed by.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PointRequest {
    /// A streamed layer's CPU site (`SiteRequest`): its admission by the
    /// precision system, then its pool and lines (`CpuCosts`). Until both
    /// are stored the site serves nothing.
    Site(SiteRequest),
    /// A CPU site's pool and lines at its form, admitted by nothing
    /// (`CpuCosts`): a standing request at the reference form
    /// (`CpuForm::reference`) where a bank can have a host tier, every
    /// streamed estimate's sited misses.
    CpuSite(CpuForm),
    /// The executor's whole cost walk (`ExecutorCosts`): a standing
    /// request, every estimate's input.
    Costs,
    /// The read line of the volume `directory` lies on (`StorageCosts`): a
    /// standing request, every streamed estimate's file reads.
    Storage { directory: std::path::PathBuf },
    /// A bank group's shrink move line at its record form and moves
    /// (`MoveCosts`).
    Moves(MoveForm),
    /// A bank group's convert ceiling at its record form (`ConvertCosts`); a
    /// standing request at the reference form (`ConvertForm::reference`),
    /// every streamed estimate's landing.
    Convert(ConvertForm),
    /// The host memory's read line with `readers` threads reading at once
    /// (`MemoryCosts`): a host tier's record read from host memory, a
    /// standing request where a bank can have one.
    HostMemory { readers: u64 },
    /// The blocks a host tier's copy takes over the link (`TierCopyCosts`).
    TierCopy,
    /// What the executor's recorded graphs hold (`GraphMemory`): a standing
    /// request, every fit's graphs, priced at its backend's bound
    /// (`seismic::unmeasured_graph_cost`) until it is recorded.
    GraphCost,
    /// How the volume `file` lies on is read, for the urgent request of
    /// `records` records of `parts` where a step waits on it (`VolumeCosts`).
    Volume { file: std::path::PathBuf, urgent: Option<(Vec<u64>, u64)> },
}

impl PointRequest {
    /// Whether a job has anything of the request left to measure on
    /// `executor` beside `store`'s records: a point of it no record holds.
    /// A request whose records cannot be derived from otherwise (a walk's
    /// probe recorded unobtainable, a point recorded so, a store that cannot
    /// be read) has none: the job's measuring ends there as the derivation
    /// does. The costs' is `costs_outstanding`, whether the walk of the
    /// executor's costs ([`super::ExecutorCosts::stored`]), which its caller
    /// walks once for whatever else it reads of them, has a point no record
    /// holds.
    pub fn outstanding(&self, topology: &DeviceTopology, executor: &Executor, store: &dyn RecordStore<PointKey>, costs_outstanding: bool) -> bool {
        match self {
            Self::Costs => costs_outstanding,
            Self::Storage { directory } => StorageCosts::outstanding(topology, executor, directory, store),
            Self::Moves(form) => {
                matches!(MoveCosts::stored(topology, executor, *form, store), Err(MoveError::Unrecorded { .. }))
            }
            Self::Convert(form) => {
                matches!(ConvertCosts::stored(topology, executor, *form, store), Err(CeilingError::Unrecorded { .. }))
            }
            Self::HostMemory { readers } => {
                matches!(MemoryCosts::stored(topology, executor, *readers, store), Err(MemoryError::Unrecorded { .. }))
            }
            Self::TierCopy => matches!(TierCopyCosts::stored(topology, executor, store), Err(TierCopyError::Unrecorded { .. })),
            Self::GraphCost => matches!(GraphMemory::stored(topology, executor, store), Ok(None)),
            Self::Site(site) => site.outstanding(topology, executor, store),
            Self::CpuSite(form) => matches!(CpuCosts::stored(topology, executor, form, store), Ok(None)),
            Self::Volume { file, urgent } => {
                let urgent = VolumeCosts::urgent(urgent);
                matches!(VolumeCosts::stored(topology, executor, file, urgent.as_ref(), store), Ok(None | Some(StoredVolume::Earlier(_))))
            }
        }
    }

    /// The points the job visits for it.
    fn points(&self, topology: &DeviceTopology, executor: &Executor) -> Result<usize, ExecutorCostsError> {
        match self {
            Self::Costs => walk_points(topology, executor),
            // Two points each: a line's ends, or a site's admission and its
            // pool's lines.
            Self::Storage { .. } | Self::Moves(_) | Self::Convert(_) | Self::HostMemory { .. } | Self::TierCopy | Self::Site(_) => Ok(2),
            // Its pool's lines.
            Self::CpuSite(_) => Ok(1),
            // Measured as the job's device opened: nothing to time.
            Self::GraphCost => Ok(0),
            // One probe of the volume, its own points its own.
            Self::Volume { .. } => Ok(1),
        }
    }

    /// Measure it in the job (`walks`): why its measuring ended before its
    /// end where no visited point's outcome says, and why a figure could
    /// not be read outside its points' timings.
    fn measure(&self, walks: &mut Walks<'_, '_>) -> (Option<ExecutorCostsError>, Vec<ExecutorCostsError>) {
        match self {
            Self::Costs => walks.walk(),
            Self::Storage { directory } => (StorageCosts::measure(directory, walks).err().map(|error| ExecutorCostsError(error.to_string())), Vec::new()),
            Self::Moves(form) => (form.measure(walks).err(), Vec::new()),
            Self::Convert(form) => (form.measure(walks).err(), Vec::new()),
            Self::HostMemory { readers } => {
                MemoryCosts::measure(*readers, walks);
                (None, Vec::new())
            }
            Self::TierCopy => (TierCopyCosts::measure(walks).err(), Vec::new()),
            Self::GraphCost => (GraphMemory::record(walks.device(), walks.store()).err(), Vec::new()),
            Self::Site(site) => (site.measure(walks).err(), Vec::new()),
            Self::CpuSite(form) => (CpuCosts::measure(walks.device(), form, walks.store()).err(), Vec::new()),
            Self::Volume { file, urgent } => {
                let urgent = VolumeCosts::urgent(urgent);
                (VolumeCosts::measure(walks.device(), file, urgent.as_ref(), walks.store()).err(), Vec::new())
            }
        }
    }
}

/// A streamed layer's CPU site as the job admits and prices it: its form
/// (the records it reads and its extents, `CpuForm`), its tier's records,
/// the element its arithmetic is held to and the load's numerical
/// requirement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteRequest {
    pub(crate) form: CpuForm,
    pub(crate) tier: u64,
    pub(crate) activation: seismic::Element,
    pub(crate) requirement: seismic::NumericalRequirement,
}

impl SiteRequest {
    /// The request of `binding`'s site of form `form` under `requirement`.
    pub(crate) fn of(binding: &BankBinding, form: SiteForm, requirement: seismic::NumericalRequirement) -> Self {
        Self { form: CpuForm::of(binding, form.function), tier: binding.tier, activation: form.activation, requirement }
    }

    /// The precision system's verdict is a function of the request, the
    /// CPU's architecture and native tier, and the executor build that holds
    /// the site's arithmetic and the reference; the executor's identity
    /// names the machine's job.
    fn key(&self, identity: &Identity) -> PointKey {
        let digest = crate::kernel_cache::TuningCacheKey::of(&format!(
            "site admission\nbuild {}\ncpu {} {:?}\nrows {DECODE_ROWS}\n{self:?}",
            crate::EXECUTOR_BUILD,
            std::env::consts::ARCH,
            seismic::cpu::Tier::detected(),
        ));
        let term = |name: &str| std::collections::BTreeMap::from([(name.to_owned(), 1)]);
        PointKey { executor: identity.clone(), kernels: Vec::new(), configuration: term("site_admission"), point: term(digest.as_str()) }
    }

    /// The verdict `store` records on `executor`: admitted, pending until
    /// the job has run the site, or unavailable with its refusal.
    pub fn verdict(&self, topology: &DeviceTopology, executor: &Executor, store: &dyn RecordStore<PointKey>) -> Priced<()> {
        let identity = match executor_identity(topology, executor) {
            Ok(identity) => identity,
            Err(error) => return Priced::Unavailable(error.to_string()),
        };
        match store.load(&self.key(&identity)) {
            Ok(Some(Record::Measured(_))) => Priced::Measured(()),
            Ok(Some(Record::Unobtainable { reason })) => Priced::Unavailable(reason.to_string()),
            Ok(None) => Priced::Pending("its CPU site's verdict".into()),
            Err(error) => Priced::Unavailable(error.to_string()),
        }
    }

    /// What `store` records of the site's pool and lines on `executor`;
    /// none until the job measured them.
    pub fn costs(
        &self,
        topology: &DeviceTopology,
        executor: &Executor,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Option<CpuCosts>, ExecutorCostsError> {
        CpuCosts::stored(topology, executor, &self.form, store)
    }

    /// Whether the job has anything of the site left to run: its verdict,
    /// or an admitted site's lines. A refused site, or a store that cannot
    /// be read, has none.
    fn outstanding(&self, topology: &DeviceTopology, executor: &Executor, store: &dyn RecordStore<PointKey>) -> bool {
        match self.verdict(topology, executor, store) {
            Priced::Pending(_) => true,
            Priced::Measured(()) => matches!(self.costs(topology, executor, store), Ok(None)),
            Priced::Unavailable(_) => false,
        }
    }

    /// Run the site in the job: the verdict is its record, an admission as
    /// the seconds it took, a refusal unobtainable with why; then an admitted
    /// site's pool and lines where none are stored. A pool that fails its
    /// request is no verdict: the job's two-job rule ends it.
    fn measure(&self, walks: &mut Walks<'_, '_>) -> Result<(), ExecutorCostsError> {
        let key = self.key(&walks.identity());
        walks.record(key, |device| {
            let began = std::time::Instant::now();
            Ok(match crate::native::admit_site(device, self)? {
                Ok(()) => Record::Measured(Timing { median: began.elapsed().as_secs_f64(), interval: None, samples: 1 }),
                Err(refusal) => Record::Unobtainable { reason: Reason::new(refusal) },
            })
        });
        let (device, store) = (walks.device(), walks.store());
        let (topology, executor) = (device.topology(), device.info());
        match self.verdict(topology, executor, store).measured().is_some() && self.outstanding(topology, executor, store) {
            true => CpuCosts::measure(device, &self.form, store),
            false => Ok(()),
        }
    }
}

/// Whether the walk has a point no record holds.
#[cfg(test)]
fn walk_outstanding(topology: &DeviceTopology, executor: &Executor, store: &dyn RecordStore<PointKey>) -> bool {
    super::executor_costs::retraced(topology, executor, store).is_ok_and(|stored| stored.outstanding())
}

/// Assessment's standing requests on `executor`, in the order the job
/// measures them: what its graphs hold, which its device read as it opened;
/// the read line of the volume the model store `storage`
/// lies on, the host memory's line with every CPU reading and a CPU site's
/// lines at its reference form where a bank can have a host tier (a
/// discrete device), the device's convert at its reference form, and the
/// executor's costs. Each is
/// of the machine alone, so what is measured is fixed per backend.
pub fn standing(topology: &DeviceTopology, executor: &Executor, storage: &Path) -> Result<Vec<PointRequest>, String> {
    let tiered = !topology.shared_ranges_lie_in_device_memory(executor);
    let memory = tiered
        .then(|| std::thread::available_parallelism().map(|readers| PointRequest::HostMemory { readers: readers.get() as u64 }))
        .transpose()
        .map_err(|error| format!("the host's processors: {error}"))?;
    let site = match tiered {
        true => CpuForm::reference(executor.backend()).map_err(|error| error.to_string())?,
        false => None,
    };
    let convert = ConvertForm::reference(executor.backend()).map_err(|error| error.to_string())?;
    Ok([PointRequest::GraphCost, PointRequest::Storage { directory: storage.to_owned() }]
        .into_iter()
        .chain(memory)
        .chain(site.map(PointRequest::CpuSite))
        .chain(convert.map(PointRequest::Convert))
        .chain([PointRequest::Costs])
        .collect())
}

/// Who waits on a stored request: a load, as of the latest time it touched
/// the request (nanoseconds since the Unix epoch), and the request's place
/// among those it touched together (`rank`, 0 first): the job measures the
/// latest waiter first, and of one touch, in its order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Waiter {
    pub touched: u64,
    pub rank: u64,
}

impl Waiter {
    /// A load waiting now, on its request of place `rank` in its touch.
    pub fn now(rank: u64) -> Self {
        let since = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).expect("the clock is past the Unix epoch");
        Self { touched: u64::try_from(since.as_nanos()).expect("nanoseconds since the Unix epoch fit in u64"), rank }
    }
}

/// A waiter is ahead of (greater than) another touched earlier, or touched
/// at once at a later place.
impl Ord for Waiter {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.touched.cmp(&other.touched).then(other.rank.cmp(&self.rank))
    }
}

impl PartialOrd for Waiter {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A request as it is stored: its executor, what it asks for and who waits
/// on it. Its form is its build's own: a load asks its own build's job, so
/// the schema is the executor build, and a request another build stored in a
/// shared kernel cache lies apart, never read here.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Stored {
    executor: Identity,
    request: PointRequest,
    waiter: Waiter,
}

impl seismic::measure::Stored for Stored {
    const SCHEMA: &'static str = concat!("point-request.", env!("MAGNITUDE_EXECUTOR_BUILD"));
}

/// The most bytes a stored request is written as: an executor's identity
/// and a request are a few hundred.
const LARGEST_REQUEST: u64 = 64 << 10;

/// The point requests kept in a kernel cache
/// ([`crate::KernelCache::measurement_requests`]).
pub struct PointRequests {
    files: JsonFileStore,
}

impl PointRequests {
    /// The requests kept in the kernel cache at `root`.
    pub fn new(root: &Path) -> Self {
        Self { files: crate::KernelCache::measurement_requests(root) }
    }

    /// Ask for `request`'s points on `executor`, waited on by `waiter`: a
    /// later touch of a stored request moves it ahead, an earlier one leaves
    /// it as it is.
    pub fn request(&self, executor: &Identity, request: PointRequest, waiter: Waiter) -> Result<(), StoreError> {
        let key = (executor, &request);
        let stored = Stored { executor: executor.clone(), request: request.clone(), waiter };
        self.files
            .update(&key, LARGEST_REQUEST, |held: Option<Stored>| {
                Ok::<_, std::convert::Infallible>(match held {
                    Some(held) if held.waiter >= waiter => held,
                    _ => stored,
                })
            })
            .map(|updated| match updated {
                Ok(()) => (),
                Err(never) => match never {},
            })
    }

    /// The requests on `executor` whose points `store`'s records do not all
    /// hold: assessment's standing requests of the model store `storage`
    /// ([`standing`]) first, then the loads', the load that waited latest
    /// first; each load's request satisfied is forgotten. The costs' is
    /// `costs_outstanding`, as [`PointRequest::outstanding`] takes it.
    pub fn outstanding(
        &self,
        topology: &DeviceTopology,
        executor: &Executor,
        storage: &Path,
        store: &dyn RecordStore<PointKey>,
        costs_outstanding: bool,
    ) -> Result<Vec<PointRequest>, String> {
        let identity = executor_identity(topology, executor).map_err(|error| error.to_string())?;
        let mut requests = standing(topology, executor, storage)?;
        requests.retain(|request| request.outstanding(topology, executor, store, costs_outstanding));
        let mut waiting = Vec::new();
        for stored in self.files.values::<Stored>(LARGEST_REQUEST).map_err(|error| error.to_string())? {
            if stored.executor != identity {
                continue;
            }
            match stored.request.outstanding(topology, executor, store, costs_outstanding) {
                true => waiting.push(stored),
                false => self.files.remove::<Stored>(&(&stored.executor, &stored.request)).map_err(|error| error.to_string())?,
            }
        }
        waiting.sort_by(|first, second| second.waiter.cmp(&first.waiter));
        let loads = waiting.into_iter().map(|stored| stored.request).filter(|request| !requests.contains(request)).collect::<Vec<_>>();
        requests.extend(loads);
        Ok(requests)
    }
}

/// Measure `requests` (in the order [`PointRequests::outstanding`] gives
/// them) on `device`, each point stored in `store` as it is taken, within
/// `deadline`: each point loads from the records or is reported to
/// `attempting`, then measured for an equal part of the job's time left
/// (or until its median is resolved); `visit` is told what became of each
/// point as soon as it is known. What the deadline leaves unread the next
/// job measures. Fails only when `device`'s executor is not of its own
/// topology's revision.
pub fn measure(
    measurer: &mut Measurer,
    device: &Device,
    store: &dyn RecordStore<PointKey>,
    requests: &[PointRequest],
    deadline: std::time::Instant,
    attempting: &mut dyn FnMut(&PointKey) -> Attempt,
    visit: &mut Visit<'_>,
) -> Result<Measurement, ExecutorCostsError> {
    let (topology, executor) = (device.topology(), device.info());
    let mut ended = Vec::new();
    let mut left = 0;
    for request in requests {
        match request.points(topology, executor) {
            Ok(points) => left += points,
            Err(error) => ended.push(error),
        }
    }
    let mut walks = Walks::new(measurer, device, store, deadline, left, attempting, visit)?;
    let mut failures = Vec::new();
    for request in requests {
        let (end, failed) = request.measure(&mut walks);
        ended.extend(end);
        failures.extend(failed);
    }
    Ok(Measurement { ended, failures, held: walks.held() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host's CPU executor, its topology and identity: one every host
    /// derives kernel identities of.
    fn host() -> (std::sync::Arc<DeviceTopology>, Executor, Identity) {
        let topology = seismic::Devices::discover(&[]).unwrap().topology();
        let executor = topology.executors().iter().find(|executor| executor.backend() == seismic::BackendName::Cpu).unwrap().clone();
        let identity = topology.executor_identity(executor.id()).unwrap();
        (topology, executor, identity)
    }

    /// Requests are kept across readers, after the standing requests; a
    /// later touch moves a load's request ahead and an earlier one does not;
    /// each executor's are its own.
    #[test]
    fn requests_are_durable_and_the_latest_waiter_first() {
        let root = std::env::temp_dir().join(format!("point-requests-{}", std::process::id()));
        let storage = std::env::temp_dir();
        let requests = PointRequests::new(&root);
        let (topology, executor, identity) = host();
        // Another executor's identity: the recorded RTX 3090's CUDA one.
        let recorded = seismic::fixtures::recorded::rtx3090();
        let other = recorded.executors().iter().find(|executor| executor.backend() == seismic::BackendName::Cuda).unwrap();
        let other_identity = recorded.executor_identity(other.id()).unwrap();
        let (tier, moves) = (PointRequest::TierCopy, PointRequest::HostMemory { readers: 1 });
        requests.request(&identity, tier.clone(), Waiter { touched: 5, rank: 0 }).unwrap();
        requests.request(&identity, moves.clone(), Waiter { touched: 7, rank: 0 }).unwrap();
        requests.request(&other_identity, tier.clone(), Waiter { touched: 9, rank: 0 }).unwrap();
        let records = seismic::measure::HeldStore::default();
        let costs = walk_outstanding(&topology, &executor, &records);
        let standing = standing(&topology, &executor, &storage).unwrap();
        let order = |requests: &PointRequests| requests.outstanding(&topology, &executor, &storage, &records, costs).unwrap();
        // What the machine measures of them: a unified device has no tier copy.
        let measured = |listed: Vec<PointRequest>| listed.into_iter().filter(|request| request.outstanding(&topology, &executor, &records, costs)).collect::<Vec<_>>();
        assert_eq!(order(&requests), measured([standing.clone(), vec![moves.clone(), tier.clone()]].concat()));
        // A later touch moves the tier copy ahead; an earlier does not move
        // it back. Another reader sees the same.
        requests.request(&identity, tier.clone(), Waiter { touched: 8, rank: 0 }).unwrap();
        requests.request(&identity, tier.clone(), Waiter { touched: 1, rank: 0 }).unwrap();
        assert_eq!(order(&PointRequests::new(&root)), measured([standing, vec![tier, moves]].concat()));
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// A job measures its requests in their order within its deadline,
    /// and the next job resumes from the records: on the host's CPU, two
    /// host memory lines are measured in the order requested, a job of no
    /// time measures nothing, and points stored are loaded, never timed
    /// again.
    #[test]
    fn a_job_measures_in_order_within_its_time_and_the_next_resumes() {
        let device = seismic::Devices::discover(&[]).unwrap().open_backend(seismic::BackendName::Cpu).unwrap();
        let root = std::env::temp_dir().join(format!("point-requests-job-{}", std::process::id()));
        let store = seismic::measure::JsonFileStore::new(&root);
        let requests = [PointRequest::HostMemory { readers: 2 }, PointRequest::HostMemory { readers: 1 }];
        let job = |seconds: f64| {
            let mut measurer = Measurer::new(crate::SEARCH_SETTINGS.error_rate, crate::MIN_SAMPLE_SECONDS);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(seconds);
            let mut visited = Vec::new();
            measure(&mut measurer, &device, &store, &requests, deadline, &mut |_| Attempt::Proceed, &mut |key, outcome| {
                let readers = key.configuration["readers"];
                visited.push((readers, outcome));
            })
            .unwrap();
            visited
        };
        use seismic::measure::Outcome;
        let none = job(0.);
        assert!(none.iter().all(|(_, outcome)| matches!(outcome, Outcome::Expired)), "{none:?}");
        let first = job(30.);
        assert_eq!(first.iter().map(|(readers, _)| *readers).collect::<Vec<_>>(), [2, 2, 1, 1], "in the order requested");
        assert!(first.iter().all(|(_, outcome)| matches!(outcome, Outcome::Measured(_))), "{first:?}");
        let second = job(30.);
        assert!(second.iter().all(|(_, outcome)| matches!(outcome, Outcome::Loaded(_))), "{second:?}");
        let topology = device.topology();
        let records = |readers| MemoryCosts::stored(topology, device.info(), readers, &store);
        assert!(records(1).is_ok() && records(2).is_ok());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Assessment's standing requests are always waited on, and are of the
    /// machine alone: with no load's request, the model store's read line,
    /// then the host memory's line and the reference CPU site where a bank
    /// can have a host tier, the reference convert where a device streams,
    /// then the costs, while none is measured.
    #[test]
    fn the_standing_requests_are_the_machines() {
        let root = std::env::temp_dir().join(format!("point-requests-standing-{}", std::process::id()));
        let (topology, executor, _) = host();
        let storage = std::env::temp_dir();
        let records = seismic::measure::HeldStore::default();
        let costs = walk_outstanding(&topology, &executor, &records);
        let outstanding = PointRequests::new(&root).outstanding(&topology, &executor, &storage, &records, costs).unwrap();
        assert_eq!(outstanding.first(), Some(&PointRequest::Storage { directory: storage.clone() }));
        assert_eq!(outstanding.last(), Some(&PointRequest::Costs));
        let measured = standing(&topology, &executor, &storage).unwrap().into_iter().filter(|request| request.outstanding(&topology, &executor, &records, costs)).collect::<Vec<_>>();
        assert_eq!(outstanding, measured);
        // A device that streams measures every streamed estimate's landing:
        // on any recorded machine it stands, and on this host's own executors
        // (only a host's toolchain derives their kernels' identities) it is
        // outstanding until measured.
        let recorded = seismic::fixtures::recorded::rtx3090();
        let cuda = recorded.executors().iter().find(|executor| executor.backend() == seismic::BackendName::Cuda).unwrap();
        let convert = PointRequest::Convert(ConvertForm::reference(cuda.backend()).unwrap().unwrap());
        assert!(standing(&recorded, cuda, &storage).unwrap().contains(&convert));
        // A device that can hold a host tier also measures every streamed
        // estimate's sited misses, at the reference CPU site.
        let site = PointRequest::CpuSite(CpuForm::reference(cuda.backend()).unwrap().unwrap());
        assert!(standing(&recorded, cuda, &storage).unwrap().contains(&site));
        for executor in topology.executors() {
            if let Some(form) = ConvertForm::reference(executor.backend()).unwrap() {
                let convert = PointRequest::Convert(form);
                assert!(standing(&topology, executor, &storage).unwrap().contains(&convert));
                assert!(convert.outstanding(&topology, executor, &records, walk_outstanding(&topology, executor, &records)), "{:?}", executor.backend());
            }
        }
    }
}
