//! Point requests: what a
//! consumer asks the measurement job to measure, keyed by what its points
//! depend on, kept beside the records in the kernel cache, and measured by
//! the job in the order of who waits on them ([`measure`]).
//!
//! - **A request** ([`PointRequest`]) names points by their dependencies
//!   beside the executor. Its points are measured once their records exist;
//!   a request whose points all have records is satisfied, and is forgotten
//!   as it is next read ([`PointRequests::outstanding`]).
//! - **Who waits** ([`Waiter`]): a load serving with a decision pending on a
//!   request touches it when it loads and each time it prices again, so the
//!   load that touched last (the resident one) is measured for first, its
//!   requests in the order it lists them; then assessment's standing
//!   request for the executor's costs ([`PointRequest::Costs`]) where no
//!   load requested them, which is never absent: every estimate waits on
//!   it.
//! - **The job** ([`measure`]) measures the outstanding requests in that
//!   order within its time, each point sampling for an equal part of what
//!   is left, and stops where its time ends: the next job resumes from the
//!   records stored. No load measures, and no load waits on the job.

use super::executor_costs::{executor_identity, retraced, walk_points, ExecutorCostsError, Measurement, Scope, Visit, Walks};
use super::streaming_costs::{CpuCosts, CpuForm, CeilingError, ConvertCosts, ConvertForm, MemoryCosts, MemoryError, MoveCosts, MoveError, MoveForm, TierCopyCosts, TierCopyError, GraphMemory, VolumeCosts};
use super::{Priced, StoredCosts};
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
    /// The executor's whole cost walk (`ExecutorCosts`): assessment's
    /// standing request, every estimate's input, and a streamed load's
    /// request after host storage's routes, as its lines, pass and resident
    /// step read it.
    Costs,
    /// The probe and host storage's routes of the walk: a host tier's link
    /// line (`ExpertPass::lines`).
    HostRoutes,
    /// A bank group's shrink move line at its record form and moves
    /// (`MoveCosts`).
    Moves(MoveForm),
    /// A bank group's convert ceiling at its record form (`ConvertCosts`).
    Convert(ConvertForm),
    /// The host memory's read line with `readers` threads reading at once
    /// (`MemoryCosts`): a host tier's record read from host memory.
    HostMemory { readers: u64 },
    /// The blocks a host tier's copy takes over the link (`TierCopyCosts`).
    TierCopy,
    /// What the executor's recorded graphs hold (`GraphMemory`): a load
    /// prices its graphs at its backend's bound
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
    /// does.
    pub fn outstanding(&self, topology: &DeviceTopology, executor: &Executor, store: &dyn RecordStore<PointKey>) -> bool {
        match self {
            Self::Costs => walk_outstanding(topology, executor, store, Scope::Whole),
            Self::HostRoutes => walk_outstanding(topology, executor, store, Scope::HostRoutes),
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
            Self::Volume { file, urgent } => {
                let urgent = VolumeCosts::urgent(urgent);
                matches!(VolumeCosts::stored(topology, executor, file, urgent.as_ref(), store), Ok(None))
            }
        }
    }

    /// The points the job visits for it.
    fn points(&self, topology: &DeviceTopology, executor: &Executor) -> Result<usize, ExecutorCostsError> {
        match self {
            Self::Costs => walk_points(topology, executor, Scope::Whole),
            Self::HostRoutes => walk_points(topology, executor, Scope::HostRoutes),
            // Two points each: a line's ends, or a site's admission and its
            // pool's lines.
            Self::Moves(_) | Self::Convert(_) | Self::HostMemory { .. } | Self::TierCopy | Self::Site(_) => Ok(2),
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
            Self::Costs => walks.walk(Scope::Whole),
            Self::HostRoutes => walks.walk(Scope::HostRoutes),
            Self::Moves(form) => (form.measure(walks).err(), Vec::new()),
            Self::Convert(form) => (form.measure(walks).err(), Vec::new()),
            Self::HostMemory { readers } => {
                MemoryCosts::measure(*readers, walks);
                (None, Vec::new())
            }
            Self::TierCopy => (TierCopyCosts::measure(walks).err(), Vec::new()),
            Self::GraphCost => (GraphMemory::record(walks.device(), walks.store()).err(), Vec::new()),
            Self::Site(site) => (site.measure(walks).err(), Vec::new()),
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
    /// CPU's architecture and native tier, and the engine build that holds
    /// the site's arithmetic and the reference; the executor's identity
    /// names the machine's job.
    fn key(&self, identity: &Identity) -> PointKey {
        let digest = crate::kernel_cache::TuningCacheKey::of(&format!(
            "site admission\nbuild {}\ncpu {} {:?}\nrows {DECODE_ROWS}\n{self:?}",
            crate::ENGINE_BUILD,
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

/// Whether the walk of `scope` has a point no record holds.
fn walk_outstanding(topology: &DeviceTopology, executor: &Executor, store: &dyn RecordStore<PointKey>, scope: Scope) -> bool {
    retraced(topology, executor, store, scope).is_ok_and(|stored| stored.outstanding())
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
/// on it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Stored {
    executor: Identity,
    request: PointRequest,
    waiter: Waiter,
}

impl seismic::measure::Stored for Stored {
    const SCHEMA: &'static str = "point-request.2";
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
    /// hold, the load that waited latest first, then assessment's standing
    /// request for the costs where its walk is not whole; each request
    /// satisfied is forgotten.
    pub fn outstanding(
        &self,
        topology: &DeviceTopology,
        executor: &Executor,
        store: &dyn RecordStore<PointKey>,
    ) -> Result<Vec<PointRequest>, String> {
        let identity = executor_identity(topology, executor).map_err(|error| error.to_string())?;
        let mut waiting = Vec::new();
        for stored in self.files.values::<Stored>(LARGEST_REQUEST).map_err(|error| error.to_string())? {
            if stored.executor != identity {
                continue;
            }
            match stored.request.outstanding(topology, executor, store) {
                true => waiting.push(stored),
                false => self.files.remove::<Stored>(&(&stored.executor, &stored.request)).map_err(|error| error.to_string())?,
            }
        }
        waiting.sort_by(|first, second| second.waiter.cmp(&first.waiter));
        let mut requests = waiting.into_iter().map(|stored| stored.request).collect::<Vec<_>>();
        if !requests.contains(&PointRequest::Costs) && PointRequest::Costs.outstanding(topology, executor, store) {
            requests.push(PointRequest::Costs);
        }
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

    /// Requests are kept across readers; a later touch moves a request
    /// ahead and an earlier one does not; each executor's are its own.
    #[test]
    fn requests_are_durable_and_the_latest_waiter_first() {
        let root = std::env::temp_dir().join(format!("point-requests-{}", std::process::id()));
        let requests = PointRequests::new(&root);
        let (topology, executor, identity) = host();
        // Another executor's identity: the recorded RTX 3090's CUDA one.
        let recorded = seismic::fixtures::recorded::rtx3090();
        let other = recorded.executors().iter().find(|executor| executor.backend() == seismic::BackendName::Cuda).unwrap();
        let other_identity = recorded.executor_identity(other.id()).unwrap();
        requests.request(&identity, PointRequest::HostRoutes, Waiter { touched: 5, rank: 0 }).unwrap();
        requests.request(&identity, PointRequest::Costs, Waiter { touched: 7, rank: 0 }).unwrap();
        requests.request(&other_identity, PointRequest::HostRoutes, Waiter { touched: 9, rank: 0 }).unwrap();
        let records = seismic::measure::HeldStore::default();
        let order = |requests: &PointRequests| requests.outstanding(&topology, &executor, &records).unwrap();
        assert_eq!(order(&requests), [PointRequest::Costs, PointRequest::HostRoutes]);
        // A later touch moves the host routes ahead; an earlier does not
        // move them back. Another reader sees the same.
        requests.request(&identity, PointRequest::HostRoutes, Waiter { touched: 8, rank: 0 }).unwrap();
        requests.request(&identity, PointRequest::HostRoutes, Waiter { touched: 1, rank: 0 }).unwrap();
        assert_eq!(order(&PointRequests::new(&root)), [PointRequest::HostRoutes, PointRequest::Costs]);
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

    /// Assessment's costs are always waited on: with no load's request, the
    /// costs alone are outstanding while their walk is not whole.
    #[test]
    fn the_costs_are_a_standing_request() {
        let root = std::env::temp_dir().join(format!("point-requests-standing-{}", std::process::id()));
        let (topology, executor, _) = host();
        let records = seismic::measure::HeldStore::default();
        assert_eq!(PointRequests::new(&root).outstanding(&topology, &executor, &records).unwrap(), [PointRequest::Costs]);
    }
}
