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
//!   graphs hold ([`PointRequest::GraphCost`]), the executor's costs
//!   ([`PointRequest::Costs`]), whose required figures every load waits on,
//!   then the model store's read line ([`PointRequest::Storage`]), the
//!   device's convert at its reference form ([`PointRequest::Convert`]), and
//!   the host memory's line ([`PointRequest::HostMemory`]) and a CPU site's
//!   lines at its reference form ([`PointRequest::CpuSite`]) where a bank can
//!   have a host tier, which a streamed load's bank is priced by, then the
//!   rest of the walk. They are measured first, whatever a load asks: their
//!   samples share [`MEASUREMENT_SECONDS`](super::executor_costs::MEASUREMENT_SECONDS) of sampling among them alone
//!   ([`JobDeadlines`]), so no load moves them later or earlier, and
//!   a point of them the job's time does not reach is left unrecorded for
//!   the next job, never priced: what is recorded of a figure never follows
//!   from how busy the host was.
//! - **The machine's line** (`line_outstanding`, with the `machine-line`
//!   feature): what the hardware itself reaches, which floors are judged
//!   against, is a development measurement. Nothing in loading, planning or
//!   serving reads it, so no standing request asks for it and no job of the
//!   runtime measures it: a tool asks for it, in a job of its own.
//! - **Who waits** ([`Waiter`]): a load serving with a runtime decision
//!   pending on a request (its sites, moves, converts, tier copies and
//!   volumes) touches it when it loads and each time it prices again, so
//!   the load that touched last (the resident one) is measured for next,
//!   its requests in the order it lists them. A load's requests are its
//!   runtime's: none enters assessment, which waits on its standing
//!   requests alone.
//! - **The job** ([`measure`]) measures the outstanding requests in that
//!   order: assessment's standing points, each sampling for its part of
//!   what is left of [`MEASUREMENT_SECONDS`](super::executor_costs::MEASUREMENT_SECONDS) beside the fewest passes
//!   forecast of those after it; then the loads', each sampling for an
//!   equal part of what is left of the job's time. It begins points until
//!   its time ends, each begun timed to its end, and the next job resumes
//!   the rest from the records stored. No load measures, and no load waits
//!   on the job.

use super::executor_costs::{executor_identity, walk_points, CostWalk, ExecutorCostsError, Measurement, ProgramSet, Visit, Walks, Whose};
use super::streaming_costs::{CpuCosts, CpuForm, ConvertCosts, ConvertForm, MemoryCosts, StorageCosts, TierCopyCosts, GraphMemory, StoredVolume, VolumeCosts};
use super::Priced;
use crate::native::SiteForm;
use crate::planning::BankBinding;
use magnitude_kernels::constants::VERIFY_ROWS;
use seismic::measure::{Attempt, JsonFileStore, Reason, Record, RecordStore, StoreError, Timing};
use seismic::{DeviceTopology, Executor, Identity, Measurer, PointKey};
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
    /// The executor's cost walk (`ExecutorCosts`), to its required figures
    /// or whole ([`CostWalk`]): standing requests, every estimate's input.
    Costs(CostWalk),
    /// The read line of the volume `directory` lies on (`StorageCosts`): a
    /// standing request, every streamed estimate's file reads.
    Storage { directory: std::path::PathBuf },
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
    /// The machine's line (`MachineLine`): device memory's stream rate and
    /// each operand's arithmetic peak, which floors are judged against and
    /// nothing prices. A tool asks for it (`line_outstanding`); no standing
    /// request does.
    #[cfg(feature = "machine-line")]
    Line,
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
    /// does. The costs' follows `costs_outstanding`, the first part of the
    /// walk of the executor's costs ([`super::ExecutorCosts::stored`]) with a
    /// point no record holds ([`super::StoredCosts::outstanding`]), which its
    /// caller walks once for whatever else it reads of them.
    pub fn outstanding(&self, topology: &DeviceTopology, executor: &Executor, store: &dyn RecordStore<PointKey>, costs_outstanding: Option<CostWalk>) -> bool {
        match self {
            Self::Costs(part) => costs_outstanding.is_some_and(|first| first <= *part),
            Self::Storage { directory } => matches!(StorageCosts::priced(topology, executor, directory, store), Priced::Pending(_)),
            Self::Convert(form) => matches!(ConvertCosts::priced(topology, executor, *form, store), Priced::Pending(_)),
            Self::HostMemory { readers } => matches!(MemoryCosts::priced(topology, executor, *readers, store), Priced::Pending(_)),
            Self::TierCopy => matches!(TierCopyCosts::priced(topology, executor, store), Priced::Pending(_)),
            Self::GraphCost => matches!(GraphMemory::stored(topology, executor, store), Ok(None)),
            #[cfg(feature = "machine-line")]
            Self::Line => {
                use super::line::MachineLineError::{Unrecorded, UnrecordedSweep};
                matches!(
                    super::line::MachineLine::stored(topology, executor, store),
                    Err(Unrecorded { .. } | UnrecordedSweep { .. })
                )
            }
            Self::Site(site) => site.outstanding(topology, executor, store),
            Self::CpuSite(form) => matches!(CpuCosts::priced(topology, executor, form, store), Priced::Pending(_)),
            Self::Volume { file, urgent } => {
                let urgent = VolumeCosts::urgent(urgent);
                matches!(VolumeCosts::stored(topology, executor, file, urgent.as_ref(), store), Ok(None | Some(StoredVolume::Earlier(_))))
            }
        }
    }

    /// The points the job visits for it.
    fn points(&self, topology: &DeviceTopology, executor: &Executor) -> Result<usize, ExecutorCostsError> {
        match self {
            Self::Costs(part) => walk_points(topology, executor, *part),
            // Two points each: a line's ends, or a site's admission and its
            // pool's lines.
            Self::Storage { .. } | Self::Convert(_) | Self::HostMemory { .. } | Self::TierCopy | Self::Site(_) => Ok(2),
            // Its pool's lines.
            Self::CpuSite(_) => Ok(1),
            // Measured as the job's device opened: nothing to time.
            Self::GraphCost => Ok(0),
            // Every configuration of every probe.
            #[cfg(feature = "machine-line")]
            Self::Line => super::line::MachineLine::points(topology, executor),
            // One probe of the volume, its own points its own.
            Self::Volume { .. } => Ok(1),
        }
    }

    /// Peak memory demand across `requests` run sequentially in-process via
    /// `measure_in_process` on `device`: the elementwise maximum of each
    /// request's demand, rounded to `device`'s native allocation granularity.
    pub fn measure_demand(device: &seismic::Device, requests: &[Self]) -> Result<crate::memory::ClaimDemand, ExecutorCostsError> {
        let granularity = device.native_allocations().granularity.max(1);
        let mut peak = crate::memory::ClaimDemand::default();
        for request in requests {
            let demand = match request {
                Self::TierCopy => TierCopyCosts::measure_demand(granularity),
                Self::Convert(form) => form.measure_demand(granularity)?,
                _ => crate::memory::ClaimDemand::default(),
            };
            peak.device = peak.device.max(demand.device);
            peak.upload = peak.upload.max(demand.upload);
            peak.shared = peak.shared.max(demand.shared);
            peak.host = peak.host.max(demand.host);
            peak.staged = peak.staged.max(demand.staged);
        }
        Ok(peak)
    }

    /// Measure it in the job (`walks`): why its measuring ended before its
    /// end where no visited point's outcome says, and why a figure could
    /// not be read outside its points' timings.
    fn measure(&self, walks: &mut Walks<'_, '_>) -> (Option<ExecutorCostsError>, Vec<ExecutorCostsError>) {
        match self {
            Self::Costs(part) => walks.walk(*part),
            Self::Storage { directory } => (StorageCosts::measure(directory, walks).err().map(|error| ExecutorCostsError(error.to_string())), Vec::new()),
            Self::Convert(form) => (form.measure(walks).err(), Vec::new()),
            Self::HostMemory { readers } => {
                MemoryCosts::measure(*readers, walks);
                (None, Vec::new())
            }
            Self::TierCopy => (TierCopyCosts::measure(walks).err(), Vec::new()),
            Self::GraphCost => (GraphMemory::record(walks.device(), walks.store()).err(), Vec::new()),
            #[cfg(feature = "machine-line")]
            Self::Line => (super::line::MachineLine::measure(walks).err(), Vec::new()),
            Self::Site(site) => (site.measure(walks).err(), Vec::new()),
            Self::CpuSite(form) => (CpuCosts::measure(walks.charges(), form, walks.store()).err(), Vec::new()),
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
        Self { form: CpuForm::of(binding, form), tier: binding.tier, activation: form.activation, requirement }
    }

    /// Digest of the CPU site arithmetic, its reference math, and the compiled
    /// CPU kernel library — excluding unrelated GPU kernels, planner, and
    /// executor changes so CPU site admission is not invalidated by them.
    fn site_admission_build() -> &'static str {
        static BUILD: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        BUILD.get_or_init(|| {
            crate::kernel_cache::TuningCacheKey::of(&format!(
                "cpu-site-admission\ncpu-kernels {}\nsite {}\nqualification {}",
                magnitude_kernels::routed_expand::cpu_native::KERNELS.digest,
                include_str!("../experts/site.rs"),
                include_str!("../native/qualification/site.rs"),
            ))
            .as_str()
            .to_owned()
        })
    }

    /// The precision system's verdict is a function of the request, the
    /// CPU's architecture and native tier, and the CPU site/reference math;
    /// keyed by the host hardware identity rather than any GPU executor.
    fn key(&self, topology: &DeviceTopology) -> PointKey {
        let digest = crate::kernel_cache::TuningCacheKey::of(&format!(
            "site admission\nsite_build {}\ncpu {} {:?}\nrows {VERIFY_ROWS}\n{self:?}",
            Self::site_admission_build(),
            std::env::consts::ARCH,
            seismic::cpu::Tier::detected(),
        ));
        let term = |name: &str| std::collections::BTreeMap::from([(name.to_owned(), 1)]);
        PointKey {
            subject: seismic::PointSubject::host(topology),
            kernels: Vec::new(),
            configuration: term("site_admission"),
            point: term(digest.as_str()),
        }
    }

    /// The verdict `store` records on `topology`'s host: admitted, pending
    /// until the job has run the site, or unavailable with its refusal.
    pub fn verdict(&self, topology: &DeviceTopology, _executor: &Executor, store: &dyn RecordStore<PointKey>) -> Priced<()> {
        match store.load(&self.key(topology)) {
            Ok(Some(Record::Measured(_))) => Priced::Measured(()),
            Ok(Some(Record::Unobtainable { reason })) => Priced::Unavailable(reason.to_string()),
            Ok(None) => Priced::Pending(PendingPrices::load(PointRequest::Site(self.clone()))),
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
        let key = self.key(walks.device().topology());
        walks.record(key, |charges| {
            let began = std::time::Instant::now();
            Ok(match crate::native::admit_site(charges, self)? {
                Ok(()) => Record::Measured(Timing { median: began.elapsed().as_secs_f64(), interval: None, samples: 1 }),
                Err(refusal) => Record::Unobtainable { reason: Reason::new(refusal) },
            })
        });
        let (device, store) = (walks.device(), walks.store());
        let (topology, executor) = (device.topology(), device.info());
        match self.verdict(topology, executor, store).measured().is_some() && self.outstanding(topology, executor, store) {
            true => CpuCosts::measure(walks.charges(), &self.form, store),
            false => Ok(()),
        }
    }
}

/// The first part of the walk with a point no record holds.
#[cfg(test)]
fn walk_outstanding(topology: &DeviceTopology, executor: &Executor, store: &dyn RecordStore<PointKey>) -> Option<CostWalk> {
    super::executor_costs::retraced(topology, executor, store).ok().and_then(|stored| stored.outstanding())
}

/// Assessment's standing requests on `executor`, in the order the job
/// measures them: what its graphs hold, which its device read as it opened;
/// the cost walk's required figures, which every load plans and serves on
/// and waits for once per machine, so nothing is measured ahead of them;
/// then what a streamed load's bank is priced by: the read line of the
/// volume the model store `storage` lies on, the device's convert at its
/// reference form, and the host memory's line with every CPU reading and a
/// CPU site's lines at its reference form where a bank can have a host tier
/// (a discrete device); then the rest of the walk, whose figures are priced
/// closer without it. Each is of the machine alone, so what is measured is
/// fixed per backend, and their samples share [`MEASUREMENT_SECONDS`](super::executor_costs::MEASUREMENT_SECONDS) of
/// sampling ([`JobDeadlines`]).
pub fn standing(topology: &DeviceTopology, executor: &Executor, storage: &Path) -> Result<Vec<PointRequest>, String> {
    let tiered = !topology.shared_ranges_lie_in_device_memory(executor);
    let memory = tiered.then(|| PointRequest::HostMemory { readers: seismic::threads::quota().get() as u64 });
    let site = match tiered {
        true => CpuForm::reference(executor.backend()).map_err(|error| error.to_string())?,
        false => None,
    };
    let convert = ConvertForm::reference(executor.backend()).map_err(|error| error.to_string())?;
    Ok([PointRequest::GraphCost, PointRequest::Costs(CostWalk::Required), PointRequest::Storage { directory: storage.to_owned() }]
        .into_iter()
        .chain(convert.map(PointRequest::Convert))
        .chain(memory)
        .chain(site.map(PointRequest::CpuSite))
        .chain([PointRequest::Costs(CostWalk::Whole)])
        .collect())
}

/// What a tool that needs the machine's line alone has the job measure: the
/// line, where a point of it no record holds, and no other request (not the
/// costs, not a load's). The job takes it as the loads' requests, with no
/// standing one: each point an equal part of the job's time, one the time
/// does not reach left unrecorded for the next job.
#[cfg(feature = "machine-line")]
pub fn line_outstanding(
    topology: &DeviceTopology,
    executor: &Executor,
    store: &dyn RecordStore<PointKey>,
    costs_outstanding: Option<CostWalk>,
) -> Outstanding {
    let line = PointRequest::Line;
    let loads = if line.outstanding(topology, executor, store, costs_outstanding) { vec![line] } else { Vec::new() };
    Outstanding { standing: Vec::new(), loads }
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
        // A request held by a waiter as late or later is left as it is.
        self.files
            .update(&key, LARGEST_REQUEST, |held: Option<Stored>| match held {
                Some(held) if held.waiter >= waiter => Err(()),
                _ => Ok(Some(stored)),
            })
            .map(|_written_or_left| ())
    }

    /// The requests on `executor` whose points `store`'s records do not all
    /// hold: assessment's standing requests of the model store `storage`
    /// ([`standing`]), and the loads', the load that waited latest first,
    /// none of them a standing one; each load's request satisfied is
    /// forgotten. The costs' is `costs_outstanding`, as
    /// [`PointRequest::outstanding`] takes it.
    pub fn outstanding(
        &self,
        topology: &DeviceTopology,
        executor: &Executor,
        storage: &Path,
        store: &dyn RecordStore<PointKey>,
        costs_outstanding: Option<CostWalk>,
    ) -> Result<Outstanding, String> {
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
                // Forgotten under the request's lock, and only as it was
                // judged: a waiter that touched it since keeps it.
                false => {
                    let judged = &stored.waiter;
                    let _forgotten_or_kept = self
                        .files
                        .update(&(&stored.executor, &stored.request), LARGEST_REQUEST, |held: Option<Stored>| match held {
                            Some(held) if held.waiter != *judged => Err(()),
                            _ => Ok(None),
                        })
                        .map_err(|error| error.to_string())?;
                }
            }
        }
        waiting.sort_by(|first, second| second.waiter.cmp(&first.waiter));
        let loads = waiting.into_iter().map(|stored| stored.request).filter(|request| !requests.contains(request)).collect();
        Ok(Outstanding { standing: requests, loads })
    }
}

/// What a measurement job measures, in order ([`measure`]): assessment's
/// standing requests ([`standing`]), then the loads', while its time lasts
/// ([`JobDeadlines`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Outstanding {
    pub standing: Vec<PointRequest>,
    pub loads: Vec<PointRequest>,
}

impl Outstanding {
    /// Every request, in the order a job measures them.
    pub fn requests(&self) -> impl Iterator<Item = &PointRequest> {
        self.standing.iter().chain(&self.loads)
    }

    /// Whether no request is outstanding.
    pub fn is_empty(&self) -> bool {
        self.standing.is_empty() && self.loads.is_empty()
    }
}

/// Non-empty set of outstanding measurement requests blocking a price or plan.
///
/// Constructed only when at least one request is outstanding, so "pending on
/// nothing" is unrepresentable by construction.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PendingPrices(Outstanding);

impl PendingPrices {
    /// Wrap `outstanding` if at least one measurement request is outstanding.
    pub fn new(outstanding: Outstanding) -> Option<Self> {
        (!outstanding.is_empty()).then_some(Self(outstanding))
    }

    /// A pending set containing a single standing request.
    pub fn standing(request: PointRequest) -> Self {
        Self(Outstanding {
            standing: vec![request],
            loads: Vec::new(),
        })
    }

    /// A pending set containing a single load request.
    pub fn load(request: PointRequest) -> Self {
        Self(Outstanding {
            standing: Vec::new(),
            loads: vec![request],
        })
    }

    /// The underlying non-empty outstanding requests.
    pub fn outstanding(&self) -> &Outstanding {
        &self.0
    }
}

impl std::fmt::Display for PendingPrices {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} standing and {} load measurement request(s) pending",
            self.0.standing.len(),
            self.0.loads.len()
        )
    }
}

/// A measurement job's sampling budget: how much sampling time (`budget`)
/// the job spends across its points before it stops beginning new ones.
/// Discovering the machine, probing and opening the device, forming kernels
/// (cold JIT), and constructing points take none of the budget — only the
/// sampling window after a point is made (`Self::charge`) counts against it.
/// A point begun while budget remains is formed and timed to its end, and
/// one reached after the budget is exhausted is left unrecorded for the next
/// job. The standing points' samples share
/// [`MEASUREMENT_SECONDS`](super::executor_costs::MEASUREMENT_SECONDS) of sampling, which nothing but sampling takes.
/// So a busy host stretches each sample, and fewer points begin in a job
/// and fewer samples are read of each, but no point begun expires, none is
/// priced for want of time, and what is recorded of one never follows from
/// how busy its host was.
#[derive(Clone, Copy, Debug)]
pub struct JobDeadlines {
    started: std::time::Instant,
    budget: std::time::Duration,
    spent: std::time::Duration,
}

impl JobDeadlines {
    /// A job opened at `started` with a sampling budget of `time`.
    pub fn new(started: std::time::Instant, time: std::time::Duration) -> Self {
        Self { started, budget: time, spent: std::time::Duration::ZERO }
    }

    /// The sampling budget remaining before the job stops beginning points.
    pub fn remaining(&self) -> std::time::Duration {
        self.budget.saturating_sub(self.spent)
    }

    /// When the job begins no more points: already passed once the sampling
    /// budget is exhausted, else `remaining` from now.
    pub fn ends(&self) -> std::time::Instant {
        let remaining = self.remaining();
        if remaining.is_zero() {
            self.started
        } else {
            std::time::Instant::now() + remaining
        }
    }

    /// Charge `sampled` (time spent actually sampling a made point) against
    /// the job's budget.
    pub(super) fn charge(&mut self, sampled: std::time::Duration) {
        self.spent = self.spent.saturating_add(sampled);
    }
}

/// Measure `requests` on the device of `charges`, assessment's standing
/// ones first and then the loads', each in the order given, each point
/// stored in `store` as it is taken, by `deadlines`: each point loads from
/// the records or, reached before the job's time ends, is reported to
/// `attempting`, then measured to its end, sampling for its part of its
/// time left (or until its median is resolved); `visit` is told what
/// became of each point as soon as it is known. What the job's time does
/// not reach the next job measures. Every point's operands, backings and
/// kernels are allocated through `charges`: a job's own device's tally
/// (`Charges::tally`), or a measurement holding of a device's ledgers in
/// this process (`DeviceLedgers::measurement`). The job's split is
/// traced, point by point and whole. Fails only when the device's
/// executor is not of its own topology's revision.
pub fn measure(
    measurer: &mut Measurer,
    charges: &seismic::Charges,
    store: &dyn RecordStore<PointKey>,
    requests: &Outstanding,
    deadlines: JobDeadlines,
    attempting: &mut dyn FnMut(&PointKey) -> Attempt,
    visit: &mut Visit<'_>,
) -> Result<Measurement, ExecutorCostsError> {
    let device = &charges.device();
    let (topology, executor) = (device.topology(), device.info());
    let requests = (requests.standing.iter().map(|request| (Whose::Standing, request)))
        .chain(requests.loads.iter().map(|request| (Whose::Loads, request)))
        .collect::<Vec<_>>();
    let mut ended = Vec::new();
    let points = requests
        .iter()
        .map(|(_, request)| request.points(topology, executor).unwrap_or_else(|error| {
            ended.push(error);
            0
        }))
        .collect::<Vec<_>>();
    let mut converts = Vec::new();
    for (_, request) in &requests {
        if let PointRequest::Convert(form) = request {
            if !converts.contains(form) {
                converts.push(*form);
            }
        }
    }
    let programs = if requests
        .iter()
        .any(|(_, request)| matches!(request, PointRequest::Costs(_)))
    {
        ProgramSet::with_walk(executor, &converts)?
    } else {
        ProgramSet::converts(executor, &converts)
    };
    let mut walks = Walks::new(
        measurer,
        device,
        charges,
        store,
        deadlines,
        programs,
        attempting,
        visit,
    )?;
    let mut failures = Vec::new();
    for (index, (whose, request)) in requests.iter().enumerate() {
        // Its points and those after it of the same whose.
        let left = requests[index..]
            .iter()
            .zip(&points[index..])
            .filter(|((other, _), _)| other == whose)
            .map(|(_, points)| points)
            .sum();
        walks.begin(*whose, left);
        let (end, failed) = request.measure(&mut walks);
        ended.extend(end);
        failures.extend(failed);
    }
    tracing::info!(seconds = deadlines.started.elapsed().as_secs_f64(), "measurement job measured its requests");
    Ok(Measurement { ended, failures, held: walks.held() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::executor_costs::MEASUREMENT_SECONDS;
    use super::super::streaming_costs::MemoryError;

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
        let order = |requests: &PointRequests| {
            requests.outstanding(&topology, &executor, &storage, &records, costs).unwrap().requests().cloned().collect::<Vec<_>>()
        };
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

    /// What a job records of a point never follows from how long its host
    /// took to reach it: on the host's CPU, two host memory lines (standing
    /// requests) reached after a busy host spent more than their whole
    /// budget's wall time are each still read, in the order requested, as
    /// only sampling takes their budget; a job whose time has ended begins
    /// none of them, nor the probe, and leaves them unrecorded for the next;
    /// and a later job loads every point stored, measuring none again,
    /// whatever time it has.
    #[test]
    fn a_job_reads_every_point_it_begins_however_busy_its_host_and_leaves_the_rest_to_the_next() {
        let device = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
        let requests = Outstanding {
            standing: vec![PointRequest::HostMemory { readers: 2 }, PointRequest::HostMemory { readers: 1 }],
            loads: Vec::new(),
        };
        // A job of `seconds` that its host started `spent` seconds ago.
        let job = |store: &JsonFileStore, spent: u64, seconds: u64| {
            let mut measurer = Measurer::new(crate::SEARCH_SETTINGS.error_rate, crate::MIN_SAMPLE_SECONDS);
            let started = std::time::Instant::now()
                .checked_sub(std::time::Duration::from_secs(spent))
                .expect("the host's clock reaches back past the job's start");
            let deadlines = JobDeadlines::new(started, std::time::Duration::from_secs(seconds));
            let mut visited = Vec::new();
            measure(&mut measurer, &seismic::Charges::tally(&device), store, &requests, deadlines, &mut |_| Attempt::Proceed, &mut |key, outcome| {
                let readers = key.configuration["readers"];
                visited.push((readers, outcome));
            })
            .unwrap();
            visited
        };
        use seismic::measure::Outcome;
        let topology = device.topology();
        let root = std::env::temp_dir().join(format!("point-requests-job-{}", std::process::id()));
        // No time: nothing is begun or recorded, so the next job measures
        // them.
        let store = JsonFileStore::new(&root.join("records"));
        let none = job(&store, 0, 0);
        assert!(none.iter().all(|(_, outcome)| matches!(outcome, Outcome::Expired)), "{none:?}");
        assert!(matches!(MemoryCosts::stored(topology, device.info(), 2, &store), Err(MemoryError::Unrecorded { .. })));
        // Nor the probe: no record holds it, and the next job measures it.
        let mut measurer = Measurer::new(crate::SEARCH_SETTINGS.error_rate, crate::MIN_SAMPLE_SECONDS);
        let deadlines = JobDeadlines::new(std::time::Instant::now(), std::time::Duration::ZERO);
        let costs = Outstanding { standing: vec![PointRequest::Costs(CostWalk::Required)], loads: Vec::new() };
        measure(&mut measurer, &seismic::Charges::tally(&device), &store, &costs, deadlines, &mut |_| Attempt::Proceed, &mut |_, _| {}).unwrap();
        assert!(matches!(
            super::super::ExecutorCosts::stored(topology, device.info(), &store),
            Ok(super::super::StoredCosts::Missing(_))
        ));
        // A busy host spent more than the standing points' budget before
        // they were reached: each is still read, in the order requested.
        let spent = MEASUREMENT_SECONDS as u64 + 1;
        let busy = job(&store, spent, 600);
        assert_eq!(busy.iter().map(|(readers, _)| *readers).collect::<Vec<_>>(), [2, 2, 1, 1], "in the order requested");
        assert!(busy.iter().all(|(_, outcome)| matches!(outcome, Outcome::Measured(_))), "{busy:?}");
        let again = job(&store, 0, 0);
        assert!(again.iter().all(|(_, outcome)| matches!(outcome, Outcome::Loaded(_))), "{again:?}");
        let records = |readers| MemoryCosts::stored(topology, device.info(), readers, &store);
        assert!(records(1).is_ok() && records(2).is_ok());
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// The machine's line runs only on request: no standing request, and no
    /// request outstanding for a load, is the line's, and a tool that asks
    /// for the line has the job measure it alone.
    #[cfg(feature = "machine-line")]
    #[test]
    fn the_line_is_asked_for_and_never_stood() {
        let (topology, executor, _) = host();
        let storage = std::env::temp_dir();
        let records = seismic::measure::HeldStore::default();
        let costs = walk_outstanding(&topology, &executor, &records);
        let root = std::env::temp_dir().join(format!("point-requests-scope-{}", std::process::id()));
        let all = PointRequests::new(&root).outstanding(&topology, &executor, &storage, &records, costs).unwrap();
        assert!(all.standing.iter().any(|request| matches!(request, PointRequest::Costs(_))));
        let standing = standing(&topology, &executor, &storage).unwrap();
        assert!(!standing.iter().chain(all.requests()).any(|request| *request == PointRequest::Line));
        let line = line_outstanding(&topology, &executor, &records, costs);
        assert_eq!(line, Outstanding { standing: Vec::new(), loads: vec![PointRequest::Line] });
    }

    /// The CPU's line is measured by the job, its probes formed and run on
    /// the host: after one job (a line is tens of points, so several jobs
    /// complete it; the records of a complete line are read back in
    /// `line`'s tests) the line is whole or has only points the job's time
    /// did not hold left, none defective or unobtainable.
    #[cfg(feature = "machine-line")]
    #[test]
    fn the_job_measures_the_cpu_line() {
        use super::super::line::{MachineLine, MachineLineError, Peak, Stream};
        let device =
            crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu)
                .unwrap();
        let (topology, executor) = (device.topology(), device.info());
        let root = std::env::temp_dir().join(format!("point-requests-line-{}", std::process::id()));
        let store = JsonFileStore::new(&root);
        let requests = line_outstanding(topology, executor, &store, None);
        assert!(!requests.is_empty());
        let mut measurer = Measurer::new(crate::SEARCH_SETTINGS.error_rate, crate::MIN_SAMPLE_SECONDS);
        let deadlines = JobDeadlines::new(std::time::Instant::now(), std::time::Duration::from_secs(60));
        measure(&mut measurer, &seismic::Charges::tally(&device), &store, &requests, deadlines, &mut |_| Attempt::Proceed, &mut |_, _| {})
            .unwrap();
        match MachineLine::stored(topology, executor, &store) {
            Ok(line) => {
                assert!(matches!(line.stream, Stream::Host(_)), "{:?}", line.stream);
                assert!(
                    line.peaks.iter().all(|(_, peak)| !matches!(peak, Peak::Unobtainable(_) | Peak::Defect(_))),
                    "{:?}",
                    line.peaks
                );
            }
            Err(MachineLineError::Unrecorded { .. } | MachineLineError::UnrecordedSweep { .. }) => {}
            Err(error) => panic!("{error}"),
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    /// Assessment's standing requests are always waited on, and are of the
    /// machine alone: with no load's request, the costs, then the reference
    /// convert where a device streams, the model store's read line, and the
    /// host memory's line and the reference CPU site where a bank can have a
    /// host tier, while none is measured.
    #[test]
    fn the_standing_requests_are_the_machines() {
        let root = std::env::temp_dir().join(format!("point-requests-standing-{}", std::process::id()));
        let (topology, executor, _) = host();
        let storage = std::env::temp_dir();
        let records = seismic::measure::HeldStore::default();
        let costs = walk_outstanding(&topology, &executor, &records);
        let outstanding = PointRequests::new(&root).outstanding(&topology, &executor, &storage, &records, costs).unwrap();
        assert!(outstanding.loads.is_empty(), "{outstanding:?}");
        let outstanding = outstanding.standing;
        assert_eq!(outstanding.first(), Some(&PointRequest::Costs(CostWalk::Required)));
        assert!(outstanding.contains(&PointRequest::Storage { directory: storage.clone() }));
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

    #[test]
    fn sampling_budget_ignores_elapsed_wall_time_until_charged() {
        let started = std::time::Instant::now() - std::time::Duration::from_secs(30);
        let budget = std::time::Duration::from_secs(10);
        let mut deadlines = JobDeadlines::new(started, budget);
        // 30 s of wall time (e.g. cold JIT) passed since `started`, yet none
        // of the 10 s sampling budget has been spent.
        assert_eq!(deadlines.remaining(), budget);
        assert!(deadlines.ends() > std::time::Instant::now());
        deadlines.charge(std::time::Duration::from_secs(4));
        assert_eq!(deadlines.remaining(), std::time::Duration::from_secs(6));
        assert!(deadlines.ends() > std::time::Instant::now());
        deadlines.charge(std::time::Duration::from_secs(6));
        assert_eq!(deadlines.remaining(), std::time::Duration::ZERO);
        assert!(deadlines.ends() <= std::time::Instant::now());
        // A zero-budget deadline is immediately exhausted.
        let empty = JobDeadlines::new(std::time::Instant::now(), std::time::Duration::ZERO);
        assert_eq!(empty.remaining(), std::time::Duration::ZERO);
        assert!(empty.ends() <= std::time::Instant::now());
    }

    /// Every leaf `Priced::Pending` request agrees with `PointRequest::outstanding`
    /// and is satisfied when the measurement job runs that request.
    #[test]
    fn every_leaf_pending_request_is_satisfied_by_measuring_it() {
        let device = crate::test_support::open_backend(&seismic::Devices::discover(&[]).unwrap(), seismic::BackendName::Cpu).unwrap();
        let (topology, executor) = (device.topology(), device.info());
        let root = std::env::temp_dir().join(format!("leaf-roundtrip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let storage = root.join("storage");
        std::fs::create_dir_all(&storage).unwrap();
        let store = JsonFileStore::new(&root.join("records"));

        let run_job = |requests: &Outstanding| {
            let mut measurer = Measurer::new(crate::SEARCH_SETTINGS.error_rate, crate::MIN_SAMPLE_SECONDS);
            let deadlines = JobDeadlines::new(std::time::Instant::now(), std::time::Duration::from_secs(600));
            measure(
                &mut measurer,
                &seismic::Charges::tally(&device),
                &store,
                requests,
                deadlines,
                &mut |_| Attempt::Proceed,
                &mut |_, _| {},
            )
            .unwrap();
        };

        // 1. StorageCosts leaf
        let Priced::Pending(pending) = StorageCosts::priced(topology, executor, &storage, &store) else {
            panic!("empty store must leave StorageCosts pending");
        };
        assert!(pending.outstanding().requests().all(|r| r.outstanding(topology, executor, &store, None)));
        run_job(pending.outstanding());
        assert!(matches!(StorageCosts::priced(topology, executor, &storage, &store), Priced::Measured(_)));
        assert!(pending.outstanding().requests().all(|r| !r.outstanding(topology, executor, &store, None)));

        // 2. MemoryCosts leaf
        let Priced::Pending(pending) = MemoryCosts::priced(topology, executor, 1, &store) else {
            panic!("empty store must leave MemoryCosts pending");
        };
        assert!(pending.outstanding().requests().all(|r| r.outstanding(topology, executor, &store, None)));
        run_job(pending.outstanding());
        assert!(matches!(MemoryCosts::priced(topology, executor, 1, &store), Priced::Measured(_)));
        assert!(pending.outstanding().requests().all(|r| !r.outstanding(topology, executor, &store, None)));

        // 3. StoredCosts leaf (CostWalk::Required)
        let walked = super::super::ExecutorCosts::stored(topology, executor, &store);
        let costs_out = super::super::StoredCosts::outstanding_of(&walked);
        let Priced::Pending(pending) = super::super::StoredCosts::priced(walked) else {
            panic!("empty store must leave StoredCosts pending");
        };
        assert!(pending.outstanding().requests().all(|r| r.outstanding(topology, executor, &store, costs_out)));
        run_job(pending.outstanding());
        let walked_after = super::super::ExecutorCosts::stored(topology, executor, &store);
        let costs_out_after = super::super::StoredCosts::outstanding_of(&walked_after);
        let Priced::Measured(costs) = super::super::StoredCosts::priced(walked_after) else {
            panic!("measuring CostWalk::Required must satisfy StoredCosts");
        };
        assert!(pending.outstanding().requests().all(|r| !r.outstanding(topology, executor, &store, costs_out_after)));

        // 4. expert_lines host link line leaf (CostWalk::Whole)
        let unread_pass = magnitude_cost_model::ExpertPass::new(
            costs.rates(&[], &[]),
            vec![(magnitude_cost_model::RoutedKind(0), Vec::new(), Vec::new())],
            magnitude_cost_model::HostReads::Unread,
        );
        let dummy_host = magnitude_cost_model::HostLines {
            cpu: None,
            memory: Some(magnitude_cost_model::Nanoseconds::of(1.0).unwrap()),
            landing: None,
            record_bytes: 1,
        };
        let Priced::Pending(pending) = super::super::expert_lines(&[dummy_host], &unread_pass) else {
            panic!("unread host route must leave expert_lines pending");
        };
        assert_eq!(pending.outstanding().standing, vec![PointRequest::Costs(CostWalk::Whole)]);
        assert!(pending.outstanding().requests().all(|r| r.outstanding(topology, executor, &store, Some(CostWalk::Whole))));
        run_job(pending.outstanding());
        let walked_whole = super::super::ExecutorCosts::stored(topology, executor, &store);
        let costs_out_whole = super::super::StoredCosts::outstanding_of(&walked_whole);
        assert_eq!(costs_out_whole, None, "measuring CostWalk::Whole must satisfy every cost point");
        assert!(pending.outstanding().requests().all(|r| !r.outstanding(topology, executor, &store, costs_out_whole)));

        // 5. ConvertCosts (on every host executor whose backend streams) and CpuCosts
        //    leaves agree with PointRequest::outstanding.
        let empty_store = seismic::measure::HeldStore::default();
        for exec in topology.executors() {
            if let Some(convert_form) = ConvertForm::reference(exec.backend()).unwrap() {
                let Priced::Pending(convert_pending) = ConvertCosts::priced(topology, exec, convert_form, &empty_store) else {
                    panic!("empty store must leave ConvertCosts pending on {:?}", exec.backend());
                };
                assert_eq!(convert_pending.outstanding().standing, vec![PointRequest::Convert(convert_form)]);
                assert!(convert_pending.outstanding().requests().all(|r| r.outstanding(topology, exec, &empty_store, None)));
            }
        }

        let recorded = seismic::fixtures::recorded::rtx3090();
        let cuda = recorded.executors().iter().find(|e| e.backend() == seismic::BackendName::Cuda).unwrap();
        let cpu_form = CpuForm::reference(cuda.backend()).unwrap().unwrap();
        let Priced::Pending(cpu_pending) = CpuCosts::priced(&recorded, cuda, &cpu_form, &empty_store) else {
            panic!("empty store must leave CpuCosts pending");
        };
        assert_eq!(cpu_pending.outstanding().standing, vec![PointRequest::CpuSite(cpu_form.clone())]);
        assert!(cpu_pending.outstanding().requests().all(|r| r.outstanding(&recorded, cuda, &empty_store, None)));

        let _ = std::fs::remove_dir_all(&root);
    }
}
