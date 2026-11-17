//! The measurement job (device topology spec §2.15): a mode of the worker
//! process that measures this machine for assessment, so that nothing in the
//! host process opens an executor or waits on a native submission.
//!
//! The job discovers the machine, opens the selected executor and measures
//! the point requests outstanding on it
//! ([`magnitude_executor::assessment::requests`]: assessment's standing
//! requests first, their samples sharing `MEASUREMENT_SECONDS` of sampling
//! whatever time the job is given; then the loads', the load that waited
//! latest first; each point begun while the job's time lasts, from its
//! device's opening ([`job_time`]), formed and timed to its end, what it
//! does not reach the next job's), through
//! `seismic-measure`, each record stored in the kernel cache as it is
//! taken, and ends with the points it could not measure. Before each point it records the key it is
//! attempting ([`AttemptMarkers`]): a point that ends two jobs (the second
//! confirms the first was no transient) is recorded unobtainable with its
//! reason and not measured again while the hardware and driver it names are
//! the machine's; the costs complete without it, and only estimates that
//! need it are unavailable. Opening the device is no point's attempt: it is
//! the machine's present state, not a property of a point, so the job
//! reports that it opened ([`MeasureEvent::Opened`]) and its host keeps
//! account of the jobs that could not. A point whose timing fails is left unrecorded
//! and counts as one ended job, so only this rule makes a timing failure
//! permanent, as does a point whose operands could not be made now; a
//! kernel that cannot be formed is recorded unobtainable at once, as
//! formation is deterministic for its key.

use super::client::{handshake, receive, Lost};
use super::protocol::{EngineBuild, HostMessage, WorkerMessage};
use super::transport::{HostTransport, MessageSender};
use super::{Held, Served, WorkerExit};
use magnitude_executor::assessment::{requests, CostWalk, ExecutorCosts, PointRequests, StoredCosts};
use magnitude_executor::KernelCache;
use seismic::measure::{Attempt, JsonFileStore, Outcome, Reason, Record, RecordStore, StoreError};
use seismic::{Devices, KernelIdentity, Measurer, PointKey};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The jobs a point may end before it is unobtainable: the second confirms
/// the first was not a transient.
const ENDED_JOBS: u32 = 2;

/// The time each measurement job is given to begin points in, whichever
/// host runs the jobs: in its own process ([`InProcessJobs`]) or in a worker
/// beside it, from the moment the job's device is open
/// ([`MeasureEvent::Opened`]; `requests::JobDeadlines`), so discovering the
/// machine, probing the device and opening it take none of it. It is the
/// standing points' budget for their samples (`MEASUREMENT_SECONDS`), so a
/// job on an idle host reaches all of them that sampling alone allows. It
/// decides only when the job begins no more points: each begun is formed
/// and timed to its end, and one not begun is left unrecorded, the next job
/// resuming from the records stored. So a busy host stretches each sample,
/// and fewer points begin in a job, but no point begun expires and none is
/// priced for want of time: how busy the host is decides which job reads a
/// figure, never what is recorded of it.
pub fn job_time() -> std::time::Duration {
    std::time::Duration::from_secs_f64(magnitude_executor::assessment::MEASUREMENT_SECONDS)
}

/// How long a running job may be silent before it has hung: the hang
/// guard every host of a job keeps ([`hang_silence`]), its silence counted
/// from the job's last event ([`MeasureEvent`]), or its start. A job reports
/// a point's attempt as it begins it and its record as it stores it, so
/// once it is open its longest silence is one point's forming and sampling,
/// the two things it does between those events: the worst cold formation
/// of a point's kernels measured, 23 s (`routed_land` for the convert point
/// on CUDA, an RTX 3090, 2026-11-17, against 0.18 s once stored), plus a
/// point's most sampling, a standing point's whole `MEASUREMENT_SECONDS`;
/// and, as [`seismic::PROBE_LIMIT`] is over the longest probes seen, 7.5x
/// their sum, for slower compilers and hosts no rig has shown. A compile and
/// a sample are each one native call, which reports nothing until it ends,
/// so no finer event bounds them. A job silent past it has hung (as in a
/// driver), and the point it was inside counts as one its job ended.
pub const POINT_LIMIT: std::time::Duration = std::time::Duration::from_secs(250);

/// How long a job may be silent before it opens its device
/// ([`MeasureEvent::Opened`]), its first event: its probes begin within
/// [`seismic::PROBE_LIMIT`] of its start (`measure_machine`), the last begun
/// runs at most another, and opening the device is what a probe's process
/// does before it probes (a start and a device open, which that limit
/// covers), so a third covers it. A job not open by then has hung in its
/// opening, and could not open the machine.
pub const OPENING_LIMIT: std::time::Duration = std::time::Duration::from_secs(3 * seismic::PROBE_LIMIT.as_secs());

/// How long a job, `opened` its device or not, may be silent from its last
/// event (or its start) before it has hung, and its host kills it: the one
/// hang policy of every host of a job, which reads its events as they come
/// ([`observe_measurement`], [`measure_worker`]).
pub fn hang_silence(opened: bool) -> std::time::Duration {
    if opened {
        POINT_LIMIT
    } else {
        OPENING_LIMIT
    }
}

/// What one measurement job measures.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasureRequest {
    /// The paths the host's inventory serves: the job discovers with them,
    /// as the host did.
    pub paths: Vec<PathBuf>,
    /// The kernel cache the records and their attempts are stored in.
    pub kernels: PathBuf,
    /// The model store, whose volume's read line the job measures
    /// (`PointRequest::Storage`).
    pub storage: PathBuf,
    /// The executor whose cost points are measured: the one a native load
    /// selects.
    pub executor: seismic::ExecutorSelector,
    /// This job, as its attempt markers name it, so the attempt of a job
    /// its host preempts is released rather than counted ([`release_killed`]).
    pub job: MeasureJob,
    /// The time the job has to begin points in from its device's opening
    /// ([`job_time`]), however many are requested (`requests::JobDeadlines`).
    pub time: std::time::Duration,
}

/// One measurement job's identity, which its host chooses distinct from the
/// jobs before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasureJob(pub u64);

/// What a running job reports as it goes, in order.
#[derive(Clone, Debug, PartialEq)]
pub enum MeasureEvent {
    /// It is about to record its attempt of the point and time it.
    Attempting(PointKey),
    /// It opened its device, before any attempt: a job that ends, or is
    /// killed, before reporting this could not open the machine, which no
    /// point's attempt counts, as it is the machine's present state rather
    /// than a point's; its host keeps account of it.
    Opened,
    /// It stored the point's record.
    Recorded(PointKey, Record),
}

/// What a job could not measure; everything else it measured is stored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasureFailure {
    /// The machine could not be discovered, or the selected executor opened.
    Machine { reason: String },
    /// The selected executor could not be opened now: it is out of memory,
    /// or another process holds it (`seismic::TargetError::DeviceBusy`).
    /// Opening it later may succeed.
    Busy { reason: String },
    /// One cost point: its kernel could not be formed (it is recorded
    /// unobtainable with the failure), its timing failed (it is left
    /// unrecorded, for the next job), or its record could not be read or
    /// written.
    Point { key: PointKey, reason: String },
    /// An attempt could not be recorded, so a point ran unrecorded.
    Attempt { reason: String },
    /// A figure after the probe could not be read outside its points'
    /// timings: the costs keep pricing it from those read.
    Figure { reason: String },
    /// A device-property probe kept no result (`seismic::ProbeFailure`):
    /// the device keeps that probe's safe value, and the next job runs it
    /// again. What was measured stands.
    Probe { reason: String },
}

impl fmt::Display for MeasureFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Machine { reason } | Self::Busy { reason } => write!(formatter, "the machine: {reason}"),
            Self::Point { key, reason } => {
                write!(formatter, "the point")?;
                for (name, value) in key.configuration.iter().chain(&key.point) {
                    write!(formatter, " {name}={value}")?;
                }
                // The executor is the request's; the kernels' identities tell
                // the points' kinds apart.
                let kernels = key.kernels.iter().map(KernelIdentity::as_str);
                let kernels = kernels.collect::<Vec<_>>().join(", ");
                write!(formatter, " of kernels {kernels}: {reason}")
            }
            Self::Attempt { reason } => write!(formatter, "recording an attempt: {reason}"),
            Self::Figure { reason } => write!(formatter, "a priced figure: {reason}"),
            Self::Probe { reason } => write!(formatter, "a device probe: {reason}"),
        }
    }
}

impl MeasureFailure {
    /// Whether it is the job's own failure. A figure that could not be read
    /// outside its points' timings fails each job alike and stays priced,
    /// and a probe that kept nothing leaves its property at its safe value,
    /// so neither is any job's.
    pub fn fails_job(&self) -> bool {
        !matches!(self, Self::Figure { .. } | Self::Probe { .. })
    }
}

/// A job that ended without reporting its outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobLost(pub String);

impl fmt::Display for JobLost {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for JobLost {}

/// The jobs that ended while measuring one point, since it last resolved,
/// and the job whose attempt counted last.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
struct AttemptRecord {
    ended_jobs: u32,
    job: Option<MeasureJob>,
}

impl seismic::measure::Stored for AttemptRecord {
    const SCHEMA: &'static str = "attempt.1";
}

/// The most bytes an [`AttemptRecord`] is written as: `{"ended_jobs":`,
/// the ten digits of `u32::MAX`, `,"job":`, the twenty of `u64::MAX` and
/// `}`.
const LARGEST_RECORD: u64 = 14 + 10 + 7 + 20 + 1;

/// The measurement job's attempts, kept in the kernel cache beside the
/// records ([`KernelCache::measurement_attempts`]). An attempt is recorded
/// as one more ended job before its point is formed and timed, and cleared
/// when the point resolves with a record; so a job that ends while timing
/// it, or whose timing of it fails, leaves its key counted. Keyed by the
/// point's key, so a changed driver or device starts its count anew.
struct AttemptMarkers {
    files: JsonFileStore,
}

impl AttemptMarkers {
    /// The attempts kept in the kernel cache at `kernels`.
    fn new(kernels: &Path) -> Self {
        Self {
            files: KernelCache::measurement_attempts(kernels),
        }
    }

    fn record(&self, key: &impl Serialize) -> Result<AttemptRecord, StoreError> {
        Ok(self.files.load(key, LARGEST_RECORD)?.unwrap_or_default())
    }

    /// Replace `key`'s record by `change` of it, every other change of it,
    /// from this job, another or the host, waiting
    /// ([`JsonFileStore::update`]); none leaves it as it is.
    fn change(&self, key: &impl Serialize, change: impl FnOnce(AttemptRecord) -> Option<AttemptRecord>) -> Result<(), StoreError> {
        self.files
            .update(key, LARGEST_RECORD, |record: Option<AttemptRecord>| change(record.unwrap_or_default()).ok_or(()))
            .map(|_changed_or_left| ())
    }

    /// Why `key` cannot be measured on this machine: it ended two jobs.
    /// `None` while it may be attempted.
    fn unobtainable(&self, key: &impl Serialize) -> Result<Option<String>, StoreError> {
        let ended = self.record(key)?.ended_jobs;
        Ok((ended >= ENDED_JOBS).then(|| {
            format!("measuring it ended {ended} measurement jobs on this hardware and driver")
        }))
    }

    /// Record that `job` is about to time `key`.
    fn attempting(&self, key: &impl Serialize, job: MeasureJob) -> Result<(), StoreError> {
        self.change(key, |record| {
            Some(AttemptRecord {
                ended_jobs: record.ended_jobs.saturating_add(1),
                job: Some(job),
            })
        })
    }

    /// Undo `job`'s attempt of `key`, where it is the attempt counted last:
    /// its host preempted the job, which says nothing of the point.
    fn release(&self, key: &impl Serialize, job: MeasureJob) -> Result<(), StoreError> {
        self.change(key, |record| {
            (record.job == Some(job)).then(|| AttemptRecord {
                ended_jobs: record.ended_jobs.saturating_sub(1),
                job: None,
            })
        })
    }

    /// Record that `key`'s point resolved, with a record: the job did not
    /// end on it.
    fn resolved(&self, key: &impl Serialize) -> Result<(), StoreError> {
        self.change(key, |record| (record.ended_jobs != 0).then(AttemptRecord::default))
    }
}

/// Release `job`'s attempt of `key` in the kernel cache at `kernels`: its
/// host preempted the job while timing it (for a load), which says nothing
/// of the point, so it is attempted again uncounted. Only `job`'s own attempt is
/// undone; a key it never marked is left as it is.
pub fn release_killed(
    kernels: &Path,
    job: MeasureJob,
    key: &PointKey,
) -> Result<(), StoreError> {
    AttemptMarkers::new(kernels).release(key, job)
}

/// A Measure: run the job of `request` and report each attempt and each
/// record it stores, then what could not be measured. The process ends
/// after the outcome, with what the job's points still hold.
pub(super) fn measure_only(request: MeasureRequest, sender: impl MessageSender<WorkerMessage>) -> Served {
    report_measurement(&request, sender, measure_machine)
}

/// The job of `request` run by `job` ([`measure_machine`], or a tool's
/// other job), its events and outcome reported to its host over `sender` as
/// a worker reports a Measure, greeting it first: what a host reads to keep
/// the job's hang guard ([`observe_measurement`]). The process ends after
/// the outcome, with what the job's points still hold.
pub fn report_measurement(
    request: &MeasureRequest,
    mut sender: impl MessageSender<WorkerMessage>,
    job: fn(&MeasureRequest, &mut dyn FnMut(MeasureEvent)) -> MeasureEnd,
) -> Served {
    let hello = WorkerMessage::Hello {
        build: EngineBuild::current(),
    };
    if let Err(error) = sender.send(hello) {
        return WorkerExit::TransportFailed(error.to_string()).into();
    }
    let mut lost = None;
    let MeasureEnd { failures, held } = job(request, &mut |event| {
        if lost.is_none() {
            let message = match event {
                MeasureEvent::Attempting(key) => WorkerMessage::Attempting { key },
                MeasureEvent::Opened => WorkerMessage::Opened,
                MeasureEvent::Recorded(key, record) => WorkerMessage::Recorded { key, record },
            };
            lost = sender.send(message).err().map(|error| error.to_string());
        }
    });
    let exit = match lost {
        Some(error) => WorkerExit::TransportFailed(error),
        None => match sender.send(WorkerMessage::MeasureFinished { failures }) {
            Ok(()) => WorkerExit::Measured,
            Err(error) => WorkerExit::TransportFailed(error.to_string()),
        },
    };
    // Every record is stored: what the points hold is released as the
    // process ends, past the job's report ([`end_measured`]).
    Served {
        exit,
        _held: held.map(|points| Held::Measurement { _points: points }),
    }
}

/// End a measurement job's process with `code` once the job is reported,
/// with what its points hold (`held`): at once ([`super::end_at_once`]),
/// with no wait for compiles still forming. A job that held nothing exits
/// normally.
pub fn end_measured(held: Option<magnitude_executor::assessment::Held>, code: i32) -> ! {
    match held {
        // Held, not dropped, until the process ends.
        Some(_points) => super::end_at_once(code),
        None => std::process::exit(code),
    }
}

/// How a measurement job ended: what it could not measure, and what its
/// points were made in and from, which its caller releases once it has
/// reported the job ([`magnitude_executor::assessment::Measurement::held`]).
pub struct MeasureEnd {
    pub failures: Vec<MeasureFailure>,
    pub held: Option<magnitude_executor::assessment::Held>,
}

/// Measurement jobs run in this process ([`measure_machine`]), one after
/// another, on the executor a load of a manifest selects, each given the
/// [`job_time`]: how a host that runs no job beside it (a tool, a load in its
/// own process) has what a load waits on measured.
pub struct InProcessJobs {
    request: MeasureRequest,
    topology: Arc<seismic::DeviceTopology>,
    executor: seismic::Executor,
    /// Whether the jobs take the machine's line alone ([`Self::of_line`]).
    #[cfg(feature = "machine-line")]
    line: bool,
}

/// What the in-process jobs measured.
pub struct JobRun {
    /// The requests still outstanding after them.
    pub outstanding: magnitude_executor::assessment::Outstanding,
    /// What the job that measured could not.
    pub failures: Vec<MeasureFailure>,
}

/// Why the in-process jobs stopped measuring.
#[derive(Debug)]
pub enum JobsStopped {
    /// The outstanding requests could not be read.
    Requests(String),
    /// A job measured nothing: every point it began failed, or it was
    /// given no time to begin one.
    Stalled {
        outstanding: magnitude_executor::assessment::Outstanding,
        failures: Vec<MeasureFailure>,
    },
}

impl fmt::Display for JobsStopped {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Requests(reason) => write!(formatter, "the outstanding point requests: {reason}"),
            Self::Stalled { outstanding, failures } => {
                write!(formatter, "a measurement job measured nothing of {outstanding:?}")?;
                for failure in failures {
                    write!(formatter, "; {failure}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for JobsStopped {}

impl InProcessJobs {
    /// Jobs for `manifest`'s load on the machine `catalog` discovered with
    /// `paths`, its records in its kernel cache; none without one, where
    /// nothing measured reaches a load.
    pub fn of_load(catalog: &Devices, paths: Vec<PathBuf>, manifest: &crate::options::ExecutionManifest) -> Result<Option<Self>, String> {
        let Some(kernels) = manifest.kernel_cache.clone() else { return Ok(None) };
        let selected = crate::planning::select_load_device(catalog, manifest).map_err(|error| error.to_string())?;
        Ok(Some(Self {
            request: MeasureRequest {
                paths,
                kernels,
                storage: manifest.store().to_path_buf(),
                executor: selected.info.selector(),
                job: MeasureJob(0),
                time: job_time(),
            },
            topology: catalog.topology(),
            executor: selected.info,
            #[cfg(feature = "machine-line")]
            line: false,
        }))
    }

    /// Jobs for the machine's line alone on `executor` of the machine
    /// `catalog` discovered, its records in `kernels`: a development tool
    /// that needs an executor's line (the host CPU's beside a GPU's load)
    /// asks for it and no other figure.
    #[cfg(feature = "machine-line")]
    pub fn of_line(catalog: &Devices, kernels: PathBuf, storage: PathBuf, executor: seismic::Executor) -> Self {
        Self {
            request: MeasureRequest {
                paths: Vec::new(),
                kernels,
                storage,
                executor: executor.selector(),
                job: MeasureJob(0),
                time: job_time(),
            },
            topology: catalog.topology(),
            executor,
            line: true,
        }
    }

    /// The requests a job has points of left to measure, in its order
    /// ([`PointRequests::outstanding`]).
    pub fn outstanding(&self) -> Result<magnitude_executor::assessment::Outstanding, String> {
        let records = KernelCache::measurement_records(&self.request.kernels);
        let costs = StoredCosts::outstanding_of(&ExecutorCosts::stored(&self.topology, &self.executor, &records));
        #[cfg(feature = "machine-line")]
        if self.line {
            use magnitude_executor::assessment::line_outstanding;
            return Ok(line_outstanding(&self.topology, &self.executor, &records, costs));
        }
        PointRequests::new(&self.request.kernels).outstanding(&self.topology, &self.executor, &self.request.storage, &records, costs)
    }

    /// Run a job to its end, releasing what it held: what it left
    /// outstanding, or that it measured nothing.
    pub fn run(&mut self) -> Result<JobRun, JobsStopped> {
        let before = self.outstanding().map_err(JobsStopped::Requests)?;
        self.request.job = MeasureJob(self.request.job.0 + 1);
        let mut recorded = false;
        let mut report = |event: MeasureEvent| recorded |= matches!(event, MeasureEvent::Recorded(..));
        #[cfg(feature = "machine-line")]
        let end = match self.line {
            true => measure_line(&self.request, &mut report),
            false => measure_machine(&self.request, &mut report),
        };
        #[cfg(not(feature = "machine-line"))]
        let end = measure_machine(&self.request, &mut report);
        drop(end.held);
        let outstanding = self.outstanding().map_err(JobsStopped::Requests)?;
        match recorded || outstanding != before {
            true => Ok(JobRun { outstanding, failures: end.failures }),
            false => Err(JobsStopped::Stalled { outstanding, failures: end.failures }),
        }
    }
}

/// The measurement job's probing of the device alone, as a job probes before
/// it measures ([`measure_machine`]): executor `executor` of the machine
/// discovered with `paths` runs each of its probes the process's store
/// (`seismic::ProgramStore::user`, which every device opens with) holds no
/// result of, in a process of its own, within `time`. A tool whose check
/// reads probed properties runs it before the load it checks; a load never
/// does.
pub fn probe_machine(
    paths: &[PathBuf],
    executor: seismic::ExecutorSelector,
    time: std::time::Duration,
) -> Result<seismic::ProbeRun, String> {
    let deadline = std::time::Instant::now() + time;
    let catalog = Devices::discover(paths).map_err(|error| error.to_string())?;
    let executor = catalog.resolve(executor).map_err(|error| error.to_string())?;
    catalog.probe(executor, &*seismic::ProgramStore::user(), deadline).map_err(|error| error.to_string())
}

/// Measure the outstanding point requests of the request's executor, in
/// the order of who waits on them ([`PointRequests::outstanding`]), by
/// the job's time from its device's opening (`requests::JobDeadlines`):
/// assessment's first, then the loads', each begun while the request's time
/// lasts formed and timed to its end, what it does not reach left for the
/// next job, its attempts released. A point that ended two jobs is not timed: it
/// is recorded unobtainable, so the executor's costs complete without it. Each attempt reaches `report`
/// before it is recorded, and each record as soon as the job stores it.
/// Returns what could not be measured, and what the job's points were made
/// in and from. The worker's measure mode runs it; a tool may run it in its
/// own process.
pub fn measure_machine(
    request: &MeasureRequest,
    report: &mut dyn FnMut(MeasureEvent),
) -> MeasureEnd {
    measure_job(request, report, &|topology, executor, records, costs| {
        PointRequests::new(&request.kernels).outstanding(topology, executor, &request.storage, records, costs)
    })
}

/// The job of [`measure_machine`] for the machine's line alone: a
/// development measurement no served engine's job takes, which a tool that
/// needs an executor's line asks for and no other figure.
#[cfg(feature = "machine-line")]
pub fn measure_line(request: &MeasureRequest, report: &mut dyn FnMut(MeasureEvent)) -> MeasureEnd {
    measure_job(request, report, &|topology, executor, records, costs| {
        Ok(magnitude_executor::assessment::line_outstanding(topology, executor, records, costs))
    })
}

/// The requests a job measures: those of the executor on the topology that
/// are outstanding beside the records, given the costs' outstanding walk.
type Requests<'a> = &'a dyn Fn(
    &seismic::DeviceTopology,
    &seismic::Executor,
    &dyn RecordStore<PointKey>,
    Option<CostWalk>,
) -> Result<magnitude_executor::assessment::Outstanding, String>;

/// A measurement job of `requests` ([`measure_machine`]).
fn measure_job(request: &MeasureRequest, report: &mut dyn FnMut(MeasureEvent), requests: Requests<'_>) -> MeasureEnd {
    // Probes begin within a probe's limit of the job's start, so the job
    // opens within its host's `OPENING_LIMIT`.
    let probes_begin_until = std::time::Instant::now() + seismic::PROBE_LIMIT;
    let machine = |reason: String| MeasureEnd {
        failures: vec![MeasureFailure::Machine { reason }],
        held: None,
    };
    let catalog = match Devices::discover(&request.paths) {
        Ok(catalog) => catalog,
        Err(error) => return machine(error.to_string()),
    };
    // The device forms through the process's one store, as a load's does: a
    // program this job or a load compiled is loaded, not compiled again. It
    // holds no limit: the walk passes over its paths by holding them. Its
    // property probes run first, each in a process of its own, and keep
    // their results in the same store, so the device opens with them, as
    // every later load does; a probe that keeps nothing is reported, its
    // property at its safe value. A probe begins only within a probe's limit
    // of the job's start, and one begun runs to its own end; one not begun
    // leaves its property for the next job. None of it is the job's time.
    let mut probing = Vec::new();
    // An opening the device refused as busy (out of memory, or held by
    // another process) is the device's for now, not the machine's.
    let opening = |error: seismic::OpenError| match error {
        seismic::OpenError::Backend(seismic::TargetError::DeviceBusy(_)) => MeasureFailure::Busy { reason: error.to_string() },
        error => MeasureFailure::Machine { reason: error.to_string() },
    };
    let device = match catalog
        .resolve(request.executor)
        .map_err(|error| MeasureFailure::Machine { reason: error.to_string() })
        .and_then(|executor| {
            let probes = catalog
                .probe(executor, &*seismic::ProgramStore::user(), probes_begin_until)
                .map_err(opening)?;
            for kept in &probes.kept {
                tracing::info!(probe = kept.probe, result = %kept.result.line(), report = %kept.report, "probed the device");
            }
            probing.extend(probes.failures.iter().map(|failure| MeasureFailure::Probe { reason: failure.to_string() }));
            catalog
                .open_with(
                    executor,
                    seismic::DeviceOptions {
                        artifacts: None,
                        // The job's point of it (`PointRequest::GraphCost`).
                        graph_cost: None,
                    },
                )
                .map_err(opening)
        }) {
        Ok(device) => device,
        Err(failure) => return MeasureEnd { failures: vec![failure], held: None },
    };
    report(MeasureEvent::Opened);
    // The job's time runs from its device's opening, as its host's does:
    // discovering the machine, probing and opening it take none of it.
    let deadlines = requests::JobDeadlines::new(std::time::Instant::now(), request.time);
    let records = KernelCache::measurement_records(&request.kernels);
    let (topology, executor) = (device.topology(), device.info());
    let costs = StoredCosts::outstanding_of(&ExecutorCosts::stored(topology, executor, &records));
    let outstanding = requests(topology, executor, &records, costs);
    let outstanding = match outstanding {
        Ok(outstanding) => outstanding,
        Err(reason) => return machine(format!("the point requests: {reason}")),
    };
    let markers = AttemptMarkers::new(&request.kernels);
    let unrecorded = RefCell::new(Vec::new());
    let mut measurer = Measurer::new(
        magnitude_executor::SEARCH_SETTINGS.error_rate,
        magnitude_executor::MIN_SAMPLE_SECONDS,
    );
    // Reports attempts and records as the walk goes; a RefCell as both the
    // attempting and the visiting closures report.
    let report = RefCell::new(report);
    let mut failures = probing;
    // The job's process opened its device on tallies of its own
    // (`Devices::open_with`): every byte it allocates is the job's.
    let measured = requests::measure(
        &mut measurer,
        &seismic::Charges::tally(&device),
        &records,
        &outstanding,
        deadlines,
        &mut |key| {
            // A point that ended two jobs is not timed again; the measurer
            // records it unobtainable with this reason.
            match markers.unobtainable(key) {
                Ok(Some(reason)) => return Attempt::Skip(Reason::new(reason)),
                Ok(None) => {}
                Err(error) => unrecorded.borrow_mut().push(error),
            }
            // Reported before it is recorded, so a host that kills the job
            // knows every attempt the job may have counted.
            (report.borrow_mut())(MeasureEvent::Attempting(key.clone()));
            if let Err(error) = markers.attempting(key, request.job) {
                unrecorded.borrow_mut().push(error);
            }
            Attempt::Proceed
        },
        &mut |key, outcome| {
            let record = match outcome {
                Outcome::Loaded(_) => None,
                Outcome::Measured(timing) => Some(Record::Measured(timing)),
                Outcome::Skipped(reason) => Some(Record::Unobtainable { reason }),
                // Formation is deterministic for its key: the measurer
                // recorded it unobtainable with the failure, and the job
                // reports it too.
                Outcome::Unformed(error) => {
                    let reason = error.to_string();
                    failures.push(MeasureFailure::Point {
                        key: key.clone(),
                        reason: reason.clone(),
                    });
                    Some(Record::Unobtainable {
                        reason: Reason::new(reason),
                    })
                }
                // Left unrecorded, with its attempt still counted as one
                // ended job: the next job takes it, and only the two-job rule
                // makes the failure permanent.
                Outcome::Unmade(error) => {
                    failures.push(MeasureFailure::Point {
                        key,
                        reason: error.to_string(),
                    });
                    return;
                }
                Outcome::Failed(error) => {
                    failures.push(MeasureFailure::Point {
                        key,
                        reason: error.to_string(),
                    });
                    return;
                }
                // The job's time ended before the point was begun: it was
                // never attempted, and the next job takes it.
                Outcome::Expired => return,
                Outcome::Store(error) => {
                    failures.push(MeasureFailure::Point {
                        key: key.clone(),
                        reason: error.to_string(),
                    });
                    None
                }
            };
            if let Err(error) = markers.resolved(&key) {
                unrecorded.borrow_mut().push(error);
            }
            if let Some(record) = record {
                (report.borrow_mut())(MeasureEvent::Recorded(key, record));
            }
        },
    );
    let measurement = match measured {
        Ok(measurement) => measurement,
        Err(error) => return machine(error.to_string()),
    };
    failures.extend(measurement.ended.into_iter().map(|error| MeasureFailure::Machine {
        reason: error.to_string(),
    }));
    failures.extend(
        measurement
            .failures
            .into_iter()
            .map(|error| MeasureFailure::Figure {
                reason: error.to_string(),
            }),
    );
    failures.extend(
        unrecorded
            .into_inner()
            .into_iter()
            .map(|error| MeasureFailure::Attempt {
                reason: error.to_string(),
            }),
    );
    MeasureEnd {
        failures,
        held: Some(measurement.held),
    }
}

/// Run the measurement job of `request` in the worker at the other end of
/// `transport`, passing each attempt and each record it stores to `report`
/// as the worker reports it. Returns what the job could not measure, or how
/// it was lost. Blocking until the worker reports its outcome or its stream
/// ends: the caller keeps the job's hang guard ([`hang_silence`]) over the
/// events and ends the process.
pub fn measure_worker(
    transport: impl HostTransport,
    request: MeasureRequest,
    report: impl FnMut(MeasureEvent),
) -> Result<Vec<MeasureFailure>, JobLost> {
    observed(transport, Some(HostMessage::Measure { request }), report)
}

/// [`measure_worker`] for a job its process was given directly, which
/// reports over `transport` ([`report_measurement`]): a tool's own job
/// process.
pub fn observe_measurement(
    transport: impl HostTransport,
    report: impl FnMut(MeasureEvent),
) -> Result<Vec<MeasureFailure>, JobLost> {
    observed(transport, None, report)
}

/// A job's events read off `transport`, once its worker greeted this build
/// (and was sent `first`, where it takes its job from its host), each
/// reported as it comes, until its outcome.
fn observed(
    transport: impl HostTransport,
    first: Option<HostMessage>,
    mut report: impl FnMut(MeasureEvent),
) -> Result<Vec<MeasureFailure>, JobLost> {
    let (mut receiver, mut sender) = transport.split();
    let ended = "it reported its outcome";
    let lost = |lost: Lost| JobLost(lost.reason);
    handshake(&mut sender, &mut receiver, first, ended, lost)?;
    // The device opens once, before any point: a host reads a job that
    // reported no opening as one that could not open the machine.
    let mut opened = false;
    loop {
        match receive(&mut receiver, ended, lost)? {
            WorkerMessage::Opened if !opened => {
                opened = true;
                report(MeasureEvent::Opened);
            }
            WorkerMessage::Attempting { key } if opened => report(MeasureEvent::Attempting(key)),
            WorkerMessage::Recorded { key, record } if opened => report(MeasureEvent::Recorded(key, record)),
            WorkerMessage::Attempting { .. } | WorkerMessage::Recorded { .. } => {
                return Err(JobLost("a point was reported before the device opened".into()))
            }
            WorkerMessage::MeasureFinished { failures } => return Ok(failures),
            _ => return Err(JobLost("unexpected message during measurement".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A point a job ends on once may be attempted again; the second job it
    /// ends makes it unobtainable. A point that resolves clears the count,
    /// and every key counts on its own.
    #[test]
    fn a_key_that_ends_two_jobs_becomes_unobtainable() {
        let kernels = tempfile::tempdir().unwrap();
        let markers = AttemptMarkers::new(kernels.path());
        let key = "cuda driver 12.8: launch chain";
        // The first job ends while timing the point.
        markers.attempting(&key, MeasureJob(1)).unwrap();
        assert_eq!(markers.unobtainable(&key).unwrap(), None);
        // The second resolves it: the first was a transient.
        markers.attempting(&key, MeasureJob(2)).unwrap();
        markers.resolved(&key).unwrap();
        assert_eq!(markers.unobtainable(&key).unwrap(), None);
        // Two jobs end on it in a row.
        markers.attempting(&key, MeasureJob(3)).unwrap();
        let markers = AttemptMarkers::new(kernels.path());
        markers.attempting(&key, MeasureJob(4)).unwrap();
        let reason = markers.unobtainable(&key).unwrap();
        assert!(reason.is_some_and(|reason| reason.contains("ended 2 measurement jobs")));
        // Another driver's key is its own.
        let other = "cuda driver 13.0: launch chain";
        assert_eq!(markers.unobtainable(&other).unwrap(), None);
    }

    /// A killed job's attempt is released, not counted: only the job's
    /// own attempt, and an earlier job's ended attempt still counts.
    #[test]
    fn a_killed_attempt_is_released() {
        let kernels = tempfile::tempdir().unwrap();
        let markers = AttemptMarkers::new(kernels.path());
        let key = "vulkan driver 580: composition chain";
        // A job ends on the point, then the next is killed on it twice.
        markers.attempting(&key, MeasureJob(1)).unwrap();
        markers.attempting(&key, MeasureJob(2)).unwrap();
        markers.release(&key, MeasureJob(2)).unwrap();
        // Releasing again, or another job's release, changes nothing.
        markers.release(&key, MeasureJob(2)).unwrap();
        markers.release(&key, MeasureJob(3)).unwrap();
        markers.attempting(&key, MeasureJob(3)).unwrap();
        markers.release(&key, MeasureJob(3)).unwrap();
        assert_eq!(markers.record(&key).unwrap().ended_jobs, 1);
        assert_eq!(markers.unobtainable(&key).unwrap(), None);
    }

    /// In-process jobs on the host's CPU executor, their records and their
    /// model store in `root`, the first job given no time.
    fn cut_off_jobs(root: &Path) -> InProcessJobs {
        let topology = Devices::discover(&[]).unwrap().topology();
        let executor = topology.executors().iter().find(|executor| executor.backend() == seismic::BackendName::Cpu).unwrap().clone();
        InProcessJobs {
            request: MeasureRequest {
                paths: Vec::new(),
                kernels: root.to_path_buf(),
                storage: root.to_path_buf(),
                executor: executor.selector(),
                job: MeasureJob(0),
                time: std::time::Duration::ZERO,
            },
            topology,
            executor,
            #[cfg(feature = "machine-line")]
            line: false,
        }
    }

    /// A job given no time stores nothing: it begins no point, leaving every
    /// one unrecorded, and reports it measured nothing, with no failure of
    /// its own. The next, given a job's time, begins its points and measures.
    #[test]
    fn a_job_given_no_time_leaves_every_point_to_the_next() {
        let root = tempfile::tempdir().unwrap();
        let mut jobs = cut_off_jobs(root.path());
        match jobs.run() {
            Err(JobsStopped::Stalled { failures, .. }) => {
                assert!(failures.iter().all(|failure| !failure.fails_job()), "{failures:?}")
            }
            Err(stopped) => panic!("{stopped}"),
            Ok(_) => panic!("a job given no time measured"),
        }
        jobs.request.time = job_time();
        let run = jobs.run().unwrap();
        assert_eq!(jobs.request.job, MeasureJob(2), "the cut-off job stored nothing, and the next measured");
        assert!(run.failures.iter().all(|failure| !failure.fails_job()), "{:?}", run.failures);
    }

    /// Requests and outcomes cross the process encoding.
    #[test]
    fn measurement_messages_round_trip_through_the_worker_encoding() {
        let request = MeasureRequest {
            paths: vec!["/models".into(), "/cache".into()],
            kernels: "/cache/kernels".into(),
            storage: "/models".into(),
            executor: seismic::ExecutorSelector::Cuda { uuid: [7; 16] },
            job: MeasureJob(9),
            time: std::time::Duration::from_millis(9_500),
        };
        let encoded = postcard::to_allocvec(&HostMessage::Measure {
            request: request.clone(),
        })
        .unwrap();
        let HostMessage::Measure { request: decoded } = postcard::from_bytes(&encoded).unwrap()
        else {
            panic!("the request changed shape")
        };
        assert_eq!(decoded, request);
        let failures = vec![
            MeasureFailure::Machine {
                reason: "no executor".into(),
            },
            MeasureFailure::Attempt {
                reason: "read-only cache".into(),
            },
            MeasureFailure::Figure {
                reason: "no implementation".into(),
            },
        ];
        let encoded = postcard::to_allocvec(&WorkerMessage::MeasureFinished {
            failures: failures.clone(),
        })
        .unwrap();
        let WorkerMessage::MeasureFinished { failures: decoded } =
            postcard::from_bytes(&encoded).unwrap()
        else {
            panic!("the outcome changed shape")
        };
        assert_eq!(decoded, failures);
    }
}
