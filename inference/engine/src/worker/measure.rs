//! The measurement job (device topology spec §2.15): a mode of the worker
//! process that measures this machine for assessment, so that nothing in the
//! host process opens an executor or waits on a native submission.
//!
//! The job discovers the machine, opens the selected executor and measures
//! the point requests outstanding on it in the order of who waits on them
//! ([`magnitude_executor::assessment::requests`]: the load that waited
//! latest first, then assessment's costs), through `seismic-measure`, each
//! record stored in the kernel cache as it is taken, and ends with the
//! points it could not measure. Before each point it records the key it is
//! attempting ([`AttemptMarkers`]): a point that ends two jobs (the second
//! confirms the first was no transient) is recorded unobtainable with its
//! reason and not measured again while the hardware and driver it names are
//! the machine's; the costs complete without it, and only estimates that
//! need it are unavailable. A point whose timing fails is left unrecorded
//! and counts as one ended job, so only this rule makes a timing failure
//! permanent, as does a point whose operands could not be made now; a
//! kernel that cannot be formed is recorded unobtainable at once, as
//! formation is deterministic for its key.

use super::client::{handshake, receive};
use super::protocol::{EngineBuild, HostMessage, WorkerMessage};
use super::transport::{HostTransport, MessageSender};
use super::{Held, Served, WorkerExit};
use magnitude_executor::assessment::{requests, ExecutorCosts, PointRequests};
use magnitude_executor::{KernelCache, DEFAULT_KERNEL_CACHE_BYTES};
use seismic::measure::{Attempt, JsonFileStore, Outcome, Reason, Record, StoreError};
use seismic::{Devices, KernelIdentity, Measurer, PointKey};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The jobs a point may end before it is unobtainable: the second confirms
/// the first was not a transient.
const ENDED_JOBS: u32 = 2;

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
    /// its host kills is released rather than counted ([`release_killed`]).
    pub job: MeasureJob,
    /// The time the job has from its start: what its host's time for it
    /// ([`magnitude_executor::assessment::MEASUREMENT_SECONDS`]) has left,
    /// the bound however many points are requested: the order, not a
    /// larger budget, measures what loads wait on first.
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
    /// It stored the point's record.
    Recorded(PointKey, Record),
}

/// What a job could not measure; everything else it measured is stored.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasureFailure {
    /// The machine could not be discovered, or the selected executor opened.
    Machine { reason: String },
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
            Self::Machine { reason } => write!(formatter, "the machine: {reason}"),
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

    fn store(&self, key: &impl Serialize, record: &AttemptRecord) -> Result<(), StoreError> {
        self.files.store(key, record, LARGEST_RECORD)
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
        let record = self.record(key)?;
        self.store(
            key,
            &AttemptRecord {
                ended_jobs: record.ended_jobs.saturating_add(1),
                job: Some(job),
            },
        )
    }

    /// Undo `job`'s attempt of `key`, where it is the attempt counted last:
    /// the job was killed, or its time ended, which says nothing of the
    /// point.
    fn release(&self, key: &impl Serialize, job: MeasureJob) -> Result<(), StoreError> {
        let record = self.record(key)?;
        if record.job != Some(job) {
            return Ok(());
        }
        self.store(
            key,
            &AttemptRecord {
                ended_jobs: record.ended_jobs.saturating_sub(1),
                job: None,
            },
        )
    }

    /// Record that `key`'s point resolved, with a record: the job did not
    /// end on it.
    fn resolved(&self, key: &impl Serialize) -> Result<(), StoreError> {
        if self.record(key)?.ended_jobs == 0 {
            return Ok(());
        }
        self.store(key, &AttemptRecord::default())
    }
}

/// Release `job`'s attempt of `key` in the kernel cache at `kernels`: its
/// host killed the job while timing it (for a load, or as its time ended),
/// so the point is attempted again uncounted. Only `job`'s own attempt is
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
pub(super) fn measure_only(
    request: MeasureRequest,
    mut sender: impl MessageSender<WorkerMessage>,
) -> Served {
    let hello = WorkerMessage::Hello {
        build: EngineBuild::current(),
    };
    if let Err(error) = sender.send(hello) {
        return WorkerExit::TransportFailed(error.to_string()).into();
    }
    let mut lost = None;
    let MeasureEnd { failures, held } = measure_machine(&request, &mut |event| {
        if lost.is_none() {
            let message = match event {
                MeasureEvent::Attempting(key) => WorkerMessage::Attempting { key },
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
/// with what its points hold (`held`). It ends at once, what they hold
/// released by the system, as by a kill, with no teardown and no wait for
/// compiles still forming, except where what the job's compiles formed is
/// kept in the platform's cache (`Held::platform_cached`): that process
/// drops what its points hold, so the compiles it began end, and exits
/// normally, its exit handlers leaving the platform's cache written, as a
/// Prepare's does. A job that held nothing exits normally.
pub fn end_measured(held: Option<magnitude_executor::assessment::Held>, code: i32) -> ! {
    match held {
        Some(points) if points.platform_cached() => {
            drop(points);
            std::process::exit(code)
        }
        Some(_) => magnitude_service_utils::process::end(code),
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

/// The measurement job's probing of the device alone, as a job probes before
/// it measures ([`measure_machine`]): executor `executor` of the machine
/// discovered with `paths` runs each of its probes the kernel cache at
/// `kernels` holds no result of, in a process of its own, within `time`. A
/// tool whose check reads probed properties runs it before the load it
/// checks; a load never does.
pub fn probe_machine(
    paths: &[PathBuf],
    kernels: &Path,
    executor: seismic::ExecutorSelector,
    time: std::time::Duration,
) -> Result<seismic::ProbeRun, String> {
    let deadline = std::time::Instant::now() + time;
    let catalog = Devices::discover(paths).map_err(|error| error.to_string())?;
    let cache = KernelCache::open(kernels.to_path_buf(), DEFAULT_KERNEL_CACHE_BYTES).map_err(|error| error.to_string())?;
    let executor = catalog.resolve(executor).map_err(|error| error.to_string())?;
    catalog.probe(executor, &cache, deadline).map_err(|error| error.to_string())
}

/// Measure the outstanding point requests of the request's executor, in
/// the order of who waits on them ([`PointRequests::outstanding`]), within
/// the request's time: what it does not reach is left for the next
/// job, its attempts released. A point that ended two jobs is not timed: it
/// is recorded unobtainable, so the executor's costs complete without it. Each attempt reaches `report`
/// before it is recorded, and each record as soon as the job stores it.
/// Returns what could not be measured, and what the job's points were made
/// in and from. The worker's measure mode runs it; a tool may run it in its
/// own process.
pub fn measure_machine(
    request: &MeasureRequest,
    report: &mut dyn FnMut(MeasureEvent),
) -> MeasureEnd {
    // The job's time runs from its start: discovering the machine and
    // opening the executor are part of it.
    let deadline = std::time::Instant::now() + request.time;
    let machine = |reason: String| MeasureEnd {
        failures: vec![MeasureFailure::Machine { reason }],
        held: None,
    };
    let catalog = match Devices::discover(&request.paths) {
        Ok(catalog) => catalog,
        Err(error) => return machine(error.to_string()),
    };
    // The device forms through the kernel cache's artifact store, as a load's
    // does: a program this job or a load compiled is loaded, not compiled
    // again. It holds no limit: the walk passes over its paths by holding
    // them. Its property probes run first, each in a process of its own,
    // and keep their results in the same store, so the device opens with
    // them, as every later load does; a probe that keeps nothing is
    // reported, its property at its safe value.
    let mut probing = Vec::new();
    let device = match KernelCache::open(request.kernels.clone(), DEFAULT_KERNEL_CACHE_BYTES)
        .map_err(|error| error.to_string())
        .and_then(|cache| {
            let executor = catalog
                .resolve(request.executor)
                .map_err(|error| error.to_string())?;
            let probes = catalog.probe(executor, &cache, deadline).map_err(|error| error.to_string())?;
            for kept in &probes.kept {
                tracing::info!(probe = kept.probe, result = %kept.result.line(), report = %kept.report, "probed the device");
            }
            probing.extend(probes.failures.iter().map(|failure| MeasureFailure::Probe { reason: failure.to_string() }));
            catalog
                .open_with(
                    executor,
                    seismic::DeviceOptions {
                        artifacts: Some(Arc::new(cache)),
                        // The job's point of it (`PointRequest::GraphCost`).
                        graph_cost: None,
                    },
                )
                .map_err(|error| error.to_string())
        }) {
        Ok(device) => device,
        Err(reason) => return machine(reason),
    };
    let records = KernelCache::measurement_records(&request.kernels);
    let (topology, executor) = (device.topology(), device.info());
    let costs = ExecutorCosts::stored(topology, executor, &records).ok().and_then(|walked| walked.outstanding());
    let outstanding = PointRequests::new(&request.kernels).outstanding(topology, executor, &request.storage, &records, costs);
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
    let measured = requests::measure(
        &mut measurer,
        &device,
        &records,
        &outstanding,
        deadline,
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
                // The job's time ended before the point read a sample: no
                // failure of it, so its attempt is released, as a killed
                // job's is, and the next job takes it.
                Outcome::Expired => {
                    if let Err(error) = markers.release(&key, request.job) {
                        unrecorded.borrow_mut().push(error);
                    }
                    return;
                }
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
/// ends: the caller owns the deadline and ends the process.
pub fn measure_worker(
    transport: impl HostTransport,
    request: MeasureRequest,
    mut report: impl FnMut(MeasureEvent),
) -> Result<Vec<MeasureFailure>, JobLost> {
    let (mut receiver, mut sender) = transport.split();
    let ended = "it reported its outcome";
    let measure = HostMessage::Measure { request };
    handshake(&mut sender, &mut receiver, Some(measure), ended, JobLost)?;
    loop {
        match receive(&mut receiver, ended, JobLost)? {
            WorkerMessage::Attempting { key } => report(MeasureEvent::Attempting(key)),
            WorkerMessage::Recorded { key, record } => report(MeasureEvent::Recorded(key, record)),
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
