//! The measurement job's owner (device topology spec §2.15). Measuring opens executors and runs
//! kernels, which can fault or hang in a driver, and a native submission cannot be cancelled once
//! started; so the service never measures in its own process and nothing waits on a measurement.
//! The job is a mode of the inference worker, run only with the device to itself
//! ([`DeviceLock::try_measure`]) and held to a hard deadline, after which it is killed and waited
//! for a bounded time on a detached task. A load or catalog optimization wanting the device
//! preempts it: the job is killed at once, its attempts are released rather than counted, and the
//! next job resumes from what is stored. Its state moves `Unmeasured → Measuring → Measured |
//! Failed { retry_at }` (a preempted job back to `Unmeasured`), and what it stores reaches the
//! assessment pool, which re-finishes every assessed model under it.

use std::sync::Arc;
use std::time::Duration;

use magnitude_engine::assessment::{AssessmentStructure, Measurements};
use magnitude_engine::worker::measure::{
    JobLost, MeasureEvent, MeasureFailure, MeasureJob, MeasureRequest, measure_worker,
    release_preempted,
};
use magnitude_service_contracts::models::MeasurementState;
use seismic::PointKey;
use seismic::measure::Record;
use tokio::sync::{mpsc, watch};

use crate::device_lock::{DeviceLock, DeviceMeasurement};
use crate::residency::controller::GRACEFUL_EXIT;
use crate::residency::worker::{JobWorker, KILL_REAP};
use crate::worker_process::WorkerLauncher;

/// The slowest cold job measured: 10.65 s, every resident weight format on a GB10 through
/// NVIDIA's Vulkan driver, with an empty kernel cache and the driver's shader cache off
/// (2026-10-20, run 20261020T204050Z-merge-kernels-gb10-46655c). The same machine's CUDA job
/// took 7.2 s, an RTX 3090's 8.7 s through Vulkan and 7.8 s through CUDA, and an M4 Pro's 7.3 s.
const SLOWEST_COLD_JOB: Duration = Duration::from_millis(10_648);

/// How long a job may run before it is killed as hung: twice the slowest cold job measured.
/// The job itself has no budget: each point stops when its median is resolved. A load never
/// waits on it: a load preempts a running job.
pub const JOB_DEADLINE: Duration = SLOWEST_COLD_JOB.saturating_mul(2);

/// How long a job's process is waited for once it reported or was killed: a reported job's own
/// exit ([`GRACEFUL_EXIT`]), then the kill's reap ([`KILL_REAP`]), the two waits its retirement
/// makes. Past it the wait goes on detached and the device is released, even though the process
/// is not proven gone.
const EXIT_WAIT: Duration = GRACEFUL_EXIT.saturating_add(KILL_REAP);

/// The first retry's delay after a failed job: twice [`JOB_DEADLINE`], so a job that fails
/// every time occupies the device at most a third of the time. Each further failure doubles it,
/// up to [`LAST_RETRY`].
const FIRST_RETRY: Duration = JOB_DEADLINE.saturating_mul(2);
/// The longest retry delay: a failure is retried at least hourly while the service runs.
const LAST_RETRY: Duration = Duration::from_secs(60 * 60);

/// What the measurement job reports: its state, and what is stored of the machine for
/// assessment.
#[derive(Clone)]
pub struct MeasurementStatus<S = Measurements> {
    pub state: MeasurementState,
    pub measurements: S,
}

impl<S: StoredRecords> MeasurementStatus<S> {
    /// The status of `stored`, what is stored of the machine, before any job runs: `Measured`
    /// when nothing is pending, as no job will run.
    pub fn stored(stored: S) -> Self {
        let state = if stored.pending() {
            MeasurementState::Unmeasured
        } else {
            MeasurementState::Measured
        };
        Self {
            state,
            measurements: stored,
        }
    }
}

/// What the job reads of what is stored: whether a job still has something to measure.
pub trait StoredRecords: Clone + Send + Sync + 'static {
    fn pending(&self) -> bool;
}

impl StoredRecords for Measurements {
    fn pending(&self) -> bool {
        match self {
            Self::Pending => true,
            Self::Unavailable { .. } | Self::Measured { .. } => false,
        }
    }
}

/// What a job's reader passes on, in order: each point the job attempts and each record it
/// stores, then the outcome.
pub enum JobFrame {
    Attempting(PointKey),
    Recorded(PointKey, Record),
    Finished(Result<Vec<MeasureFailure>, JobLost>),
}

/// A started job's process.
pub trait JobProcess: Send + 'static {
    /// Return once the process is gone. One that `reported` its outcome is given
    /// [`GRACEFUL_EXIT`] to exit on its own, as a reported Prepare is; any other is killed.
    /// Blocking.
    fn retire(self: Box<Self>, reported: bool) -> std::io::Result<()>;
    /// The process's recent standard error, for a lost job's report.
    fn diagnostics(&self) -> String;
}

/// A running job: its identity, its reader's frames and its process.
pub struct StartedJob {
    pub job: MeasureJob,
    pub frames: mpsc::Receiver<JobFrame>,
    pub process: Box<dyn JobProcess>,
}

/// The machine the job measures: what is stored of it for assessment, and a job that measures
/// the rest.
pub trait MeasuredMachine: Send + Sync + 'static {
    /// What assessment reads of the stored records.
    type Stored: StoredRecords;
    /// What the stored records give assessment now. Blocking; measures nothing.
    fn stored(&self) -> Self::Stored;
    /// Start a measurement job. Blocking.
    fn start(&self) -> anyhow::Result<StartedJob>;
    /// Release preempted `job`'s attempts of `keys`, so none is counted. Blocking.
    fn release_preempted(&self, job: MeasureJob, keys: &[PointKey]) -> anyhow::Result<()>;
}

/// This service's machine: the assessment structure and the job it runs in an inference worker
/// process.
pub struct ServiceMachine {
    pub structure: Arc<AssessmentStructure>,
    pub launcher: WorkerLauncher,
    /// The paths the service discovered with, which the job discovers with.
    pub served: Vec<std::path::PathBuf>,
    /// The kernel cache the job stores its records and attempts in.
    pub kernels: std::path::PathBuf,
}

impl ServiceMachine {
    /// The job measuring `structure`'s selected executor into the kernel cache `kernels`,
    /// discovering with the paths `served`, which the service discovered with.
    pub fn new(
        structure: Arc<AssessmentStructure>,
        launcher: WorkerLauncher,
        served: Vec<std::path::PathBuf>,
        kernels: std::path::PathBuf,
    ) -> Self {
        Self {
            structure,
            launcher,
            served,
            kernels,
        }
    }
}

/// A job identity distinct from the jobs before it: the time it starts, in nanoseconds since the
/// Unix epoch. Jobs on a device run one at a time, each after the last ended, so their start
/// times differ unless the system clock is set back (a clock before the epoch reads as the
/// epoch).
fn next_job() -> MeasureJob {
    let since = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    MeasureJob(u64::try_from(since.as_nanos()).unwrap_or(u64::MAX))
}

impl MeasuredMachine for ServiceMachine {
    type Stored = Measurements;

    fn stored(&self) -> Measurements {
        self.structure.stored_measurements(&self.kernels)
    }

    fn start(&self) -> anyhow::Result<StartedJob> {
        let job = next_job();
        let request = MeasureRequest {
            paths: self.served.clone(),
            kernels: self.kernels.clone(),
            executor: self.structure.selected.info.selector(),
            job,
        };
        let (worker, transport) = JobWorker::spawn(&self.launcher)?;
        let (frames, receiver) = mpsc::channel(64);
        let reader = std::thread::Builder::new()
            .name(format!("measurement-job-{}", worker.pid()))
            .spawn(move || {
                // A driver past its deadline has stopped reading: what is left is dropped.
                let outcome = measure_worker(transport, request, |event| {
                    let _ = frames.blocking_send(match event {
                        MeasureEvent::Attempting(key) => JobFrame::Attempting(key),
                        MeasureEvent::Recorded(key, record) => JobFrame::Recorded(key, record),
                    });
                });
                let _ = frames.blocking_send(JobFrame::Finished(outcome));
            });
        if let Err(error) = reader {
            worker.retire(false)?;
            return Err(error.into());
        }
        Ok(StartedJob {
            job,
            frames: receiver,
            process: Box::new(worker),
        })
    }

    fn release_preempted(&self, job: MeasureJob, keys: &[PointKey]) -> anyhow::Result<()> {
        for key in keys {
            release_preempted(&self.kernels, job, key)?;
        }
        Ok(())
    }
}

impl JobProcess for JobWorker {
    fn retire(self: Box<Self>, reported: bool) -> std::io::Result<()> {
        JobWorker::retire(*self, reported).map(drop)
    }

    fn diagnostics(&self) -> String {
        JobWorker::diagnostics(self)
    }
}

/// Start owning the measurement job of `machine` on `device`, from `stored`, what was stored of
/// it at start. The returned status is the job's, from [`MeasurementStatus::stored`] on.
pub fn start<M: MeasuredMachine>(
    machine: Arc<M>,
    stored: M::Stored,
    device: DeviceLock,
) -> watch::Receiver<MeasurementStatus<M::Stored>> {
    let (status, receiver) = watch::channel(MeasurementStatus::stored(stored));
    tokio::spawn(run(machine, device, status, JOB_DEADLINE));
    receiver
}

/// What the stored records give assessment, read on the blocking pool; none when that task
/// failed, which is reported.
async fn read_stored<M: MeasuredMachine>(machine: &Arc<M>) -> Option<M::Stored> {
    let machine = Arc::clone(machine);
    match crate::spawn_blocking_traced(move || machine.stored()).await {
        Ok(stored) => Some(stored),
        Err(error) => {
            tracing::error!(%error, "the task reading the stored measurement records failed");
            None
        }
    }
}

/// Run jobs until what assessment reads is stored: the executor's fixed cost points, whatever
/// the models assessed. Each job runs only with the device to itself, deferring while residency
/// shares it until a share ends ([`DeviceLock::released`]), preempted when residency wants it
/// back, and retried after a failure with a backoff that doubles. A preempted job is no
/// failure: the next resumes from what it stored. The hardware and driver identities are the
/// inventory's revision, fixed for the service's life, so the backoff is this process's and a
/// changed identity is the next process's discovery, starting anew.
async fn run<M: MeasuredMachine>(
    machine: Arc<M>,
    device: DeviceLock,
    status: watch::Sender<MeasurementStatus<M::Stored>>,
    deadline: Duration,
) {
    let mut backoff = FIRST_RETRY;
    while status.borrow().measurements.pending() {
        // Every share's end wakes this, after its release: a failed try is always followed by
        // the end of a share it saw.
        let exclusive = loop {
            if let Some(exclusive) = device.try_measure() {
                break exclusive;
            }
            device.released().await;
        };
        status.send_modify(|status| status.state = MeasurementState::Measuring);
        let outcome = match run_job(&machine, exclusive, deadline).await {
            JobEnd::Ended(outcome) => outcome,
            JobEnd::Preempted => {
                status.send_modify(|status| status.state = MeasurementState::Unmeasured);
                continue;
            }
        };
        match &outcome {
            Ok(failures) => {
                for failure in failures {
                    tracing::warn!(%failure, "the measurement job could not measure");
                }
            }
            Err(error) => tracing::warn!(%error, "the measurement job was lost"),
        }
        // A point still without a record fails the job: the next one takes it.
        match read_stored(&machine).await {
            Some(measurements) if !measurements.pending() => {
                backoff = FIRST_RETRY;
                status.send_replace(MeasurementStatus {
                    state: MeasurementState::Measured,
                    measurements,
                });
            }
            _ => retry(&status, &mut backoff).await,
        }
    }
    // The last state stands for as long as anything reads it.
    status.closed().await;
}

/// Report the job failed with what is stored unchanged, and wait out the backoff, which then
/// doubles.
async fn retry<S: StoredRecords>(
    status: &watch::Sender<MeasurementStatus<S>>,
    backoff: &mut Duration,
) {
    let retry_at = std::time::SystemTime::now() + *backoff;
    status.send_modify(|status| {
        status.state = MeasurementState::Failed {
            retry_at: epoch_millis(retry_at),
        }
    });
    tokio::time::sleep(*backoff).await;
    *backoff = backoff.saturating_mul(2).min(LAST_RETRY);
}

/// How one job ended.
enum JobEnd {
    /// It reported its outcome, or was lost.
    Ended(Result<Vec<MeasureFailure>, JobLost>),
    /// Residency wanted the device: the job was killed, and its attempts released.
    Preempted,
}

/// One job held to `deadline` on `exclusive`: its frames are read until it
/// reports its outcome, or until residency wants the device, which kills it at once; its process
/// is then retired, waiting at most [`EXIT_WAIT`] on a detached task, and the device released. A
/// preempted job's attempts are then released, so preemption never counts against a point.
async fn run_job<M: MeasuredMachine>(
    machine: &Arc<M>,
    mut exclusive: DeviceMeasurement,
    deadline: Duration,
) -> JobEnd {
    let started = {
        let machine = Arc::clone(machine);
        crate::spawn_blocking_traced(move || machine.start()).await
    };
    let StartedJob {
        job,
        mut frames,
        process,
    } = match started {
        Ok(Ok(started)) => started,
        Ok(Err(error)) => {
            return JobEnd::Ended(Err(JobLost(format!("starting the job: {error:#}"))));
        }
        Err(error) => return JobEnd::Ended(Err(JobLost(format!("starting the job: {error}")))),
    };
    let started = tokio::time::Instant::now();
    let end = started + deadline;
    // Every point the job reported attempting: those a preempted job's markers may count.
    let mut attempted = Vec::new();
    let outcome = loop {
        tokio::select! {
            frame = frames.recv() => match frame {
                Some(JobFrame::Attempting(key)) => attempted.push(key),
                Some(JobFrame::Recorded(key, record)) => log_record(&key, &record),
                Some(JobFrame::Finished(outcome)) => break Some(outcome),
                None => break Some(Err(JobLost("the job's reader ended".into()))),
            },
            () = tokio::time::sleep_until(end) => {
                break Some(Err(JobLost(format!("the job did not finish within {deadline:?}"))));
            }
            () = exclusive.preempted() => break None,
        }
    };
    tracing::info!(
        elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        outcome = match &outcome {
            Some(Ok(_)) => "reported",
            Some(Err(_)) => "lost",
            None => "preempted",
        },
        "measurement job ended"
    );
    let outcome = outcome.map(|outcome| {
        outcome.map_err(|lost| match process.diagnostics() {
            diagnostics if diagnostics.is_empty() => lost,
            diagnostics => JobLost(format!("{lost}; worker diagnostics: {diagnostics}")),
        })
    });
    let reported = matches!(outcome, Some(Ok(_)));
    let retired = tokio::task::spawn_blocking(move || process.retire(reported));
    match tokio::time::timeout(EXIT_WAIT, retired).await {
        Ok(Ok(Ok(()))) => {}
        Ok(Ok(Err(error))) => {
            tracing::warn!(%error, "the measurement job could not be shown to have ended");
        }
        Ok(Err(error)) => {
            tracing::warn!(%error, "retiring the measurement job failed");
        }
        Err(_) => tracing::warn!(
            wait = ?EXIT_WAIT,
            "the measurement job did not end within its exit wait; it is left to end detached"
        ),
    }
    // The device is released: a waiting share proceeds.
    drop(exclusive);
    if let Some(outcome) = outcome {
        return JobEnd::Ended(outcome);
    }
    // The frames the job sent before it was killed end once its stream does; an attempt it
    // reported is one its markers may count. Read for at most the exit wait.
    let drained = tokio::time::timeout(EXIT_WAIT, async {
        while let Some(frame) = frames.recv().await {
            if let JobFrame::Attempting(key) = frame {
                attempted.push(key);
            }
        }
    })
    .await;
    if drained.is_err() {
        tracing::warn!(
            wait = ?EXIT_WAIT,
            "a preempted measurement job's stream did not end; attempts it sent after are counted"
        );
    }
    let released = {
        let machine = Arc::clone(machine);
        crate::spawn_blocking_traced(move || machine.release_preempted(job, &attempted)).await
    };
    match released {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            tracing::warn!(error = %format!("{error:#}"), "a preempted job's attempts stay counted");
        }
        Err(error) => tracing::warn!(%error, "releasing a preempted job's attempts failed"),
    }
    JobEnd::Preempted
}

fn log_record(key: &PointKey, record: &Record) {
    tracing::info!(key = ?key, record = ?record, "measurement recorded");
}

/// `time` in milliseconds since the Unix epoch, as the contract states times. A system clock
/// set before the epoch cannot state it: that is reported, and the time reads as the epoch.
fn epoch_millis(time: std::time::SystemTime) -> u64 {
    match time.duration_since(std::time::UNIX_EPOCH) {
        Ok(since) => u64::try_from(since.as_millis()).unwrap_or(u64::MAX),
        Err(error) => {
            tracing::error!(%error, "the system clock is set before the Unix epoch");
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Whether every point has a record: all the job reads of what is stored.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Complete(bool);

    impl StoredRecords for Complete {
        fn pending(&self) -> bool {
            !self.0
        }
    }

    const PENDING: Complete = Complete(false);
    const COMPLETE: Complete = Complete(true);

    /// A machine whose jobs are scripted: each start hands out the next job's frames, and
    /// `stored` reads what the jobs left.
    struct ScriptedMachine {
        jobs: Mutex<Vec<StartedJob>>,
        stored: Mutex<Complete>,
        /// What the reads find, one each, in order; the last stands after them.
        reads: Mutex<std::collections::VecDeque<Complete>>,
        started: AtomicUsize,
        /// Every preempted job's released attempts.
        released: Mutex<Vec<(MeasureJob, Vec<PointKey>)>>,
    }

    impl ScriptedMachine {
        fn new(jobs: Vec<StartedJob>, reads: Vec<Complete>) -> Arc<Self> {
            Arc::new(Self {
                jobs: Mutex::new(jobs),
                stored: Mutex::new(PENDING),
                reads: Mutex::new(reads.into()),
                started: AtomicUsize::new(0),
                released: Mutex::new(Vec::new()),
            })
        }
    }

    impl MeasuredMachine for ScriptedMachine {
        type Stored = Complete;

        fn stored(&self) -> Complete {
            if let Some(read) = self.reads.lock().unwrap().pop_front() {
                *self.stored.lock().unwrap() = read;
            }
            *self.stored.lock().unwrap()
        }

        fn start(&self) -> anyhow::Result<StartedJob> {
            self.started.fetch_add(1, Ordering::SeqCst);
            Ok(self.jobs.lock().unwrap().remove(0))
        }

        fn release_preempted(&self, job: MeasureJob, keys: &[PointKey]) -> anyhow::Result<()> {
            self.released.lock().unwrap().push((job, keys.to_vec()));
            Ok(())
        }
    }

    /// A process that records how it was retired.
    struct RecordedProcess(Arc<Mutex<Option<bool>>>);

    impl JobProcess for RecordedProcess {
        fn retire(self: Box<Self>, reported: bool) -> std::io::Result<()> {
            *self.0.lock().unwrap() = Some(reported);
            Ok(())
        }

        fn diagnostics(&self) -> String {
            String::new()
        }
    }

    /// A job that reports `frames` and keeps its stream open, with how it was retired.
    fn job(
        frames: Vec<JobFrame>,
    ) -> (StartedJob, Arc<Mutex<Option<bool>>>, mpsc::Sender<JobFrame>) {
        let (sender, receiver) = mpsc::channel(8);
        for frame in frames {
            assert!(sender.try_send(frame).is_ok());
        }
        let retired = Arc::new(Mutex::new(None));
        (
            StartedJob {
                job: MeasureJob(7),
                frames: receiver,
                process: Box::new(RecordedProcess(Arc::clone(&retired))),
            },
            retired,
            sender,
        )
    }

    async fn settle() {
        for _ in 0..64 {
            tokio::task::yield_now().await;
        }
    }

    /// How the process was retired, once its retirement, on the blocking pool, has run.
    fn retirement(retired: &Mutex<Option<bool>>) -> Option<bool> {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while retired.lock().unwrap().is_none() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        *retired.lock().unwrap()
    }

    /// A job that never answers is killed at its deadline, and the job fails with a retry,
    /// while everything else (here the test itself, and the status it reads) goes on.
    #[tokio::test(start_paused = true)]
    async fn a_job_that_never_answers_is_killed_at_its_deadline() {
        let (silent, retired, _open) = job(Vec::new());
        let machine = ScriptedMachine::new(vec![silent], Vec::new());
        let mut status = start(machine.clone(), PENDING, DeviceLock::default());
        status
            .wait_for(|status| status.state == MeasurementState::Measuring)
            .await
            .unwrap();
        tokio::time::sleep(JOB_DEADLINE - Duration::from_secs(1)).await;
        settle().await;
        assert_eq!(*retired.lock().unwrap(), None, "killed before its deadline");
        let failed = status
            .wait_for(|status| matches!(status.state, MeasurementState::Failed { .. }))
            .await
            .unwrap()
            .clone();
        assert_eq!(failed.measurements, PENDING);
        assert_eq!(
            retirement(&retired),
            Some(false),
            "an unreported job is killed"
        );
    }

    /// A job ending with a failed point still publishes what it stored, and its reported
    /// process is left to exit.
    #[tokio::test(start_paused = true)]
    async fn a_failed_key_does_not_stop_the_others() {
        let failure = MeasureFailure::Attempt {
            reason: "read-only cache".into(),
        };
        let (reported, retired, _open) = job(vec![JobFrame::Finished(Ok(vec![failure]))]);
        let machine = ScriptedMachine::new(vec![reported], vec![COMPLETE]);
        let mut status = start(machine, PENDING, DeviceLock::default());
        let measured = status
            .wait_for(|status| status.state == MeasurementState::Measured)
            .await
            .unwrap()
            .clone();
        assert_eq!(measured.measurements, COMPLETE);
        assert_eq!(
            retirement(&retired),
            Some(true),
            "a reported job is left to exit"
        );
    }

    /// The job never starts while residency shares the device, and starts once the share ends.
    #[tokio::test(start_paused = true)]
    async fn the_job_defers_while_a_model_is_resident() {
        let (reported, _, _open) = job(vec![JobFrame::Finished(Ok(Vec::new()))]);
        let machine = ScriptedMachine::new(vec![reported], vec![COMPLETE]);
        let device = DeviceLock::default();
        let resident = device.try_share().unwrap();
        let mut status = start(machine.clone(), PENDING, device);
        tokio::time::sleep(FIRST_RETRY * 3).await;
        settle().await;
        assert_eq!(machine.started.load(Ordering::SeqCst), 0);
        assert_eq!(status.borrow().state, MeasurementState::Unmeasured);
        drop(resident);
        status
            .wait_for(|status| status.state == MeasurementState::Measured)
            .await
            .unwrap();
        assert_eq!(machine.started.load(Ordering::SeqCst), 1);
    }

    /// A point key of the scripted jobs.
    fn key(name: &str) -> PointKey {
        PointKey {
            executor: seismic::Identity {
                physical: seismic::PhysicalIdentity::Uuid([3; 16]),
                backend: seismic::BackendName::Vulkan,
                driver: "580".into(),
            },
            kernels: Vec::new(),
            configuration: Default::default(),
            point: [(name.to_owned(), 1)].into(),
        }
    }

    /// A load during a job preempts it: the job is killed at once and its attempts released,
    /// not counted; the load gets the device without waiting for the job's deadline, and the
    /// state goes back to `Unmeasured`, not `Failed`. The next job runs once the load's share
    /// ends.
    #[tokio::test(start_paused = true)]
    async fn a_load_during_a_job_preempts_it() {
        let (running, retired, _open) = job(vec![
            JobFrame::Attempting(key("launch")),
            JobFrame::Attempting(key("history")),
        ]);
        let (finishing, _, _finished) = job(vec![JobFrame::Finished(Ok(Vec::new()))]);
        let machine = ScriptedMachine::new(vec![running, finishing], vec![COMPLETE]);
        let device = DeviceLock::default();
        let mut status = start(machine.clone(), PENDING, device.clone());
        status
            .wait_for(|status| status.state == MeasurementState::Measuring)
            .await
            .unwrap();
        settle().await;
        let asked = tokio::time::Instant::now();
        let share = device.share().await;
        assert!(
            asked.elapsed() < JOB_DEADLINE,
            "the load waited for the job's deadline"
        );
        assert_eq!(retirement(&retired), Some(false), "the job is killed");
        status
            .wait_for(|status| status.state == MeasurementState::Unmeasured)
            .await
            .unwrap();
        settle().await;
        assert_eq!(
            *machine.released.lock().unwrap(),
            [(MeasureJob(7), vec![key("launch"), key("history")])],
            "every attempt the job reported is released"
        );
        assert_eq!(machine.started.load(Ordering::SeqCst), 1);
        drop(share);
        status
            .wait_for(|status| status.state == MeasurementState::Measured)
            .await
            .unwrap();
        assert_eq!(machine.started.load(Ordering::SeqCst), 2);
    }

    /// Measurements stored at start leave nothing to measure.
    #[tokio::test(start_paused = true)]
    async fn a_start_with_everything_stored_runs_no_job() {
        let machine = ScriptedMachine::new(Vec::new(), Vec::new());
        let status = start(machine.clone(), COMPLETE, DeviceLock::default());
        settle().await;
        assert_eq!(status.borrow().state, MeasurementState::Measured);
        assert_eq!(machine.started.load(Ordering::SeqCst), 0);
    }
}
