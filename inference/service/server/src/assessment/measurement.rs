//! The measurement job's owner (device topology spec §2.15). Measuring opens executors and runs
//! kernels, which can fault or hang in a driver, and a native submission cannot be cancelled once
//! started; so the service never measures in its own process and nothing waits on a measurement.
//! The job is a mode of the inference worker, run only with the device to itself
//! ([`DeviceLock::job`]) and for its time at most
//! ([`magnitude_executor::assessment::MEASUREMENT_SECONDS`] from its start, the worker's own
//! start among it): the worker is given what is left of it, and is killed when it ends, whatever
//! it is doing, and waited for a bounded time on a detached task. A load wanting the device
//! preempts it, killing it at once. A killed job's records stand, as each is stored as it is
//! made; its attempts are released rather than counted, and the next job resumes from what is
//! stored. Its state moves `Unmeasured → Measuring → Measured | Failed { retry_at }` (a
//! preempted job back to `Unmeasured`); it is `Measured` once what is stored gives assessment
//! its costs, and the jobs that measure the rest leave it so. What each job stores reaches the
//! assessment pool, which re-finishes every assessed model under it.

use std::sync::Arc;
use std::time::Duration;

use magnitude_engine::assessment::{AssessmentStructure, Measurements};
use magnitude_engine::worker::measure::{
    JobLost, MeasureEvent, MeasureFailure, MeasureJob, MeasureRequest, measure_worker,
    release_killed,
};
use magnitude_service_contracts::models::MeasurementState;
use seismic::PointKey;
use seismic::measure::Record;
use tokio::sync::{mpsc, watch};

use crate::device_lock::{DeviceJob, DeviceLock};
use crate::residency::controller::GRACEFUL_EXIT;
use crate::residency::worker::{JobWorker, KILL_REAP};
use crate::worker_process::WorkerLauncher;

/// How long a job's process is waited for once it reported or was killed: a reported job's own
/// exit ([`GRACEFUL_EXIT`]), then the kill's reap ([`KILL_REAP`]), the two waits its retirement
/// makes. Past it the wait goes on detached and the device is released, even though the process
/// is not proven gone.
const EXIT_WAIT: Duration = GRACEFUL_EXIT.saturating_add(KILL_REAP);

/// The first retry's delay after a failed job. A retry waits at least twice as long as the failed
/// job ran, so a job that fails every time occupies the device at most a third of the time, and
/// each further failure doubles the delay, up to [`LAST_RETRY`].
const FIRST_RETRY: Duration = Duration::from_secs(40);
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
    /// once it gives assessment its costs, whether or not a later point is still to be measured.
    pub fn stored(stored: S) -> Self {
        let state = match stored.stage() {
            Stage::Unmeasured => MeasurementState::Unmeasured,
            Stage::Refining | Stage::Settled => MeasurementState::Measured,
        };
        Self {
            state,
            measurements: stored,
        }
    }
}

/// How far what is stored has come, all the job reads of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// It gives assessment no costs yet.
    Unmeasured,
    /// It gives assessment its costs, and a later point is still to be measured.
    Refining,
    /// Nothing is left to measure: the costs are complete, or there are none to be had.
    Settled,
}

/// What is stored of the machine, as far as the job reads it.
pub trait StoredRecords: Clone + Send + Sync + 'static {
    fn stage(&self) -> Stage;
}

impl StoredRecords for Measurements {
    fn stage(&self) -> Stage {
        match self {
            Self::Pending => Stage::Unmeasured,
            Self::Measured { pending: true, .. } => Stage::Refining,
            Self::Measured { pending: false, .. } | Self::Unavailable { .. } => Stage::Settled,
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
    /// Start a measurement job that has `time` from its start. Blocking.
    fn start(&self, time: Duration) -> anyhow::Result<StartedJob>;
    /// Release killed `job`'s attempts of `keys`, so none is counted. Blocking.
    fn release_killed(&self, job: MeasureJob, keys: &[PointKey]) -> anyhow::Result<()>;
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

    fn start(&self, time: Duration) -> anyhow::Result<StartedJob> {
        let job = next_job();
        let request = MeasureRequest {
            paths: self.served.clone(),
            kernels: self.kernels.clone(),
            executor: self.structure.selected.info.selector(),
            job,
            time,
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

    fn release_killed(&self, job: MeasureJob, keys: &[PointKey]) -> anyhow::Result<()> {
        for key in keys {
            release_killed(&self.kernels, job, key)?;
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
/// it at start, each job starting only once `settled` holds (the assessment pool's work on the
/// device is done, [`crate::assessment::HostDevice`]). The returned status is the job's, from
/// [`MeasurementStatus::stored`] on.
pub fn start<M: MeasuredMachine>(
    machine: Arc<M>,
    stored: M::Stored,
    device: DeviceLock,
    settled: watch::Receiver<bool>,
) -> watch::Receiver<MeasurementStatus<M::Stored>> {
    let (status, receiver) = watch::channel(MeasurementStatus::stored(stored));
    let time = Duration::from_secs_f64(magnitude_executor::assessment::MEASUREMENT_SECONDS);
    tokio::spawn(run(machine, device, settled, status, time));
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

/// Run jobs, each for `time` at most, until every point is stored: the executor's fixed cost
/// points, whatever the models assessed. Assessment has its costs, and the state is `Measured`,
/// once the probe is stored; the jobs after that refine the figures it prices meanwhile. Each
/// job runs only with the device to itself, deferring while residency shares it or a catalog
/// optimization holds it ([`DeviceLock::job`]) and while `settled` does not hold, preempted when
/// residency wants it back, and retried after a failure with a backoff that doubles. A preempted
/// job is no failure: the next resumes from what it stored. Nor is a job whose time ended, by
/// its own walk or by its kill, with records stored and no point failing: the next follows at
/// once. A figure that could not be read outside its points' timings fails each job alike and
/// stays priced, so it is no job's failure. A job that stored nothing failed. The hardware and
/// driver identities are the inventory's revision, fixed for the service's life, so the backoff
/// is this process's and a changed identity is the next process's discovery, starting anew.
async fn run<M: MeasuredMachine>(
    machine: Arc<M>,
    device: DeviceLock,
    mut settled: watch::Receiver<bool>,
    status: watch::Sender<MeasurementStatus<M::Stored>>,
    time: Duration,
) {
    let mut backoff = FIRST_RETRY;
    while status.borrow().measurements.stage() != Stage::Settled {
        // A pool that is gone has no work left on the device.
        let _ = settled.wait_for(|settled| *settled).await;
        let exclusive = device.job().await;
        // A job refining measured costs leaves them measured, whatever becomes of it.
        let measured = status.borrow().measurements.stage() != Stage::Unmeasured;
        if !measured {
            status.send_modify(|status| status.state = MeasurementState::Measuring);
        }
        let started = tokio::time::Instant::now();
        let (outcome, recorded) = match run_job(&machine, exclusive, time).await {
            JobEnd::Ended(outcome, recorded) => (Some(outcome), recorded),
            JobEnd::Expired(recorded) => (None, recorded),
            JobEnd::Preempted => {
                if !measured {
                    status.send_modify(|status| status.state = MeasurementState::Unmeasured);
                }
                continue;
            }
        };
        let failed = match &outcome {
            Some(Ok(failures)) => {
                for failure in failures {
                    tracing::warn!(%failure, "the measurement job could not measure");
                }
                failures
                    .iter()
                    .any(|failure| !matches!(failure, MeasureFailure::Figure { .. }))
            }
            Some(Err(error)) => {
                tracing::warn!(%error, "the measurement job was lost");
                true
            }
            None => false,
        };
        let progressed = recorded > 0 && !failed;
        match read_stored(&machine).await {
            Some(measurements) => {
                let stage = measurements.stage();
                let state = match stage {
                    Stage::Unmeasured => MeasurementState::Measuring,
                    Stage::Refining | Stage::Settled => MeasurementState::Measured,
                };
                status.send_replace(MeasurementStatus {
                    state,
                    measurements,
                });
                // Its time ended with what it measured stored: the next job takes the rest. A
                // point still without a record otherwise fails the job: the next one takes it.
                if stage == Stage::Settled || progressed {
                    backoff = FIRST_RETRY;
                } else {
                    retry(&status, &mut backoff, started.elapsed()).await;
                }
            }
            None => retry(&status, &mut backoff, started.elapsed()).await,
        }
    }
    // The last state stands for as long as anything reads it.
    status.closed().await;
}

/// Wait out the backoff after a job that ran for `ran` failed, at least twice `ran` and at most
/// [`LAST_RETRY`], which then doubles: reported as the state where what is stored gives
/// assessment no costs yet, and otherwise only refining them, so the state stays `Measured`.
async fn retry<S: StoredRecords>(
    status: &watch::Sender<MeasurementStatus<S>>,
    backoff: &mut Duration,
    ran: Duration,
) {
    let delay = (*backoff).max(ran.saturating_mul(2)).min(LAST_RETRY);
    let retry_at = std::time::SystemTime::now() + delay;
    status.send_if_modified(|status| {
        let failed = status.measurements.stage() == Stage::Unmeasured;
        if failed {
            status.state = MeasurementState::Failed {
                retry_at: epoch_millis(retry_at),
            };
        }
        failed
    });
    tokio::time::sleep(delay).await;
    *backoff = delay.saturating_mul(2).min(LAST_RETRY);
}

/// How one job ended.
enum JobEnd {
    /// It reported its outcome, or was lost, having stored this many records.
    Ended(Result<Vec<MeasureFailure>, JobLost>, usize),
    /// Its time ended first: it was killed having stored this many records, and its attempts
    /// were released.
    Expired(usize),
    /// Residency wanted the device: the job was killed, and its attempts released.
    Preempted,
}

/// Why a job was killed before it reported.
enum Killed {
    Expired,
    Preempted,
}

/// One job on `exclusive`, for `time` from its start at most: the worker is given what is left
/// of it as it is started, and its frames are read until it reports its outcome, its time
/// ends or residency wants the device, either of which kills it at once; its process is then
/// retired, waiting at most [`EXIT_WAIT`] on a detached task, and the device released. A killed
/// job's attempts are then released, so a kill never counts against a point.
async fn run_job<M: MeasuredMachine>(
    machine: &Arc<M>,
    mut exclusive: DeviceJob,
    time: Duration,
) -> JobEnd {
    let ends = tokio::time::Instant::now() + time;
    let started = {
        let machine = Arc::clone(machine);
        crate::spawn_blocking_traced(move || {
            machine.start(ends.saturating_duration_since(tokio::time::Instant::now()))
        })
        .await
    };
    let StartedJob {
        job,
        mut frames,
        process,
    } = match started {
        Ok(Ok(started)) => started,
        Ok(Err(error)) => {
            return JobEnd::Ended(Err(JobLost(format!("starting the job: {error:#}"))), 0);
        }
        Err(error) => {
            return JobEnd::Ended(Err(JobLost(format!("starting the job: {error}"))), 0);
        }
    };
    let started = tokio::time::Instant::now();
    // Every point the job reported attempting: those a killed job's markers may count.
    let mut attempted = Vec::new();
    // The records the job reported storing.
    let mut recorded = 0;
    let outcome = loop {
        tokio::select! {
            frame = frames.recv() => {
                match frame {
                    Some(JobFrame::Attempting(key)) => {
                        tracing::info!(
                            elapsed_ms = u64::try_from(started.elapsed().as_millis())
                                .unwrap_or(u64::MAX),
                            configuration = ?key.configuration,
                            point = ?key.point,
                            "measurement attempting"
                        );
                        attempted.push(key);
                    }
                    Some(JobFrame::Recorded(key, record)) => {
                        recorded += 1;
                        log_record(&key, &record);
                    }
                    Some(JobFrame::Finished(outcome)) => break Ok(outcome),
                    None => break Ok(Err(JobLost("the job's reader ended".into()))),
                }
            }
            () = tokio::time::sleep_until(ends) => break Err(Killed::Expired),
            () = exclusive.preempted() => break Err(Killed::Preempted),
        }
    };
    tracing::info!(
        elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
        outcome = match &outcome {
            Ok(Ok(_)) => "reported",
            Ok(Err(_)) => "lost",
            Err(Killed::Expired) => "expired",
            Err(Killed::Preempted) => "preempted",
        },
        "measurement job ended"
    );
    let outcome = outcome.map(|outcome| {
        outcome.map_err(|lost| match process.diagnostics() {
            diagnostics if diagnostics.is_empty() => lost,
            diagnostics => JobLost(format!("{lost}; worker diagnostics: {diagnostics}")),
        })
    });
    let reported = matches!(outcome, Ok(Ok(_)));
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
    let killed = match outcome {
        Ok(outcome) => return JobEnd::Ended(outcome, recorded),
        Err(killed) => killed,
    };
    // The frames the job sent before it was killed end once its stream does; an attempt it
    // reported is one its markers may count, and a record it reported is stored. Read for at
    // most the exit wait.
    let drained = tokio::time::timeout(EXIT_WAIT, async {
        while let Some(frame) = frames.recv().await {
            match frame {
                JobFrame::Attempting(key) => attempted.push(key),
                JobFrame::Recorded(key, record) => {
                    recorded += 1;
                    log_record(&key, &record);
                }
                JobFrame::Finished(_) => {}
            }
        }
    })
    .await;
    if drained.is_err() {
        tracing::warn!(
            wait = ?EXIT_WAIT,
            "a killed measurement job's stream did not end; attempts it sent after are counted"
        );
    }
    let released = {
        let machine = Arc::clone(machine);
        crate::spawn_blocking_traced(move || machine.release_killed(job, &attempted)).await
    };
    match released {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            tracing::warn!(error = %format!("{error:#}"), "a killed job's attempts stay counted");
        }
        Err(error) => tracing::warn!(%error, "releasing a killed job's attempts failed"),
    }
    match killed {
        Killed::Expired => JobEnd::Expired(recorded),
        Killed::Preempted => JobEnd::Preempted,
    }
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

    /// What the scripted jobs left stored: its stage alone.
    impl StoredRecords for Stage {
        fn stage(&self) -> Stage {
            *self
        }
    }

    const PENDING: Stage = Stage::Unmeasured;
    const REFINING: Stage = Stage::Refining;
    const COMPLETE: Stage = Stage::Settled;

    /// A machine whose jobs are scripted: each start hands out the next job's frames, and
    /// `stored` reads what the jobs left.
    struct ScriptedMachine {
        jobs: Mutex<Vec<StartedJob>>,
        stored: Mutex<Stage>,
        /// What the reads find, one each, in order; the last stands after them.
        reads: Mutex<std::collections::VecDeque<Stage>>,
        started: AtomicUsize,
        /// The time each job was given.
        given: Mutex<Vec<Duration>>,
        /// Every killed job's released attempts.
        released: Mutex<Vec<(MeasureJob, Vec<PointKey>)>>,
    }

    impl ScriptedMachine {
        fn new(jobs: Vec<StartedJob>, reads: Vec<Stage>) -> Arc<Self> {
            Arc::new(Self {
                jobs: Mutex::new(jobs),
                stored: Mutex::new(PENDING),
                reads: Mutex::new(reads.into()),
                started: AtomicUsize::new(0),
                given: Mutex::new(Vec::new()),
                released: Mutex::new(Vec::new()),
            })
        }
    }

    impl MeasuredMachine for ScriptedMachine {
        type Stored = Stage;

        fn stored(&self) -> Stage {
            if let Some(read) = self.reads.lock().unwrap().pop_front() {
                *self.stored.lock().unwrap() = read;
            }
            *self.stored.lock().unwrap()
        }

        fn start(&self, time: Duration) -> anyhow::Result<StartedJob> {
            self.started.fetch_add(1, Ordering::SeqCst);
            self.given.lock().unwrap().push(time);
            Ok(self.jobs.lock().unwrap().remove(0))
        }

        fn release_killed(&self, job: MeasureJob, keys: &[PointKey]) -> anyhow::Result<()> {
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

    /// The job's time.
    fn time() -> Duration {
        Duration::from_secs_f64(magnitude_executor::assessment::MEASUREMENT_SECONDS)
    }

    /// A job is killed when its time ends, however it reports: its attempts are released, and
    /// one that stored nothing fails with a retry, while everything else (here the test itself,
    /// and the status it reads) goes on. The worker is given no more than the time.
    #[tokio::test(start_paused = true)]
    async fn a_job_is_killed_when_its_time_ends() {
        let (running, retired, frames) = job(Vec::new());
        let machine = ScriptedMachine::new(vec![running], Vec::new());
        let mut status = start(machine.clone(), PENDING, DeviceLock::default(), settled());
        status
            .wait_for(|status| status.state == MeasurementState::Measuring)
            .await
            .unwrap();
        assert!(machine.given.lock().unwrap().iter().all(|given| *given <= time()));
        let tick = time() / 8;
        for point in 0..7 {
            tokio::time::sleep(tick).await;
            frames
                .send(JobFrame::Attempting(key(&point.to_string())))
                .await
                .unwrap();
            settle().await;
        }
        assert_eq!(*retired.lock().unwrap(), None, "killed before its time ended");
        let failed = status
            .wait_for(|status| matches!(status.state, MeasurementState::Failed { .. }))
            .await
            .unwrap()
            .clone();
        assert_eq!(failed.measurements, PENDING);
        assert_eq!(retirement(&retired), Some(false), "the job is killed");
        assert_eq!(
            *machine.released.lock().unwrap(),
            [(MeasureJob(7), (0..7).map(|point| key(&point.to_string())).collect())],
            "every attempt the job reported is released"
        );
    }

    /// A job killed as its time ends having stored records is no failure: they stand, and the
    /// next job follows at once.
    #[tokio::test(start_paused = true)]
    async fn a_job_killed_after_storing_is_followed_at_once() {
        let record = Record::Measured(seismic::measure::Timing {
            median: 1e-3,
            interval: None,
            samples: 1,
        });
        let (killed, _, _killed) = job(vec![
            JobFrame::Attempting(key("probe")),
            JobFrame::Recorded(key("probe"), record),
            JobFrame::Attempting(key("launch")),
        ]);
        let (finishing, _, _finished) = job(vec![JobFrame::Finished(Ok(Vec::new()))]);
        let machine = ScriptedMachine::new(vec![killed, finishing], vec![REFINING, COMPLETE]);
        let began = tokio::time::Instant::now();
        let mut status = start(machine.clone(), PENDING, DeviceLock::default(), settled());
        status
            .wait_for(|status| status.measurements == COMPLETE)
            .await
            .unwrap();
        assert_eq!(machine.started.load(Ordering::SeqCst), 2);
        assert!(
            began.elapsed() < FIRST_RETRY,
            "the second job followed the first with no backoff"
        );
    }

    /// A failed job is retried after at least twice as long as it ran, so a job that fails every
    /// time occupies the device at most a third of the time.
    #[tokio::test(start_paused = true)]
    async fn a_failed_job_is_retried_after_twice_its_running_time() {
        let (silent, _, _open) = job(Vec::new());
        let (finishing, _, _finished) = job(vec![JobFrame::Finished(Ok(Vec::new()))]);
        let machine = ScriptedMachine::new(vec![silent, finishing], vec![PENDING, COMPLETE]);
        let mut status = start(machine.clone(), PENDING, DeviceLock::default(), settled());
        status
            .wait_for(|status| matches!(status.state, MeasurementState::Failed { .. }))
            .await
            .unwrap();
        tokio::time::sleep(time() * 2 - Duration::from_secs(1)).await;
        settle().await;
        assert_eq!(
            machine.started.load(Ordering::SeqCst),
            1,
            "retried too soon"
        );
        status
            .wait_for(|status| status.state == MeasurementState::Measured)
            .await
            .unwrap();
        assert_eq!(machine.started.load(Ordering::SeqCst), 2);
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
        let mut status = start(machine, PENDING, DeviceLock::default(), settled());
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
        let mut status = start(machine.clone(), PENDING, device, settled());
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

    /// An assessment pool that is settled for good.
    fn settled() -> watch::Receiver<bool> {
        watch::channel(true).1
    }

    /// The job never starts while the assessment pool works the device, and starts once it
    /// settles.
    #[tokio::test(start_paused = true)]
    async fn the_job_defers_while_the_assessment_pool_works() {
        let (reported, _, _open) = job(vec![JobFrame::Finished(Ok(Vec::new()))]);
        let machine = ScriptedMachine::new(vec![reported], vec![COMPLETE]);
        let (settling, settled) = watch::channel(false);
        let mut status = start(machine.clone(), PENDING, DeviceLock::default(), settled);
        tokio::time::sleep(FIRST_RETRY * 3).await;
        settle().await;
        assert_eq!(machine.started.load(Ordering::SeqCst), 0);
        settling.send_replace(true);
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
    /// not counted; the load gets the device without waiting for the job's time to end, and the
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
        let mut status = start(machine.clone(), PENDING, device.clone(), settled());
        status
            .wait_for(|status| status.state == MeasurementState::Measuring)
            .await
            .unwrap();
        settle().await;
        let asked = tokio::time::Instant::now();
        let share = device.share().await;
        assert!(
            asked.elapsed() < time(),
            "the load waited for the job's time to end"
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

    /// A job whose time ends with records stored and nothing failed is no failure: the next job
    /// starts at once, with no backoff, and the state never reads failed. One that stored
    /// nothing fails.
    #[tokio::test(start_paused = true)]
    async fn a_job_whose_time_ends_after_storing_is_followed_at_once() {
        let record = Record::Measured(seismic::measure::Timing {
            median: 1e-3,
            interval: None,
            samples: 1,
        });
        let (first, _, _first) = job(vec![
            JobFrame::Recorded(key("a"), record),
            JobFrame::Finished(Ok(Vec::new())),
        ]);
        let (second, _, _second) = job(vec![JobFrame::Finished(Ok(Vec::new()))]);
        let machine = ScriptedMachine::new(vec![first, second], vec![PENDING, PENDING]);
        let began = tokio::time::Instant::now();
        let mut status = start(machine.clone(), PENDING, DeviceLock::default(), settled());
        let failed = status
            .wait_for(|status| matches!(status.state, MeasurementState::Failed { .. }))
            .await
            .unwrap()
            .clone();
        assert_eq!(failed.measurements, PENDING);
        assert_eq!(machine.started.load(Ordering::SeqCst), 2);
        assert!(
            began.elapsed() < FIRST_RETRY,
            "the second job followed the first with no backoff"
        );
    }

    /// Measured costs that are still refining keep the state measured through every job that
    /// refines them, one that fails included, until every point is stored.
    #[tokio::test(start_paused = true)]
    async fn jobs_refining_measured_costs_leave_them_measured() {
        let record = Record::Measured(seismic::measure::Timing {
            median: 1e-3,
            interval: None,
            samples: 1,
        });
        let (failing, _, _failing) = job(vec![JobFrame::Finished(Ok(Vec::new()))]);
        let (refining, _, _refining) = job(vec![
            JobFrame::Recorded(key("deep"), record),
            JobFrame::Finished(Ok(Vec::new())),
        ]);
        let machine = ScriptedMachine::new(vec![failing, refining], vec![REFINING, COMPLETE]);
        let status = start(machine.clone(), REFINING, DeviceLock::default(), settled());
        assert_eq!(status.borrow().state, MeasurementState::Measured);
        let mut seen = status.clone();
        let states = tokio::spawn(async move {
            let mut states = Vec::new();
            while seen.changed().await.is_ok() {
                let status = seen.borrow_and_update().clone();
                states.push(status.state);
                if status.measurements == COMPLETE {
                    break;
                }
            }
            states
        })
        .await
        .unwrap();
        assert!(
            states
                .iter()
                .all(|state| *state == MeasurementState::Measured),
            "{states:?}"
        );
        assert_eq!(machine.started.load(Ordering::SeqCst), 2);
    }

    /// Measurements stored at start leave nothing to measure.
    #[tokio::test(start_paused = true)]
    async fn a_start_with_everything_stored_runs_no_job() {
        let machine = ScriptedMachine::new(Vec::new(), Vec::new());
        let status = start(machine.clone(), COMPLETE, DeviceLock::default(), settled());
        settle().await;
        assert_eq!(status.borrow().state, MeasurementState::Measured);
        assert_eq!(machine.started.load(Ordering::SeqCst), 0);
    }
}
