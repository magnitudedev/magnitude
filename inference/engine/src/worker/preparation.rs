//! Loading with preparation: the one way a host loads a model, and the one
//! way a model's kernels are searched.
//!
//! A load never searches. The search runs in a Prepare worker, a process of
//! its own that publishes each tuning unit's record to the kernel cache and
//! exits; a load, in another process, reads the records. So nothing the
//! search allocated exists in the process that serves, and a first load's
//! serving process is a later load's.
//!
//! [`load_prepared`] is the sequence every host runs. The load is first
//! asked to report tuning units without a stored record: one that finds
//! every unit decided is simply the load, at no further cost. One that meets
//! such a unit places nothing and ends; the model's Prepare then runs to its
//! end and its process is shown gone; and the load is started again, now
//! total: a unit still without a record, because the Prepare was stopped,
//! failed or could not run, is served at its canonical default. A load
//! under a development pin that tunes nothing is total from the start: no
//! Prepare runs for it.

use super::client::prepare_worker;
use super::protocol::{HostMessage, LoadProgress, WorkerMessage};
use super::transport::{FramedTransport, HostTransport};
use crate::error::{LoadError, PreviewError};
use crate::execution::LoadAttempt;
use crate::options::ExecutionManifest;
use magnitude_executor::{UnpreparedUnits, pinned_tuning};
use std::future::Future;
use std::io::BufReader;
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};

/// A worker process a host started for one request.
pub struct WorkerProcess<T, R> {
    /// The host's end of the process's transport.
    pub transport: T,
    /// Returns how the process ended, once it no longer exists; an error
    /// when that cannot be established. A worker that reported its outcome,
    /// or whose stream ended, exits on its own and is waited for, never
    /// killed: killing it would skip everything a normal exit completes, the
    /// system's compiled-kernel cache its search wrote among it. Only a
    /// worker that reported nothing and may still run is ended.
    pub retire: R,
}

/// What a worker had told its host when the host retires it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerReport {
    /// The worker reported its outcome: it is exiting on its own.
    Reported,
    /// The worker's stream ended before it reported an outcome: its process
    /// is ending, on its own or killed, and its exit says how.
    Ended,
    /// The worker reported no outcome and its stream did not end: it broke
    /// the protocol or its stream failed, and it may still run.
    Unreported,
}

/// How a retired worker's process ended ([`WorkerProcess::retire`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerEnd {
    /// It ended on its own, with this status: an exit code, or the signal
    /// that ended it (a crash, or a kill from outside, such as the
    /// system's under memory pressure).
    Exited(ExitStatus),
    /// Its host ended it, having had no outcome from it.
    Stopped,
    /// No process was there to observe: a worker in the host's own process.
    Unobserved,
}

impl std::fmt::Display for WorkerEnd {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Exited(status) => write!(formatter, "its process ended: {status}"),
            Self::Stopped => formatter.write_str("its host ended its process"),
            Self::Unobserved => formatter.write_str("no process was observed"),
        }
    }
}

/// What became of preparation in a [`load_prepared`]. None of these fails
/// the load.
#[derive(Debug)]
pub enum Preparation {
    /// The load found every tuning unit decided, or a development pin
    /// decides them: nothing was started besides it.
    Unneeded,
    /// The load met a unit without a record, the Prepare ran to its end,
    /// and the load that followed read its records.
    Prepared,
    /// The manifest names no kernel cache, so no record could reach the
    /// load: no Prepare was started, and the load serves canonical defaults.
    Unshared,
    /// The Prepare did not run to its end. The records it published stand;
    /// the load serves canonical defaults for the units without one.
    Incomplete(LoadError),
}

/// A device hold returned by [`run_priced`]'s `acquire_device` capability.
pub struct AcquiredDevice<D> {
    pub device: D,
    /// Whether acquiring the device waited for another holder to release it,
    /// which can change the room a plan settles into and therefore requires
    /// re-previewing before running.
    pub waited: bool,
}

impl<D> AcquiredDevice<D> {
    pub fn immediate(device: D) -> Self {
        Self {
            device,
            waited: false,
        }
    }
}

/// Failure of the `run` capability inside [`run_priced`].
pub enum PricedFailure<H, C, S = ()> {
    /// A load or Prepare worker failed with [`LoadError`], carrying caller
    /// context `C` and returning caller state `S` so [`run_priced`] can retry
    /// on [`LoadError::PricesPending`] without shared interior mutability.
    Load(LoadError, C, S),
    /// A non-worker host failure; propagated immediately without settling.
    Host(H),
}

/// Final error returned by [`run_priced`].
#[derive(Debug)]
pub enum PricedError<H, C> {
    /// Preview failed (including [`PreviewError::PricesPending`] when settling
    /// left the pending set unchanged).
    Preview(PreviewError),
    /// A load or Prepare worker failed (including [`LoadError::PricesPending`]
    /// when settling made no progress).
    Load(LoadError, C),
    /// A host capability (`settle`, `acquire_device`, or `run`) failed.
    Host(H),
}

/// Run a priced operation (`load_prepared` or `prepare`) as the single owner
/// of price settling and device-hold release/re-acquire.
///
/// 1. Preview while holding nothing of the device; while preview reports
///    [`PreviewError::PricesPending`], compare its typed pending set against
///    the previous attempt's: if unchanged, propagate the pending error
///    immediately; otherwise call `settle` and preview again.
/// 2. Acquire the device via `acquire_device`. If acquiring waited for
///    another holder, preview again; if that preview reports
///    [`PreviewError::PricesPending`], release the device via
///    `release_device`, record its pending set, settle, and return to step 1
///    (never settling while holding the device).
/// 3. Run the operation with the preview, held device, and caller state `S`.
///    If it reports [`LoadError::PricesPending`], release the device via
///    `release_device` and return to step 1 with the returned state `S`; if
///    step 1 finds nothing pending (settling made no progress), propagate the
///    pending error immediately.
pub async fn run_priced<P, D, S, T, H, C, PF, SF, AF, RF>(
    mut preview: impl FnMut() -> PF,
    mut settle: impl FnMut() -> SF,
    mut acquire_device: impl FnMut(&P) -> AF,
    mut release_device: impl FnMut(D),
    mut state: S,
    mut run: impl FnMut(&P, &D, S) -> RF,
) -> Result<(T, P, D), PricedError<H, C>>
where
    PF: Future<Output = Result<P, PreviewError>>,
    SF: Future<Output = Result<(), H>>,
    AF: Future<Output = Result<AcquiredDevice<D>, H>>,
    RF: Future<Output = Result<T, PricedFailure<H, C, S>>>,
{
    let mut pending_load: Option<(LoadError, C)> = None;
    let mut previous_pending: Option<crate::error::PendingPrices> = None;
    loop {
        let mut settled = false;
        let previewed = loop {
            match preview().await {
                Ok(value) => {
                    previous_pending = None;
                    break value;
                }
                Err(PreviewError::PricesPending { reason, pending }) => {
                    if previous_pending.as_ref() == Some(&pending) {
                        return Err(PricedError::Preview(PreviewError::PricesPending {
                            reason,
                            pending,
                        }));
                    }
                    tracing::info!(%reason, "the load waits for the machine's standing figures");
                    previous_pending = Some(pending);
                    settle().await.map_err(PricedError::Host)?;
                    settled = true;
                }
                Err(error) => return Err(PricedError::Preview(error)),
            }
        };
        if let Some((error, context)) = pending_load.take() {
            if !settled {
                return Err(PricedError::Load(error, context));
            }
        }
        let acquired = acquire_device(&previewed)
            .await
            .map_err(PricedError::Host)?;
        let previewed = if acquired.waited {
            match preview().await {
                Ok(value) => value,
                Err(PreviewError::PricesPending { reason, pending }) => {
                    tracing::info!(%reason, "the load waits for the machine's standing figures");
                    release_device(acquired.device);
                    previous_pending = Some(pending);
                    settle().await.map_err(PricedError::Host)?;
                    continue;
                }
                Err(error) => {
                    release_device(acquired.device);
                    return Err(PricedError::Preview(error));
                }
            }
        } else {
            previewed
        };
        match run(&previewed, &acquired.device, state).await {
            Ok(value) => return Ok((value, previewed, acquired.device)),
            Err(PricedFailure::Load(error @ LoadError::PricesPending { .. }, context, returned)) => {
                release_device(acquired.device);
                state = returned;
                pending_load = Some((error, context));
            }
            Err(PricedFailure::Load(error, context, _)) => {
                return Err(PricedError::Load(error, context));
            }
            Err(PricedFailure::Host(error)) => return Err(PricedError::Host(error)),
        }
    }
}

/// Load `manifest`'s model, having its kernels searched first when records
/// are missing.
///
/// `load` makes one attempt to load, in the mode it is given; when it
/// returns [`LoadAttempt::Unprepared`], nothing of that attempt may remain
/// (a worker process has ended; an in-process attempt has dropped all it
/// opened). `start_prepare` starts a worker process for the model's
/// Prepare, whose progress reaches `prepare_progress`.
///
/// Besides `load`'s own failures and [`LoadError::PricesPending`] (which a
/// caller handles through [`run_priced`]), the only failure is a Prepare
/// process that cannot be shown to have ended: a load beside it would share
/// the device with a search. A Prepare that reported its outcome is waited
/// for and never killed ([`WorkerProcess::retire`]).
pub fn load_prepared<L, T, R>(
    manifest: &ExecutionManifest,
    mut load: impl FnMut(UnpreparedUnits) -> Result<LoadAttempt<L>, LoadError>,
    start_prepare: impl FnOnce() -> Result<WorkerProcess<T, R>, LoadError>,
    prepare_progress: impl FnMut(LoadProgress),
) -> Result<(L, Preparation), LoadError>
where
    T: HostTransport,
    R: FnOnce(WorkerReport) -> Result<WorkerEnd, String>,
{
    // Without a kernel cache a Prepare's records reach no load.
    if manifest.kernel_cache.is_none() {
        return Ok((serve_defaults(&mut load)?, Preparation::Unshared));
    }
    if pinned_tuning::tunes_nothing() {
        return Ok((serve_defaults(&mut load)?, Preparation::Unneeded));
    }
    if let LoadAttempt::Loaded(loaded) = load(UnpreparedUnits::Report)? {
        return Ok((*loaded, Preparation::Unneeded));
    }
    let preparation = prepare(manifest, start_prepare, prepare_progress)?;
    Ok((serve_defaults(&mut load)?, preparation))
}

/// Run `manifest`'s Prepare to its end in the worker process `start`
/// starts, reporting its progress, and return once that process no longer
/// exists. [`load_prepared`] does this for a load that found records
/// missing; a host calls it directly only to prepare a model no load is
/// waiting for (after an installation).
///
/// [`LoadError::PricesPending`] is returned as `Err` (never absorbed as
/// [`Preparation::Incomplete`]), as is a Prepare process that cannot be
/// shown to have ended; callers settle pending prices through [`run_priced`].
pub fn prepare<T, R>(
    manifest: &ExecutionManifest,
    start: impl FnOnce() -> Result<WorkerProcess<T, R>, LoadError>,
    mut progress: impl FnMut(LoadProgress),
) -> Result<Preparation, LoadError>
where
    T: HostTransport,
    R: FnOnce(WorkerReport) -> Result<WorkerEnd, String>,
{
    if manifest.kernel_cache.is_none() {
        return Ok(Preparation::Unshared);
    }
    let WorkerProcess { transport, retire } = match start() {
        Ok(process) => process,
        Err(error @ LoadError::PricesPending { .. }) => return Err(error),
        Err(error) => return Ok(Preparation::Incomplete(error)),
    };
    // The host's end of the transport is closed by now: the worker waits on
    // nothing of the host's to exit.
    let prepared = prepare_worker(transport, manifest.clone(), &mut progress);
    let report = match &prepared {
        Ok(_) => WorkerReport::Reported,
        Err(lost) if lost.ended => WorkerReport::Ended,
        Err(_) => WorkerReport::Unreported,
    };
    let end = retire(report).map_err(|reason| LoadError::WorkerLost {
        reason: format!("the Prepare worker could not be shown to have ended: {reason}"),
    })?;
    // A worker lost says how its process ended: a crash or a kill leaves
    // no other trace.
    match prepared {
        Ok(Ok(())) => Ok(Preparation::Prepared),
        Ok(Err(error @ LoadError::PricesPending { .. })) => Err(error),
        Ok(Err(error)) => Ok(Preparation::Incomplete(error)),
        Err(lost) => Ok(Preparation::Incomplete(LoadError::WorkerLost {
            reason: format!("{}; {end}", lost.reason),
        })),
    }
}

/// The total load: every unit without a record at its canonical default.
fn serve_defaults<L>(
    load: &mut impl FnMut(UnpreparedUnits) -> Result<LoadAttempt<L>, LoadError>,
) -> Result<L, LoadError> {
    match load(UnpreparedUnits::ServeDefaults)? {
        LoadAttempt::Loaded(loaded) => Ok(*loaded),
        LoadAttempt::Unprepared => Err(LoadError::Internal {
            reason: "a load serving canonical defaults stopped unprepared".into(),
        }),
    }
}

/// The host end of a child process's standard streams.
pub type ChildTransport =
    FramedTransport<BufReader<ChildStdout>, ChildStdin, WorkerMessage, HostMessage>;

/// Start `command` as a worker process speaking the worker protocol over its
/// standard input and output (its diagnostics stay on this process's
/// standard error): a host executable's own worker entry, which runs
/// [`super::serve_worker`] over [`super::transport::stdio_worker_transport`].
pub fn start_worker_process(
    command: &mut Command,
) -> Result<
    WorkerProcess<ChildTransport, impl FnOnce(WorkerReport) -> Result<WorkerEnd, String>>,
    LoadError,
> {
    spawn_worker(command, Stdio::inherit()).map(|(process, _)| process)
}

/// [`start_worker_process`], its diagnostics on a pipe this process reads
/// rather than on this process's standard error: for a host that observes
/// what the worker reports as it runs. The host keeps the pipe drained, or
/// the worker stops once its buffer fills.
pub fn observe_worker_process(
    mut command: Command,
) -> Result<
    (WorkerProcess<ChildTransport, impl FnOnce(WorkerReport) -> Result<WorkerEnd, String>>, ChildStderr),
    LoadError,
> {
    let (process, stderr) = spawn_worker(&mut command, Stdio::piped())?;
    let stderr = stderr.ok_or_else(|| LoadError::WorkerLost { reason: "the worker process has no diagnostic stream".into() })?;
    Ok((process, stderr))
}

/// `command` started with its diagnostics on `stderr`, and their pipe when
/// piped.
fn spawn_worker(
    command: &mut Command,
    stderr: Stdio,
) -> Result<
    (WorkerProcess<ChildTransport, impl FnOnce(WorkerReport) -> Result<WorkerEnd, String>>, Option<ChildStderr>),
    LoadError,
> {
    let lost = |reason: String| LoadError::WorkerLost { reason };
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(stderr)
        .spawn()
        .map_err(|error| lost(format!("starting {command:?}: {error}")))?;
    let streams = child.stdin.take().zip(child.stdout.take());
    let Some((stdin, stdout)) = streams else {
        let _ = retire(child, WorkerReport::Unreported);
        return Err(lost("the worker process has no standard streams".into()));
    };
    let diagnostics = child.stderr.take();
    Ok((
        WorkerProcess {
            transport: FramedTransport::new(BufReader::new(stdout), stdin),
            retire: move |report| retire(child, report),
        },
        diagnostics,
    ))
}

/// Return once `child` is gone, with how it ended. A worker that reported
/// its outcome, or whose stream ended, is waited for, however long its exit
/// takes: after its report it only drops what it held and exits, with
/// nothing of the host's to wait on, and its stream ends only as its
/// process does; nothing here bounds that wait, so a worker that reported
/// and then never exits holds its host here. A worker that reported nothing
/// is ended if it still runs.
fn retire(child: Child, report: WorkerReport) -> Result<WorkerEnd, String> {
    ended(child, report).map_err(|error| error.to_string())
}

/// How `child` ended, once it is gone ([`retire`]).
fn ended(mut child: Child, report: WorkerReport) -> std::io::Result<WorkerEnd> {
    if report == WorkerReport::Unreported {
        if let Some(status) = child.try_wait()? {
            return Ok(WorkerEnd::Exited(status));
        }
        // A process that ended between the two calls is gone either way.
        let _ = child.kill();
        child.wait()?;
        return Ok(WorkerEnd::Stopped);
    }
    child.wait().map(WorkerEnd::Exited)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::protocol::EngineBuild;
    use crate::worker::transport::{
        channel_pair, ChannelTransport, MessageReceiver as _, MessageSender as _, Transport as _,
    };
    use std::cell::{Cell, RefCell};

    type HostEnd = ChannelTransport<WorkerMessage, HostMessage>;
    type Retire<'a> = Box<dyn FnOnce(WorkerReport) -> Result<WorkerEnd, String> + 'a>;

    fn manifest(kernel_cache: Option<std::path::PathBuf>) -> ExecutionManifest {
        crate::options::tests::manifest_with_kernel_cache(kernel_cache)
    }

    /// A Prepare worker that answers the host's request with `script`.
    fn scripted_prepare<'a>(
        script: Vec<WorkerMessage>,
        events: &'a RefCell<Vec<&'static str>>,
    ) -> impl FnMut() -> Result<WorkerProcess<HostEnd, Retire<'a>>, LoadError> + 'a {
        let mut script = Some(script);
        move || {
            let script = script.take().expect("scripted_prepare called once");
            events.borrow_mut().push("prepare started");
            let (host_end, worker_end) = channel_pair();
            std::thread::spawn(move || {
                let (mut receiver, mut sender) = worker_end.split();
                let Ok(Some(HostMessage::Hello { .. })) = receiver.receive() else {
                    panic!("the host greets first");
                };
                let Ok(Some(HostMessage::Prepare { .. })) = receiver.receive() else {
                    panic!("the host asks this worker for a Prepare");
                };
                sender
                    .send(WorkerMessage::Hello {
                        build: EngineBuild::current(),
                    })
                    .unwrap();
                for message in script {
                    sender.send(message).unwrap();
                }
            });
            Ok(WorkerProcess {
                transport: host_end,
                retire: Box::new(move |report| {
                    events.borrow_mut().push(match report {
                        WorkerReport::Reported => "prepare waited for",
                        WorkerReport::Ended => "prepare waited for its end",
                        WorkerReport::Unreported => "prepare killed",
                    });
                    Ok(WorkerEnd::Unobserved)
                }) as Retire<'a>,
            })
        }
    }

    /// Loads that record the mode of each attempt and stop unprepared while
    /// `missing` units lack a record and the load reports them.
    fn attempts<'a>(
        missing: &'a Cell<bool>,
        events: &'a RefCell<Vec<&'static str>>,
    ) -> impl FnMut(UnpreparedUnits) -> Result<LoadAttempt<&'static str>, LoadError> + 'a {
        move |unprepared| {
            Ok(match (unprepared, missing.get()) {
                (UnpreparedUnits::Report, true) => {
                    events.borrow_mut().push("load reported unprepared");
                    LoadAttempt::Unprepared
                }
                (UnpreparedUnits::Report, false) => {
                    events.borrow_mut().push("load (reporting) loaded");
                    LoadAttempt::Loaded(Box::new("stored records"))
                }
                (UnpreparedUnits::ServeDefaults, true) => {
                    events.borrow_mut().push("load (defaults) loaded");
                    LoadAttempt::Loaded(Box::new("canonical defaults"))
                }
                (UnpreparedUnits::ServeDefaults, false) => {
                    events.borrow_mut().push("load (defaults) loaded");
                    LoadAttempt::Loaded(Box::new("stored records"))
                }
            })
        }
    }

    /// Complete records: the first load is the load. No Prepare is started
    /// and nothing else is paid.
    #[test]
    fn a_load_with_every_record_stored_is_one_load() {
        let (missing, events) = (Cell::new(false), RefCell::new(Vec::new()));
        let (loaded, preparation) = load_prepared(
            &manifest(Some("kernels".into())),
            attempts(&missing, &events),
            || -> Result<WorkerProcess<HostEnd, Retire<'static>>, _> {
                panic!("no Prepare is started when no record is missing")
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(loaded, "stored records");
        assert!(matches!(preparation, Preparation::Unneeded));
        assert_eq!(*events.borrow(), ["load (reporting) loaded"]);
    }

    /// Missing records: the reporting load ends unprepared, the Prepare
    /// runs to its end with its progress reported, its process, which
    /// reported, is waited for and not killed, and only then is the load
    /// started again, serving what is stored.
    #[test]
    fn a_first_load_is_a_prepare_and_then_a_load() {
        let (missing, events) = (Cell::new(true), RefCell::new(Vec::new()));
        let tuning = LoadProgress::Tuning {
            completed: 1,
            total: 2,
            work: None,
            measured_configurations: 0,
            formed_programs: 0,
        };
        let mut reported = Vec::new();
        let (loaded, preparation) = load_prepared(
            &manifest(Some("kernels".into())),
            attempts(&missing, &events),
            scripted_prepare(
                vec![
                    WorkerMessage::LoadProgress { progress: tuning.clone() },
                    WorkerMessage::Prepared,
                ],
                &events,
            ),
            |progress| {
                // The Prepare published the records.
                missing.set(false);
                reported.push(progress);
            },
        )
        .unwrap();
        assert_eq!(loaded, "stored records");
        assert!(matches!(preparation, Preparation::Prepared));
        assert_eq!(reported, [tuning]);
        assert_eq!(
            *events.borrow(),
            [
                "load reported unprepared",
                "prepare started",
                "prepare waited for",
                "load (defaults) loaded"
            ]
        );
    }

    /// A Prepare that fails, is killed, or cannot be started never fails the
    /// load: its process is still retired first, and the load serves
    /// canonical defaults. One that reported its failure is waited for, as
    /// is one whose stream ended without an outcome, which only its
    /// process's end closes.
    #[test]
    fn an_unfinished_prepare_leaves_the_load_to_canonical_defaults() {
        let failure = LoadError::Internal {
            reason: "a canonical kernel did not form".into(),
        };
        // Reported failure, then a transport that closes without an outcome.
        for (script, retired) in [
            (
                vec![WorkerMessage::LoadFailed {
                    error: failure.clone(),
                }],
                "prepare waited for",
            ),
            (Vec::new(), "prepare waited for its end"),
        ] {
            let (missing, events) = (Cell::new(true), RefCell::new(Vec::new()));
            let (loaded, preparation) = load_prepared(
                &manifest(Some("kernels".into())),
                attempts(&missing, &events),
                scripted_prepare(script, &events),
                |_| {},
            )
            .unwrap();
            assert_eq!(loaded, "canonical defaults");
            assert!(matches!(preparation, Preparation::Incomplete(_)));
            assert_eq!(
                events.borrow()[1..],
                ["prepare started", retired, "load (defaults) loaded"]
            );
        }
        // A host that cannot start a worker.
        let (missing, events) = (Cell::new(true), RefCell::new(Vec::new()));
        let (loaded, preparation) = load_prepared(
            &manifest(Some("kernels".into())),
            attempts(&missing, &events),
            || -> Result<WorkerProcess<HostEnd, Retire<'static>>, _> {
                Err(LoadError::WorkerLost {
                    reason: "no worker executable".into(),
                })
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(loaded, "canonical defaults");
        assert!(matches!(preparation, Preparation::Incomplete(_)));
    }

    /// A worker process that reported its outcome is waited for and exits
    /// as it chose to, however long that takes; one that reported nothing is
    /// killed.
    #[cfg(unix)]
    #[test]
    fn a_worker_that_reported_is_waited_for_and_one_that_did_not_is_killed() {
        let shell = |script: &str| Command::new("/bin/sh").args(["-c", script]).spawn().unwrap();
        // Still running when its host retires it: it finishes on its own.
        let code = |end: WorkerEnd| match end {
            WorkerEnd::Exited(status) => status.code(),
            end => panic!("{end:?}"),
        };
        let reported = ended(shell("sleep 0.3; exit 7"), WorkerReport::Reported).unwrap();
        assert_eq!(code(reported), Some(7));
        // No outcome: it is ended, by its host.
        let unreported = ended(shell("sleep 60"), WorkerReport::Unreported).unwrap();
        assert_eq!(unreported, WorkerEnd::Stopped);
        // One that reported nothing and is already gone is left as it ended.
        let mut gone = shell("exit 3");
        while gone.try_wait().unwrap().is_none() {
            std::thread::yield_now();
        }
        let gone = ended(gone, WorkerReport::Unreported).unwrap();
        assert_eq!(code(gone), Some(3));
    }

    /// A Prepare worker killed before it reported (as the system kills one
    /// under memory pressure) is a worker lost whose reason names the signal
    /// that ended it: a kill leaves no other trace, no diagnostics and no
    /// exit code.
    #[cfg(unix)]
    #[test]
    fn a_prepare_worker_killed_names_its_signal() {
        // It takes the host's request into its pipe, then kills itself.
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "sleep 0.5; kill -9 $$"]);
        let preparation = prepare(
            &manifest(Some("kernels".into())),
            || start_worker_process(&mut command),
            |_| {},
        )
        .unwrap();
        let Preparation::Incomplete(LoadError::WorkerLost { reason }) = preparation else {
            panic!("a killed Prepare is a worker lost: {preparation:?}");
        };
        assert!(reason.contains("SIGKILL"), "the reason names the signal: {reason}");
    }

    /// Without a kernel cache no record can cross to the load's process:
    /// the load serves defaults at once, and never reports.
    #[test]
    fn a_load_without_a_kernel_cache_serves_defaults_directly() {
        let (missing, events) = (Cell::new(true), RefCell::new(Vec::new()));
        let (loaded, preparation) = load_prepared(
            &manifest(None),
            attempts(&missing, &events),
            || -> Result<WorkerProcess<HostEnd, Retire<'static>>, _> {
                panic!("no Prepare is started without a kernel cache")
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(loaded, "canonical defaults");
        assert!(matches!(preparation, Preparation::Unshared));
        assert_eq!(*events.borrow(), ["load (defaults) loaded"]);
    }

    /// A Prepare process that cannot be shown gone fails the load: it would
    /// run beside a search.
    #[test]
    fn an_unretired_prepare_process_fails_the_load() {
        let (missing, events) = (Cell::new(true), RefCell::new(Vec::new()));
        let outcome = load_prepared(
            &manifest(Some("kernels".into())),
            attempts(&missing, &events),
            || {
                let (host_end, worker_end) = channel_pair();
                drop(worker_end);
                Ok(WorkerProcess {
                    transport: host_end,
                    retire: |_| Err("still running".to_owned()),
                })
            },
            |_| {},
        );
        assert!(matches!(outcome, Err(LoadError::WorkerLost { .. })));
        assert_eq!(*events.borrow(), ["load reported unprepared"]);
    }

    /// `run_priced` owns preview settling, re-previewing after waiting for
    /// the device (releasing the device before settling to avoid deadlocking),
    /// retrying `LoadError::PricesPending`, and propagating when settling
    /// makes no progress.
    #[test]
    fn run_priced_releases_device_before_settling_and_terminates_without_progress() {
        let events = RefCell::new(Vec::new());
        let device_held = Cell::new(false);
        let pending_a = crate::error::PendingPrices::new(magnitude_executor::assessment::Outstanding {
            standing: vec![magnitude_executor::assessment::PointRequest::GraphCost],
            loads: Vec::new(),
        })
        .unwrap();
        let pending_b = crate::error::PendingPrices::new(magnitude_executor::assessment::Outstanding {
            standing: vec![magnitude_executor::assessment::PointRequest::TierCopy],
            loads: Vec::new(),
        })
        .unwrap();
        let mut previews = std::collections::VecDeque::from([
            Err(PreviewError::PricesPending { reason: "cold".into(), pending: pending_a.clone() }),
            Ok("plan-1"),
            Err(PreviewError::PricesPending { reason: "changed after wait".into(), pending: pending_b.clone() }),
            Ok("plan-2"),
            Err(PreviewError::PricesPending { reason: "invalidated".into(), pending: pending_a.clone() }),
            Ok("plan-3"),
        ]);
        let mut waited_seq = std::collections::VecDeque::from([true, false, false]);
        let run_pending = Cell::new(true);
        let (loaded, preview, device) = crate::preview::now(run_priced(
            || {
                let next = previews.pop_front().expect("preview");
                events.borrow_mut().push(if next.is_ok() { "preview ok" } else { "preview pending" });
                std::future::ready(next)
            },
            || {
                assert!(!device_held.get(), "settle must never run while holding the device");
                events.borrow_mut().push("settle");
                std::future::ready(Ok::<(), &'static str>(()))
            },
            |&_plan| {
                assert!(!device_held.replace(true), "acquire while already held");
                let waited = waited_seq.pop_front().expect("acquire");
                events.borrow_mut().push(if waited { "acquire waited" } else { "acquire" });
                std::future::ready(Ok(AcquiredDevice { device: "share", waited }))
            },
            |share| {
                assert_eq!(share, "share");
                assert!(device_held.replace(false), "release without device held");
                events.borrow_mut().push("release");
            },
            "state-token",
            |&plan, &share, token| {
                assert_eq!((share, token), ("share", "state-token"));
                assert!(device_held.get(), "run requires held device");
                if run_pending.replace(false) {
                    assert_eq!(plan, "plan-2");
                    events.borrow_mut().push("run pending");
                    return std::future::ready(Err(PricedFailure::Load(
                        LoadError::PricesPending {
                            reason: "worker pending".into(),
                            pending: pending_a.clone(),
                        },
                        "worker-1",
                        token,
                    )));
                }
                assert_eq!(plan, "plan-3");
                events.borrow_mut().push("run ok");
                std::future::ready(Ok("loaded"))
            },
        ))
        .unwrap();
        assert_eq!((loaded, preview, device), ("loaded", "plan-3", "share"));
        assert_eq!(
            *events.borrow(),
            [
                "preview pending",
                "settle",
                "preview ok",
                "acquire waited",
                "preview pending",
                "release",
                "settle",
                "preview ok",
                "acquire",
                "run pending",
                "release",
                "preview pending",
                "settle",
                "preview ok",
                "acquire",
                "run ok",
            ]
        );

        // Termination by construction (worker pending, preview not pending):
        // settle does not run and the worker's PricesPending propagates.
        let settled = Cell::new(false);
        let outcome = crate::preview::now(run_priced(
            || std::future::ready(Ok("plan")),
            || {
                settled.set(true);
                std::future::ready(Ok::<(), &'static str>(()))
            },
            |&_plan| std::future::ready(Ok(AcquiredDevice::immediate("share"))),
            |_share| {},
            (),
            |&_plan, &_share, ()| {
                std::future::ready(Err::<(), _>(PricedFailure::Load(
                    LoadError::PricesPending {
                        reason: "still pending".into(),
                        pending: pending_a.clone(),
                    },
                    "worker-2",
                    (),
                )))
            },
        ));
        assert!(!settled.get());
        assert!(matches!(
            outcome,
            Err(PricedError::Load(LoadError::PricesPending { .. }, "worker-2"))
        ));

        // Termination by construction (preview pending unchanged after settle):
        // settles once, observes the same pending set, and propagates
        // PreviewError::PricesPending instead of looping forever.
        let settles = Cell::new(0);
        let outcome = crate::preview::now(run_priced(
            || {
                std::future::ready(Err::<&str, _>(PreviewError::PricesPending {
                    reason: "unmeasurable".into(),
                    pending: pending_a.clone(),
                }))
            },
            || {
                settles.set(settles.get() + 1);
                std::future::ready(Ok::<(), &'static str>(()))
            },
            |&_plan| std::future::ready(Ok(AcquiredDevice::immediate("share"))),
            |_share| {},
            (),
            |&_plan, &_share, ()| std::future::ready(Ok::<_, PricedFailure<&str, (), ()>>(())),
        ));
        assert_eq!(settles.get(), 1);
        assert!(matches!(
            outcome,
            Err(PricedError::Preview(PreviewError::PricesPending { .. }))
        ));
    }

    /// A Prepare that reports `PricesPending` propagates it as `Err` and
    /// never becomes `Preparation::Incomplete`.
    #[test]
    fn pending_prepare_never_becomes_incomplete() {
        let events = RefCell::new(Vec::new());
        let outcome = prepare(
            &manifest(Some("kernels".into())),
            scripted_prepare(
                vec![WorkerMessage::LoadFailed {
                    error: LoadError::PricesPending {
                        reason: "standing figures pending".into(),
                        pending: crate::error::PendingPrices::new(
                            magnitude_executor::assessment::Outstanding {
                                standing: vec![magnitude_executor::assessment::PointRequest::GraphCost],
                                loads: Vec::new(),
                            },
                        )
                        .unwrap(),
                    },
                }],
                &events,
            ),
            |_| {},
        );
        assert!(matches!(outcome, Err(LoadError::PricesPending { .. })));
    }
}
