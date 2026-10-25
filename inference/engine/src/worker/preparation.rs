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
//! failed or could not run, is served at its canonical default.

use super::client::prepare_worker;
use super::protocol::{HostMessage, LoadProgress, WorkerMessage};
use super::transport::{FramedTransport, HostTransport};
use crate::error::LoadError;
use crate::execution::LoadAttempt;
use crate::options::ExecutionManifest;
use magnitude_executor::UnpreparedUnits;
use std::io::BufReader;
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};

/// A worker process a host started for one request.
pub struct WorkerProcess<T, R> {
    /// The host's end of the process's transport.
    pub transport: T,
    /// Returns once the process no longer exists; an error when that cannot
    /// be established. A worker that reported its outcome exits on its own
    /// and is waited for, never killed: killing it would skip everything a
    /// normal exit completes, the system's compiled-kernel cache its search
    /// wrote among it. Only a worker that reported nothing is ended.
    pub retire: R,
}

/// What a worker had told its host when the host retires it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerReport {
    /// The worker reported its outcome: it is exiting on its own.
    Reported,
    /// The worker reported no outcome: it was lost, broke the protocol, or
    /// is still running.
    Unreported,
}

/// What became of preparation in a [`load_prepared`]. None of these fails
/// the load.
#[derive(Debug)]
pub enum Preparation {
    /// The load found every tuning unit decided: nothing was started
    /// besides it.
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

/// Load `manifest`'s model, having its kernels searched first when records
/// are missing.
///
/// `load` makes one attempt to load, in the mode it is given; when it
/// returns [`LoadAttempt::Unprepared`], nothing of that attempt may remain
/// (a worker process has ended; an in-process attempt has dropped all it
/// opened). `start_prepare` starts a worker process for the model's
/// Prepare, whose progress reaches `prepare_progress`.
///
/// Besides `load`'s own failures, the only failure is a Prepare process
/// that cannot be shown to have ended: a load beside it would share the
/// device with a search. A Prepare that reported its outcome is waited for
/// and never killed ([`WorkerProcess::retire`]).
pub fn load_prepared<L, T, R>(
    manifest: &ExecutionManifest,
    mut load: impl FnMut(UnpreparedUnits) -> Result<LoadAttempt<L>, LoadError>,
    start_prepare: impl FnOnce() -> Result<WorkerProcess<T, R>, LoadError>,
    prepare_progress: impl FnMut(LoadProgress),
) -> Result<(L, Preparation), LoadError>
where
    T: HostTransport,
    R: FnOnce(WorkerReport) -> Result<(), String>,
{
    // Without a kernel cache a Prepare's records reach no load.
    if manifest.kernel_cache.is_none() {
        return Ok((serve_defaults(&mut load)?, Preparation::Unshared));
    }
    if let LoadAttempt::Loaded(loaded) = load(UnpreparedUnits::Report)? {
        return Ok((loaded, Preparation::Unneeded));
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
/// The only failure is a Prepare process that cannot be shown to have
/// ended.
pub fn prepare<T, R>(
    manifest: &ExecutionManifest,
    start: impl FnOnce() -> Result<WorkerProcess<T, R>, LoadError>,
    progress: impl FnMut(LoadProgress),
) -> Result<Preparation, LoadError>
where
    T: HostTransport,
    R: FnOnce(WorkerReport) -> Result<(), String>,
{
    if manifest.kernel_cache.is_none() {
        return Ok(Preparation::Unshared);
    }
    let WorkerProcess { transport, retire } = match start() {
        Ok(process) => process,
        Err(error) => return Ok(Preparation::Incomplete(error)),
    };
    // The host's end of the transport is closed by now: the worker waits on
    // nothing of the host's to exit.
    let prepared = prepare_worker(transport, manifest.clone(), progress);
    let report = match &prepared {
        Ok(_) => WorkerReport::Reported,
        Err(_) => WorkerReport::Unreported,
    };
    retire(report).map_err(|reason| LoadError::WorkerLost {
        reason: format!("the Prepare worker could not be shown to have ended: {reason}"),
    })?;
    Ok(match prepared {
        Ok(Ok(())) => Preparation::Prepared,
        Ok(Err(error)) | Err(error) => Preparation::Incomplete(error),
    })
}

/// The total load: every unit without a record at its canonical default.
fn serve_defaults<L>(
    load: &mut impl FnMut(UnpreparedUnits) -> Result<LoadAttempt<L>, LoadError>,
) -> Result<L, LoadError> {
    match load(UnpreparedUnits::ServeDefaults)? {
        LoadAttempt::Loaded(loaded) => Ok(loaded),
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
    mut command: Command,
) -> Result<
    WorkerProcess<ChildTransport, impl FnOnce(WorkerReport) -> Result<(), String>>,
    LoadError,
> {
    let lost = |reason: String| LoadError::WorkerLost { reason };
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| lost(format!("starting {command:?}: {error}")))?;
    let streams = child.stdin.take().zip(child.stdout.take());
    let Some((stdin, stdout)) = streams else {
        let _ = retire(child, WorkerReport::Unreported);
        return Err(lost("the worker process has no standard streams".into()));
    };
    Ok(WorkerProcess {
        transport: FramedTransport::new(BufReader::new(stdout), stdin),
        retire: move |report| retire(child, report),
    })
}

/// Return once `child` is gone. A worker that reported its outcome is
/// waited for, however long its exit takes: after its report it only drops
/// what it held and exits, with nothing of the host's to wait on, and
/// nothing here bounds that wait, so a worker that reported and then never
/// exits holds its host here. A worker that reported nothing is ended if it
/// still runs.
fn retire(child: Child, report: WorkerReport) -> Result<(), String> {
    ended(child, report)
        .map(drop)
        .map_err(|error| error.to_string())
}

/// How `child` ended, once it is gone ([`retire`]).
fn ended(mut child: Child, report: WorkerReport) -> std::io::Result<ExitStatus> {
    if report == WorkerReport::Unreported {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        // A process that ended between the two calls is gone either way.
        let _ = child.kill();
    }
    child.wait()
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
    type Retire<'a> = Box<dyn FnOnce(WorkerReport) -> Result<(), String> + 'a>;

    fn manifest(kernel_cache: Option<std::path::PathBuf>) -> ExecutionManifest {
        crate::options::tests::manifest_with_kernel_cache(kernel_cache)
    }

    /// A Prepare worker that answers the host's request with `script`.
    fn scripted_prepare<'a>(
        script: Vec<WorkerMessage>,
        events: &'a RefCell<Vec<&'static str>>,
    ) -> impl FnOnce() -> Result<WorkerProcess<HostEnd, Retire<'a>>, LoadError> + 'a {
        move || {
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
                        WorkerReport::Unreported => "prepare killed",
                    });
                    Ok(())
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
                    LoadAttempt::Loaded("stored records")
                }
                (UnpreparedUnits::ServeDefaults, true) => {
                    events.borrow_mut().push("load (defaults) loaded");
                    LoadAttempt::Loaded("canonical defaults")
                }
                (UnpreparedUnits::ServeDefaults, false) => {
                    events.borrow_mut().push("load (defaults) loaded");
                    LoadAttempt::Loaded("stored records")
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
    /// canonical defaults. One that reported its failure is waited for; one
    /// that reported nothing is killed.
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
            (Vec::new(), "prepare killed"),
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
        let reported = ended(shell("sleep 0.3; exit 7"), WorkerReport::Reported).unwrap();
        assert_eq!(reported.code(), Some(7));
        // No outcome: it is ended, by a signal.
        let unreported = ended(shell("sleep 60"), WorkerReport::Unreported).unwrap();
        assert_eq!(unreported.code(), None);
        // One that reported nothing and is already gone is left as it ended.
        let mut gone = shell("exit 3");
        while gone.try_wait().unwrap().is_none() {
            std::thread::yield_now();
        }
        let gone = ended(gone, WorkerReport::Unreported).unwrap();
        assert_eq!(gone.code(), Some(3));
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
}
