//! The probes of device properties (`seismic_lang::device`, a backend's
//! `probes`): what the measurement job stored of them, which opening a
//! device reads, and the job's running of them, which opening never does.
//!
//! Each probe runs in a process of its own, since a probe can crash the
//! driver's compiler: the running executable started again as a probe
//! process (`EXECUTABLE seismic-probe BACKEND DEVICE PROBE`), which every
//! executable that runs the measurement job serves first thing in `main`
//! ([`serve_probe_if_asked`]). What it finds is kept in the device's
//! artifact store, keyed by the executor's identity (its device's UUID, its
//! backend, and its driver as the backend reports it, the version among
//! it), the backend's probe revision and its declared probes: a driver
//! update, or a changed probe, runs the probes again. Until then every probe
//! takes its safe value (`seismic_lang::device`): a load before the job has
//! run claims no capability a probe has not shown, so it may be slower, and
//! where a probe decides a numeric mode (fp32 round-to-nearest-even
//! division) it computes in the device's default mode until the probe holds.

use crate::artifacts::{ArtifactKey, ArtifactStore};
use seismic_lang::device::{ProbeResult, Probes, PROBING};
use seismic_topology::Identity;
use std::io::Read;
use std::path::Path;
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// The artifact namespace of stored probe results: measurement-job output,
/// which a store keeps whatever its use time.
pub const NAMESPACE: &str = "probes";

/// The first argument that starts the running executable as a probe
/// process ([`serve_probe_if_asked`]).
pub const PROBE_COMMAND: &str = "seismic-probe";

/// The longest one probe's process runs. All four Vulkan probes, each in its
/// own process (start and device open included), took 3.1 to 7.9 s
/// together on an RTX 3090 (615.71.09) and 5.8 to 6.7 s on a GB10
/// (580.159.03, with its compiler's trap and core dump); this is 7.5x the
/// longest of those, for slower compilers no host has shown. A probe still
/// running at it has hung (as in the driver) and is stopped, keeping
/// nothing, rather than hold the job until its deadline on every run.
pub const PROBE_LIMIT: Duration = Duration::from_secs(60);

/// The key of the probe results of the executor of `identity`, of a
/// backend whose probes are at `revision`. It covers the declared probes,
/// whose order and types are the stored bytes' layout.
fn key<P: Probes>(identity: &Identity, revision: u32) -> ArtifactKey {
    let identity = postcard::to_allocvec(identity).expect("an executor identity serializes");
    let layout = P::PROBES.iter().map(|(name, kind)| format!("{name}:{kind:?}")).collect::<Vec<_>>().join(";");
    ArtifactKey::of(&[NAMESPACE.as_bytes(), &revision.to_le_bytes(), layout.as_bytes(), &identity])
}

/// What `store` holds of the probes of the executor of `identity`; every
/// probe not run, or without a store, takes its safe value. Bytes the store
/// returns were written here under a key that covers their layout: bytes
/// that do not decode break the store's invariant, an error naming the key.
pub(crate) fn stored<P: Probes>(store: Option<&dyn ArtifactStore>, identity: &Identity, revision: u32) -> Result<P, String> {
    let Some(store) = store else {
        return Ok(P::default());
    };
    let key = key::<P>(identity, revision);
    let Some(bytes) = store.get(NAMESPACE, &key) else {
        return Ok(P::default());
    };
    postcard::from_bytes(&bytes).map_err(|error| {
        format!(
            "artifact store invariant broken: the probe results {NAMESPACE}/{} do not decode: {error}",
            key.as_str()
        )
    })
}

/// One probe's run that kept nothing, which the next job runs again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProbeFailure {
    /// The probe could not run, for a reason that says nothing of the
    /// device (its process could not be started, failed to open the device
    /// or panicked, or ended before it began probing), or its process wrote
    /// no result.
    Failed { probe: &'static str, reason: String },
    /// The probe had not ended within its limit ([`PROBE_LIMIT`], or the
    /// job's deadline where that came first) and was stopped.
    Stopped { probe: &'static str, after: Duration },
    /// The job's deadline had passed before the probe was started.
    Unstarted { probe: &'static str },
}

impl std::fmt::Display for ProbeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed { probe, reason } => write!(f, "probe `{probe}`: {reason}"),
            Self::Stopped { probe, after } => write!(f, "probe `{probe}` had not ended after {:.1} s and was stopped", after.as_secs_f64()),
            Self::Unstarted { probe } => write!(f, "probe `{probe}` was not started before the job's deadline"),
        }
    }
}

/// One probe's result kept, with what its process reported of its run on
/// standard error (its measurements, a mismatch, a crash's last words).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeptProbe {
    pub probe: &'static str,
    pub result: ProbeResult<u64>,
    pub report: String,
}

/// What running an executor's probes did: each result kept, and each run
/// that kept nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProbeRun {
    pub kept: Vec<KeptProbe>,
    pub failures: Vec<ProbeFailure>,
}

/// Run, each in a process of `program` (`program seismic-probe DEVICE...
/// PROBE`, `device` naming the backend and its device), the probes `store`
/// holds no result of for the executor of `identity`, keeping each result
/// as it comes. A process writes [`PROBING`] once it has opened the device
/// and begins the probe. One that exits 0 then writes its result's line;
/// one that a crash signal ends after [`PROBING`] ([`crash`], as of the
/// driver's compiler) is kept as the probe's finding: unanswerable, its safe
/// value. Every other end keeps nothing: exit 1 is a failure that says
/// nothing of the device, and a kill (the system's, out of memory, or ours at
/// the probe's limit) or termination says nothing of the probe. Each runs at
/// most [`PROBE_LIMIT`]; none is started at or past `deadline`, and one
/// running at it is stopped.
/// `Err` is a store whose bytes do not decode ([`stored`]).
pub(crate) fn run<P: Probes>(
    store: &dyn ArtifactStore,
    identity: &Identity,
    revision: u32,
    program: &Path,
    device: &[&str],
    deadline: Instant,
) -> Result<ProbeRun, String> {
    let mut probed: P = stored(Some(store), identity, revision)?;
    let mut run = ProbeRun::default();
    for probe in probed.unrun() {
        if Instant::now() >= deadline {
            run.failures.push(ProbeFailure::Unstarted { probe });
            continue;
        }
        let (result, report) = match process(program, device, probe, deadline) {
            Ok(kept) => kept,
            Err(failure) => {
                run.failures.push(failure);
                continue;
            }
        };
        if let Err(reason) = probed.record(probe, result.clone()) {
            run.failures.push(ProbeFailure::Failed { probe, reason });
            continue;
        }
        store.put(NAMESPACE, &key::<P>(identity, revision), &postcard::to_allocvec(&probed).expect("probe results serialize"));
        run.kept.push(KeptProbe { probe, result, report });
    }
    Ok(run)
}

/// One probe's process: its result, or how it kept nothing.
fn process(
    program: &Path,
    device: &[&str],
    probe: &'static str,
    deadline: Instant,
) -> Result<(ProbeResult<u64>, String), ProbeFailure> {
    let failed = |reason: String| ProbeFailure::Failed { probe, reason };
    let mut child = Command::new(program)
        .arg(PROBE_COMMAND)
        .args(device)
        .arg(probe)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| failed(format!("starting {}: {error}", program.display())))?;
    let output = drain(child.stdout.take().expect("stdout is piped"));
    let report = drain(child.stderr.take().expect("stderr is piped"));
    let started = Instant::now();
    let limit = deadline.min(started + PROBE_LIMIT);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() >= limit => {
                // A child not yet waited for can always be killed: one that
                // has ended meanwhile is killed as a no-op.
                child.kill().expect("a probe process not yet waited for is killed");
                return match child.wait() {
                    Ok(_) => Err(ProbeFailure::Stopped { probe, after: started.elapsed() }),
                    Err(error) => Err(failed(format!("waiting for its stopped process: {error}"))),
                };
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(20)),
            Err(error) => return Err(failed(format!("waiting for its process: {error}"))),
        }
    };
    let joined = |pipe: std::thread::JoinHandle<std::io::Result<Vec<u8>>>, name: &str| {
        pipe.join()
            .expect("a pipe's reader does not panic")
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
            .map_err(|error| failed(format!("reading its standard {name}: {error}")))
    };
    let (output, report) = (joined(output, "output")?, joined(report, "error")?);
    // A crash's report can hold a whole compiler log: its end is kept.
    let report = report.trim();
    let report = report.get(report.len().saturating_sub(4096)..).unwrap_or(report).to_owned();
    let mut lines = output.lines();
    let began = lines.next() == Some(PROBING);
    match status.code() {
        Some(0) => lines
            .next()
            .filter(|_| began)
            .and_then(ProbeResult::parse)
            .map(|result| (result, report))
            .ok_or_else(|| failed(format!("its process wrote no result: `{}`", output.trim()))),
        Some(1) => Err(failed(report)),
        _ if !began => Err(failed(format!("its process ended with {status} before it began probing: {report}"))),
        _ => match crash(status) {
            Some(crash) => Ok((ProbeResult::Unanswerable(format!("its process crashed ({crash}): {report}")), String::new())),
            None => Err(failed(format!("its process ended with {status}, which is no crash: {report}"))),
        },
    }
}

/// The crash that ended a process of `status`, which a probe keeps as its
/// finding: a fault, trap or abort (as a driver's compiler ends in), never
/// a kill or termination from outside, which says nothing of the probe.
#[cfg(unix)]
fn crash(status: ExitStatus) -> Option<&'static str> {
    use std::os::unix::process::ExitStatusExt;
    match status.signal()? {
        libc::SIGSEGV => Some("SIGSEGV"),
        libc::SIGTRAP => Some("SIGTRAP"),
        libc::SIGABRT => Some("SIGABRT"),
        libc::SIGILL => Some("SIGILL"),
        libc::SIGBUS => Some("SIGBUS"),
        _ => None,
    }
}

/// The crash that ended a process of `status`: on Windows its exit code is
/// the exception's status (an abort fails fast, as a stack buffer overrun).
#[cfg(windows)]
fn crash(status: ExitStatus) -> Option<&'static str> {
    use windows_sys::Win32::Foundation::{
        STATUS_ACCESS_VIOLATION, STATUS_BREAKPOINT, STATUS_DATATYPE_MISALIGNMENT, STATUS_ILLEGAL_INSTRUCTION,
        STATUS_STACK_BUFFER_OVERRUN,
    };
    match status.code()? {
        STATUS_ACCESS_VIOLATION => Some("access violation"),
        STATUS_BREAKPOINT => Some("breakpoint"),
        STATUS_STACK_BUFFER_OVERRUN => Some("fail fast"),
        STATUS_ILLEGAL_INSTRUCTION => Some("illegal instruction"),
        STATUS_DATATYPE_MISALIGNMENT => Some("misalignment"),
        _ => None,
    }
}

/// Read all of `source` on a thread of its own, so neither of a process's
/// pipes fills while the other is read.
fn drain(mut source: impl Read + Send + 'static) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        source.read_to_end(&mut bytes).map(|_| bytes)
    })
}

/// Serve one probe and exit, where this process was started as a probe
/// process (its first argument is [`PROBE_COMMAND`]); return at once
/// otherwise. Every executable that runs the measurement job calls it first
/// thing in `main`. An executable that does not starts again as itself,
/// never writes [`PROBING`], and so keeps nothing ([`run`]).
pub fn serve_probe_if_asked() {
    let mut arguments = std::env::args_os().skip(1);
    if arguments.next().is_none_or(|first| first != PROBE_COMMAND) {
        return;
    }
    let arguments = arguments.map(|argument| argument.into_string()).collect::<Result<Vec<_>, _>>();
    std::process::exit(match arguments {
        Ok(arguments) => serve(&arguments),
        Err(argument) => {
            eprintln!("the probe argument {argument:?} is not UTF-8");
            1
        }
    })
}

/// The probe process: open the device `BACKEND DEVICE` names, write
/// [`PROBING`], run probe `PROBE`, and write its result's line, exiting 0.
/// A failure that says nothing of the device, a panic included, exits 1 with
/// its reason on standard error. A probe that crashes the driver ends the
/// process otherwise, which its spawner keeps as the probe's finding.
fn serve(arguments: &[String]) -> i32 {
    // A panic is this program's failure, not the device's: it keeps nothing.
    std::panic::set_hook(Box::new(|info| {
        eprintln!("{info}");
        std::process::exit(1);
    }));
    let outcome = (|| -> Result<ProbeResult<u64>, String> {
        let [backend, device, probe] = arguments else {
            return Err(format!("usage: EXECUTABLE {PROBE_COMMAND} BACKEND DEVICE PROBE"));
        };
        match backend.as_str() {
            #[cfg(not(target_os = "macos"))]
            "vulkan" => {
                let device = seismic_vulkan::probe::open(device)?;
                use std::io::Write;
                let mut stdout = std::io::stdout();
                writeln!(stdout, "{PROBING}").and_then(|()| stdout.flush()).map_err(|error| error.to_string())?;
                seismic_vulkan::probe::run(&device, probe)
            }
            other => {
                let _ = (device, probe);
                Err(format!("this build has no probes of backend `{other}`"))
            }
        }
    })();
    match outcome {
        Ok(result) => {
            println!("{}", result.line());
            0
        }
        Err(reason) => {
            eprintln!("{reason}");
            1
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::artifacts::MemoryStore;
    use seismic_lang::device::vulkan::Probed;

    /// A process's answer is kept, and its crash after it began probing is
    /// kept as unanswerable; its failure (exit 1), a crash before it began or
    /// a kill keeps nothing, so the next run asks it again, and only it; a program
    /// that cannot start runs nothing; what is kept is keyed by the
    /// identity, so another driver keeps nothing of it; and stored bytes that
    /// do not decode are an error.
    #[test]
    fn probe_processes_keep_answers_and_crashes_and_retry_failures() {
        let directory = std::env::temp_dir().join(format!("seismic-probes-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let program = directory.join("probe");
        std::fs::write(
            &program,
            "#!/bin/sh\n[ \"$1 $2 $3\" = 'seismic-probe vulkan device' ] || exit 1\ncase \"$4\" in\n  rte_32_division) echo probing; echo 'answer 1' ;;\n  wide_accumulators) echo probing; echo 'the compiler trapped' >&2; kill -SEGV $$ ;;\n  thread_registers) [ -e \"$0.ran\" ] && { echo probing; kill -KILL $$; }; touch \"$0.ran\"; echo 'no device' >&2; kill -SEGV $$ ;;\n  register_file) echo probing; echo 'registers: measured' >&2; echo 'answer 65536' ;;\nesac\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&program, std::fs::Permissions::from_mode(0o755)).unwrap();
        let identity = Identity {
            physical: seismic_topology::PhysicalIdentity::Uuid([7; 16]),
            backend: seismic_topology::BackendName::Vulkan,
            driver: "vulkan driver 4 version 1".into(),
        };
        let store = MemoryStore::default();
        let later = Instant::now() + std::time::Duration::from_secs(30);
        let device = ["vulkan", "device"];
        let first = run::<Probed>(&store, &identity, 1, &program, &device, later).unwrap();
        let kept = first.kept.iter().map(|kept| (kept.probe, kept.result.clone())).collect::<Vec<_>>();
        assert_eq!(kept[0], ("rte_32_division", ProbeResult::Answer(1)));
        assert!(matches!(&kept[1], ("wide_accumulators", ProbeResult::Unanswerable(reason)) if reason.contains("the compiler trapped")));
        assert_eq!(kept[2], ("register_file", ProbeResult::Answer(65536)));
        assert_eq!(first.kept[2].report, "registers: measured");
        assert!(matches!(&first.failures[..], [ProbeFailure::Failed { probe: "thread_registers", reason }] if reason.contains("before it began probing")));
        let probed: Probed = stored(Some(&store), &identity, 1).unwrap();
        assert_eq!(probed.unrun(), ["thread_registers"]);
        let again = run::<Probed>(&store, &identity, 1, &program, &device, later).unwrap();
        assert!(
            again.kept.is_empty() && matches!(&again.failures[..], [ProbeFailure::Failed { probe: "thread_registers", reason }] if reason.contains("no crash")),
            "only the failed probe runs again, and its kill keeps nothing: {:?}",
            again.failures
        );
        let updated = Identity { driver: "vulkan driver 4 version 2".into(), ..identity.clone() };
        assert_eq!(stored::<Probed>(Some(&store), &updated, 1).unwrap().unrun().len(), 4);
        assert_eq!(stored::<Probed>(Some(&store), &identity, 2).unwrap().unrun().len(), 4);
        let absent = run::<Probed>(&store, &updated, 1, &directory.join("absent"), &device, later).unwrap();
        assert!(absent.kept.is_empty() && absent.failures.len() == 4);
        store.put(NAMESPACE, &key::<Probed>(&identity, 1), &[0xff; 3]);
        assert!(stored::<Probed>(Some(&store), &identity, 1).is_err_and(|error| error.contains("do not decode")));
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
