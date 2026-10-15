//! Formations run in worker processes admitted by measured memory, through
//! the ledger of what earlier runs established.
//!
//! A worker forms one program and exits ([`crate::worker`]), so its memory
//! is returned and its peak is the formation's alone. The pool starts one
//! when fewer run than the host has cores (formation is CPU-bound) and the
//! host's available memory, less what running workers may still grow by,
//! holds another worker's estimated peak. One worker always runs, so the
//! pool never stalls; until a peak is known, one runs alone. The estimate is
//! the largest peak known, measured in this run or kept by the ledger. A
//! worker that ends abnormally forms again alone: the kernel's out-of-memory
//! killer chooses the largest process, a worker rather than this driver, and
//! a worker killed while alone has a program this host cannot form.

use crate::ledger::{Established, Key, Ledger, Verdict as Kept};
use seismic::coverage::{Formation, Forming, ProgramSource, Target, Verdict};
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

/// How the pool starts a worker.
pub struct WorkerCommand {
    pub program: PathBuf,
    pub arguments: Vec<OsString>,
}

pub struct Pool {
    ledger: Ledger,
    worker: WorkerCommand,
    memory: Option<seismic::DeviceCatalog>,
    admission: Admission,
    /// Formations a worker established in this run.
    formed: AtomicUsize,
    /// Formations whose verdict the ledger already held.
    recalled: AtomicUsize,
}

impl Pool {
    pub fn new(ledger: Ledger, worker: WorkerCommand) -> Self {
        let memory = seismic::DeviceCatalog::discover()
            .map_err(|error| {
                eprintln!("host memory is unobservable ({error:?}); admitting workers by cores")
            })
            .ok();
        let admission = Admission::new(ledger.largest_peak());
        Self {
            ledger,
            worker,
            memory,
            admission,
            formed: AtomicUsize::new(0),
            recalled: AtomicUsize::new(0),
        }
    }

    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    /// What the workers and ledger have done so far, and what is running.
    pub fn activity(&self) -> String {
        format!(
            "{} formed by workers, {} known from the ledger, {} running",
            self.formed.load(Ordering::Relaxed),
            self.recalled.load(Ordering::Relaxed),
            self.admission.running(),
        )
    }

    /// What this run's workers and ledger did, and the memory they took.
    pub fn report(&self) -> String {
        let gib = |bytes: Option<u64>| {
            bytes.map_or("unmeasured".to_owned(), |bytes| {
                format!("{:.2} GiB", bytes as f64 / f64::from(1 << 30))
            })
        };
        format!(
            "{} formed by workers, {} known from the ledger; largest worker peak {}, driver peak {}",
            self.formed.load(Ordering::Relaxed),
            self.recalled.load(Ordering::Relaxed),
            gib(self.admission.largest_peak()),
            gib(process::driver_peak_bytes()),
        )
    }

    /// Run one worker over `request` once admitted: `alone`, with no other
    /// worker running.
    fn attempt(&self, request: &[u8], alone: bool) -> Ended {
        let slot = self.admission.admit(alone, || self.available_bytes());
        let ended = self.run(request, &slot);
        self.admission.release(slot, ended.peak());
        ended
    }

    fn run(&self, request: &[u8], slot: &Slot) -> Ended {
        use std::io::{Read, Write};
        let spawned = Command::new(&self.worker.program)
            .args(&self.worker.arguments)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn();
        let mut child = match spawned {
            Ok(child) => child,
            Err(error) => {
                return Ended::Abnormally {
                    description: format!("could not start: {error}"),
                    killed: false,
                };
            }
        };
        self.admission.started(slot, &child);
        // A worker that dies before reading its request ends the write; how
        // it ended is what reports it.
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(request).ok();
        }
        let mut reply = Vec::new();
        if let Some(mut stdout) = child.stdout.take() {
            stdout.read_to_end(&mut reply).ok();
        }
        // The slot refers to the process until it is released, after this
        // returns; the child is dropped only then.
        let (status, peak) = match process::wait_with_peak(&mut child) {
            Ok(waited) => waited,
            Err(error) => {
                return Ended::Abnormally {
                    description: format!("could not be waited for: {error}"),
                    killed: false,
                };
            }
        };
        let decoded = status
            .success()
            .then(|| postcard::from_bytes::<Verdict>(&reply).ok())
            .flatten();
        match decoded {
            Some(verdict) => Ended::Replied {
                verdict,
                peak_bytes: peak.unwrap_or(0),
            },
            None => Ended::Abnormally {
                description: describe(status),
                killed: process::killed(status),
            },
        }
    }

    /// What this process may still allocate: the host's headroom within any
    /// cgroup limit, or `None` if it cannot be observed.
    fn available_bytes(&self) -> Option<u64> {
        let status = self.memory.as_ref()?.host_memory_status().ok()?;
        let limited = status
            .limits
            .iter()
            .filter(|limit| {
                matches!(
                    limit.kind,
                    seismic::ProcessLimitKind::CgroupV1 { .. }
                        | seismic::ProcessLimitKind::CgroupV2 { .. }
                )
            })
            .map(seismic::ProcessMemoryLimit::remaining_bytes)
            .min();
        Some(limited.map_or(status.headroom.bytes, |limited| {
            limited.min(status.headroom.bytes)
        }))
    }
}

impl Forming for Pool {
    fn form(&self, formation: Formation<'_>) -> Verdict {
        let key = formation_key(formation.namespace, &formation.material, formation.program);
        if let Some(kept) = self.ledger.verdict(&key) {
            self.recalled.fetch_add(1, Ordering::Relaxed);
            return match kept.established {
                Established::Formed => Verdict::Formed,
                Established::Rejected => Verdict::Rejected,
            };
        }
        let request = postcard::to_allocvec(&RequestRef {
            target: formation.target,
            program: formation.program,
        })
        .expect("a formation request encodes");
        let replied = match self.attempt(&request, false) {
            Ended::Replied {
                verdict,
                peak_bytes,
            } => Ok((verdict, peak_bytes)),
            Ended::Abnormally { .. } => match self.attempt(&request, true) {
                Ended::Replied {
                    verdict,
                    peak_bytes,
                } => Ok((verdict, peak_bytes)),
                Ended::Abnormally {
                    description,
                    killed: true,
                } => Err(format!(
                    "the worker was killed while forming alone ({description}): this host's memory cannot form it"
                )),
                Ended::Abnormally { description, .. } => {
                    Err(format!("the worker {description} while forming alone"))
                }
            },
        };
        match replied {
            Ok((verdict, peak_bytes)) => {
                self.formed.fetch_add(1, Ordering::Relaxed);
                let established = match verdict {
                    Verdict::Formed => Some(Established::Formed),
                    Verdict::Rejected => Some(Established::Rejected),
                    Verdict::Failed(_) => None,
                };
                if let Some(established) = established {
                    self.ledger.record(
                        &key,
                        Kept {
                            established,
                            peak_bytes,
                        },
                    );
                }
                verdict
            }
            Err(message) => Verdict::Failed(message),
        }
    }
}

/// The ledger key of forming `program` with the toolchain `namespace`
/// identified by `material`: everything the formation reads.
fn formation_key(namespace: &str, material: &str, program: &ProgramSource) -> Key {
    let entries = program
        .entries
        .iter()
        .map(|entry| {
            format!(
                "{}@{:?}#{:?}",
                entry.symbol, entry.group_size, entry.constants
            )
        })
        .collect::<Vec<_>>()
        .join("\t");
    Key::of(&[
        b"formation",
        namespace.as_bytes(),
        material.as_bytes(),
        program.text.as_bytes(),
        entries.as_bytes(),
    ])
}

/// [`crate::worker::Request`], encoded without copying the program.
#[derive(serde::Serialize)]
struct RequestRef<'a> {
    target: Target,
    program: &'a ProgramSource,
}

enum Ended {
    Replied {
        verdict: Verdict,
        peak_bytes: u64,
    },
    /// Killed, crashed, or exited without a reply: nothing established.
    Abnormally {
        description: String,
        killed: bool,
    },
}

impl Ended {
    fn peak(&self) -> Option<u64> {
        match self {
            Self::Replied { peak_bytes, .. } => Some(*peak_bytes),
            Self::Abnormally { .. } => None,
        }
    }
}

fn describe(status: ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("ended by signal {signal}");
        }
    }
    match status.code() {
        Some(0) => "exited without a reply".to_owned(),
        Some(code) => format!("exited with status {code}"),
        None => "ended without a status".to_owned(),
    }
}

/// A worker admitted to run.
struct Slot {
    id: u64,
}

struct Running {
    id: u64,
    /// The worker's process, once started.
    process: Option<process::Handle>,
}

#[derive(Default)]
struct State {
    running: Vec<Running>,
    largest_peak: Option<u64>,
    /// Workers waiting to run alone. While one waits, no other is admitted.
    alone_waiting: usize,
    /// Whether the running worker runs alone.
    alone_running: bool,
    next_id: u64,
}

struct Admission {
    cores: usize,
    state: Mutex<State>,
    changed: Condvar,
}

/// How often a waiting admission measures again: running workers grow and
/// other programs allocate without telling the pool.
const RESAMPLE: Duration = Duration::from_millis(200);

impl Admission {
    fn new(known_peak: Option<u64>) -> Self {
        Self::with_cores(
            std::thread::available_parallelism().map_or(1, usize::from),
            known_peak,
        )
    }

    fn with_cores(cores: usize, known_peak: Option<u64>) -> Self {
        Self {
            cores,
            state: Mutex::new(State {
                largest_peak: known_peak,
                ..State::default()
            }),
            changed: Condvar::new(),
        }
    }

    fn largest_peak(&self) -> Option<u64> {
        self.lock().largest_peak
    }

    fn running(&self) -> usize {
        self.lock().running.len()
    }

    /// Wait until a worker may run: `alone`, or as `available` memory admits.
    fn admit(&self, alone: bool, available: impl Fn() -> Option<u64>) -> Slot {
        let mut state = self.lock();
        if alone {
            state.alone_waiting += 1;
        }
        loop {
            let admitted = if state.alone_running {
                false
            } else if alone {
                state.running.is_empty()
            } else if state.alone_waiting > 0 {
                false
            } else if state.running.is_empty() {
                true
            } else {
                state.running.len() < self.cores
                    && state
                        .largest_peak
                        .is_some_and(|estimate| fits(&state.running, estimate, available()))
            };
            if admitted {
                break;
            }
            state = self
                .changed
                .wait_timeout(state, RESAMPLE)
                .expect("no thread panics holding the admission")
                .0;
        }
        if alone {
            state.alone_waiting -= 1;
            state.alone_running = true;
        }
        let id = state.next_id;
        state.next_id += 1;
        state.running.push(Running { id, process: None });
        Slot { id }
    }

    fn started(&self, slot: &Slot, child: &Child) {
        let mut state = self.lock();
        if let Some(running) = state
            .running
            .iter_mut()
            .find(|running| running.id == slot.id)
        {
            running.process = Some(process::Handle::of(child));
        }
    }

    fn release(&self, slot: Slot, peak: Option<u64>) {
        let mut state = self.lock();
        state.running.retain(|running| running.id != slot.id);
        if state.running.is_empty() {
            state.alone_running = false;
        }
        if let Some(peak) = peak {
            state.largest_peak = Some(state.largest_peak.map_or(peak, |largest| largest.max(peak)));
        }
        drop(state);
        self.changed.notify_all();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .expect("no thread panics holding the admission")
    }
}

/// Whether `available` memory, less what each running worker may still grow
/// by, holds another worker at `estimate`. Unobservable memory admits by
/// cores alone.
fn fits(running: &[Running], estimate: u64, available: Option<u64>) -> bool {
    let Some(available) = available else {
        return true;
    };
    let reserved: u64 = running
        .iter()
        .map(|running| {
            let resident = running
                .process
                .as_ref()
                .and_then(process::Handle::resident_bytes)
                .unwrap_or(0);
            estimate.saturating_sub(resident)
        })
        .sum();
    available.saturating_sub(reserved) >= estimate
}

/// The memory of worker processes and of this one.
mod process {
    use std::process::{Child, ExitStatus};

    /// A running worker, for reading its resident memory.
    #[derive(Clone, Copy)]
    pub(super) struct Handle {
        #[cfg(unix)]
        pid: libc::pid_t,
        /// The process handle, valid while the worker's slot is held.
        #[cfg(windows)]
        raw: usize,
    }

    impl Handle {
        pub(super) fn of(child: &Child) -> Self {
            #[cfg(unix)]
            {
                Self {
                    pid: child.id() as libc::pid_t,
                }
            }
            #[cfg(windows)]
            {
                use std::os::windows::io::AsRawHandle;
                Self {
                    raw: child.as_raw_handle() as usize,
                }
            }
        }

        pub(super) fn resident_bytes(&self) -> Option<u64> {
            #[cfg(target_os = "linux")]
            {
                let statm = std::fs::read_to_string(format!("/proc/{}/statm", self.pid)).ok()?;
                let pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
                let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
                Some(pages * u64::try_from(page).ok()?)
            }
            #[cfg(target_os = "macos")]
            {
                let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
                let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
                let written = unsafe {
                    libc::proc_pidinfo(
                        self.pid,
                        libc::PROC_PIDTASKINFO,
                        0,
                        (&raw mut info).cast(),
                        size,
                    )
                };
                (written == size).then_some(info.pti_resident_size)
            }
            #[cfg(windows)]
            {
                counters(self.raw as _).map(|counters| counters.WorkingSetSize as u64)
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
            {
                None
            }
        }
    }

    /// Wait for `child` to end: its status and its peak resident memory.
    #[cfg(unix)]
    pub(super) fn wait_with_peak(child: &mut Child) -> std::io::Result<(ExitStatus, Option<u64>)> {
        use std::os::unix::process::ExitStatusExt;
        let pid = child.id() as libc::pid_t;
        let mut status = 0;
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        loop {
            if unsafe { libc::wait4(pid, &mut status, 0, &mut usage) } == pid {
                break;
            }
            let error = std::io::Error::last_os_error();
            if error.kind() != std::io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
        Ok((ExitStatus::from_raw(status), Some(max_rss_bytes(&usage))))
    }

    #[cfg(windows)]
    pub(super) fn wait_with_peak(child: &mut Child) -> std::io::Result<(ExitStatus, Option<u64>)> {
        use std::os::windows::io::AsRawHandle;
        let status = child.wait()?;
        let peak =
            counters(child.as_raw_handle() as _).map(|counters| counters.PeakWorkingSetSize as u64);
        Ok((status, peak))
    }

    /// Whether `status` is a kill: the signal the out-of-memory killer sends.
    pub(super) fn killed(status: ExitStatus) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            status.signal() == Some(libc::SIGKILL)
        }
        #[cfg(not(unix))]
        {
            let _ = status;
            false
        }
    }

    /// This process's own peak resident memory.
    pub(super) fn driver_peak_bytes() -> Option<u64> {
        #[cfg(unix)]
        {
            let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
            (unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } == 0)
                .then(|| max_rss_bytes(&usage))
        }
        #[cfg(windows)]
        {
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            counters(unsafe { GetCurrentProcess() })
                .map(|counters| counters.PeakWorkingSetSize as u64)
        }
        #[cfg(not(any(unix, windows)))]
        {
            None
        }
    }

    /// `ru_maxrss` in bytes: Linux counts kibibytes, macOS bytes.
    #[cfg(unix)]
    fn max_rss_bytes(usage: &libc::rusage) -> u64 {
        let reported = u64::try_from(usage.ru_maxrss).unwrap_or(0);
        if cfg!(target_os = "macos") {
            reported
        } else {
            reported * 1024
        }
    }

    #[cfg(windows)]
    fn counters(
        process: windows_sys::Win32::Foundation::HANDLE,
    ) -> Option<windows_sys::Win32::System::ProcessStatus::PROCESS_MEMORY_COUNTERS> {
        use windows_sys::Win32::System::ProcessStatus::{
            GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };
        let mut counters: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        (unsafe { GetProcessMemoryInfo(process, &mut counters, size) } != 0).then_some(counters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::coverage::ProgramEntry;

    /// Whether `admit` is still waiting after a few measurements.
    fn waits(thread: &std::thread::ScopedJoinHandle<'_, Slot>) -> bool {
        std::thread::sleep(RESAMPLE * 2);
        !thread.is_finished()
    }

    #[test]
    fn one_worker_runs_alone_until_a_peak_is_known() {
        let admission = Admission::with_cores(4, None);
        let roomy = || Some(u64::MAX);
        let first = admission.admit(false, roomy);
        let second = std::thread::scope(|scope| {
            let second = scope.spawn(|| admission.admit(false, roomy));
            assert!(waits(&second), "no estimate admits no second worker");
            admission.release(first, Some(1));
            second.join().unwrap()
        });
        assert_eq!(admission.largest_peak(), Some(1));
        admission.release(second, Some(5));
        assert_eq!(admission.largest_peak(), Some(5));
    }

    #[test]
    fn cores_bound_admission() {
        let admission = Admission::with_cores(2, Some(1));
        let roomy = || Some(u64::MAX);
        let first = admission.admit(false, roomy);
        let second = admission.admit(false, roomy);
        std::thread::scope(|scope| {
            let third = scope.spawn(|| admission.admit(false, roomy));
            assert!(waits(&third), "a third worker exceeds two cores");
            admission.release(first, Some(1));
            admission.release(third.join().unwrap(), Some(1));
        });
        admission.release(second, Some(1));
    }

    #[test]
    fn memory_bounds_admission_and_one_worker_always_runs() {
        // Each worker is estimated at 10; the host holds 15.
        let admission = Admission::with_cores(8, Some(10));
        let tight = || Some(15);
        let first = admission.admit(false, tight);
        std::thread::scope(|scope| {
            let second = scope.spawn(|| admission.admit(false, tight));
            assert!(
                waits(&second),
                "15 less the first worker's reserve of 10 holds no second"
            );
            admission.release(first, None);
            admission.release(second.join().unwrap(), None);
        });
        // An estimate beyond the host still runs one worker.
        let beyond = Admission::with_cores(8, Some(100));
        let only = beyond.admit(false, tight);
        beyond.release(only, None);
    }

    #[test]
    fn a_worker_waiting_to_run_alone_runs_once_the_others_end() {
        let admission = Admission::with_cores(8, Some(1));
        let roomy = || Some(u64::MAX);
        let first = admission.admit(false, roomy);
        std::thread::scope(|scope| {
            let alone = scope.spawn(|| admission.admit(true, roomy));
            assert!(waits(&alone), "it waits for the running worker");
            let blocked = scope.spawn(|| admission.admit(false, roomy));
            assert!(
                waits(&blocked),
                "no worker starts while one waits to run alone"
            );
            admission.release(first, Some(1));
            let alone = alone.join().unwrap();
            assert!(waits(&blocked), "no worker starts beside one running alone");
            admission.release(alone, Some(1));
            admission.release(blocked.join().unwrap(), Some(1));
        });
    }

    #[test]
    fn a_request_decodes_as_the_worker_reads_it() {
        let program = ProgramSource {
            text: "kernel".into(),
            entries: vec![ProgramEntry::named("k")],
        };
        let encoded = postcard::to_allocvec(&RequestRef {
            target: Target::Cuda { architecture: 80 },
            program: &program,
        })
        .unwrap();
        let decoded: crate::worker::Request = postcard::from_bytes(&encoded).unwrap();
        assert!(matches!(decoded.target, Target::Cuda { architecture: 80 }));
        assert_eq!(decoded.program, program);
    }

    #[test]
    fn formation_keys_cover_every_input() {
        let program = ProgramSource {
            text: "kernel".into(),
            entries: vec![ProgramEntry::named("k")],
        };
        let key = formation_key("cuda", "m", &program);
        assert_eq!(key, formation_key("cuda", "m", &program.clone()));
        assert_ne!(key, formation_key("metal", "m", &program));
        assert_ne!(key, formation_key("cuda", "n", &program));
        let mut changed = program.clone();
        changed.text.push(' ');
        assert_ne!(key, formation_key("cuda", "m", &changed));
        let mut changed = program.clone();
        changed.entries[0].symbol = "j".into();
        assert_ne!(key, formation_key("cuda", "m", &changed));
        let mut changed = program.clone();
        changed.entries[0].group_size = Some([1, 1, 1]);
        assert_ne!(key, formation_key("cuda", "m", &changed));
        let mut changed = program;
        changed.entries[0].constants = vec![1];
        assert_ne!(key, formation_key("cuda", "m", &changed));
    }
}
