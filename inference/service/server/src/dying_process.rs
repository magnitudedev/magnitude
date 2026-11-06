//! A killed process's death as the system reports it, for what host memory pressure may still
//! expect of it. A death frees what the process holds while its threads run the kernel's teardown
//! (one that takes 14 s to free 107 GB runs throughout), and a process that has ended is reaped
//! at once. So a process whose threads the system shows running, switching or leaving is being
//! reaped, however long that takes, and no time bounds it; nor is the host getting memory back
//! beside it silent, as a driver may free for it on threads of its own.
//!
//! Only a process none of whose threads run or switch, with the host getting nothing back, is
//! one the system says nothing about: a teardown waiting on a driver for a moment and one that
//! will never return look the same at any one look, as no state or count parts them. There alone
//! a bound decides: [`SILENCE`].
//!
//! - Linux reports each thread's state and context switches (`/proc/<pid>/task/*/status`).
//! - macOS reports the task's running threads, thread count, context switches and CPU time
//!   (`PROC_PIDTASKINFO`), and nothing once the task is gone.
//! - Windows documents no thread state or switch count for another process: it reports the
//!   process's CPU time and handle count, so a thread blocked in a driver and one sleeping
//!   between short waits are both silent there.
//!
//! A kernel thread spinning in a teardown that never ends shows as running, and is spared.

use std::time::{Duration, Instant};

use crate::residency::supervisor::RESIDENT_SAMPLE_INTERVAL;

/// The longest a killed CUDA process's reap was seen silent by this module's signs, read
/// continuously: 19.8 ms. A process holding a context and filled device, host and anonymous
/// memory was killed and reaped over 4.9-10.3 s on a unified-memory GB10 (16/16/16 and 32/32/32
/// GiB: longest silent 15.1, 15.4 and 19.8 ms), and over 5.4-14.7 s on a discrete RTX 3090
/// (16/8/8 GiB twice and 20/16/16 GiB: 0.2, 2.4 and 0.0 ms). Nearly every sample showed a thread
/// running. Not measured on Metal or Windows, or with a worker's own teardown.
const LONGEST_SILENT_REAP: Duration = Duration::from_micros(19_800);

/// How long a killed process may be silent before it is taken for hung: ten times the longest
/// silent span seen of a reap, the same margin as the engine's watched waits take over their
/// work's longest time, plus the interval between host memory pressure's looks, at which a
/// silence is seen to begin and end.
pub(crate) const SILENCE: Duration =
    LONGEST_SILENT_REAP.saturating_mul(10).saturating_add(RESIDENT_SAMPLE_INTERVAL);

/// What the system reports of a process at one look: whether a thread of it is running or ready
/// to, and counts that change as it is run or as it comes apart. Nothing reported (none left, or
/// none the system will read) is the default, which never changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Look {
    pub(crate) running: bool,
    pub(crate) activity: [u64; 3],
}

/// What was last seen of a dying process.
#[derive(Clone, Copy, Debug)]
struct Seen {
    activity: [u64; 3],
    /// The most headroom the host has had since the process began dying.
    headroom: u64,
    /// Since when it has been silent.
    since: Instant,
}

/// A process killed and not yet shown reaped, as host memory pressure judges it.
#[derive(Clone, Debug)]
pub(crate) struct DyingProcess {
    pub(crate) pid: u32,
    seen: Option<Seen>,
    /// Its silence past [`SILENCE`] has been reported.
    reported: bool,
}

impl DyingProcess {
    pub(crate) fn new(pid: u32) -> Self {
        Self { pid, seen: None, reported: false }
    }

    /// Whether, seen as `look` at `now` with the host at `headroom`, it may still free what it
    /// holds. The first look begins its silence; each sign of its death since ends it. Headroom
    /// counts only past its highest yet, so memory coming and going elsewhere spares it no longer
    /// than it takes the host to leave the emergency.
    pub(crate) fn giving_way(&mut self, look: Look, now: Instant, headroom: u64) -> bool {
        let seen = self.seen.get_or_insert(Seen { activity: look.activity, headroom, since: now });
        if look.running || look.activity != seen.activity || headroom > seen.headroom {
            *seen = Seen { activity: look.activity, headroom: headroom.max(seen.headroom), since: now };
        }
        let silent = now.saturating_duration_since(seen.since);
        let giving_way = silent < SILENCE;
        if !giving_way && !self.reported {
            self.reported = true;
            tracing::error!(worker.pid = self.pid, ?silent, "a killed process has shown no sign of its death; taking it for hung");
        }
        giving_way
    }
}

/// What the system reports of process `pid` now.
#[cfg(target_os = "linux")]
pub(crate) fn look(pid: u32) -> Look {
    let mut look = Look::default();
    let Ok(tasks) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
        return look;
    };
    for task in tasks.flatten() {
        // A thread gone since the listing has no status: its leaving changes the count.
        let Ok(status) = std::fs::read_to_string(task.path().join("status")) else {
            continue;
        };
        look.activity[1] += 1;
        for line in status.lines() {
            if let Some(state) = line.strip_prefix("State:") {
                look.running |= state.trim_start().starts_with('R');
            } else if let Some(switches) = line.strip_prefix("voluntary_ctxt_switches:").or_else(|| line.strip_prefix("nonvoluntary_ctxt_switches:")) {
                look.activity[0] += switches.trim().parse::<u64>().expect("the kernel's context switch count");
            }
        }
    }
    look
}

/// What the system reports of process `pid` now.
#[cfg(target_os = "macos")]
pub(crate) fn look(pid: u32) -> Look {
    let Ok(pid) = libc::c_int::try_from(pid) else {
        return Look::default();
    };
    let size = size_of::<libc::proc_taskinfo>() as libc::c_int;
    // SAFETY: zeroed is a valid `proc_taskinfo`, of plain integers.
    let mut task: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    // SAFETY: the call writes at most `size` bytes into `task`.
    let read = unsafe { libc::proc_pidinfo(pid, libc::PROC_PIDTASKINFO, 0, (&raw mut task).cast(), size) };
    if read != size {
        return Look::default();
    }
    Look {
        running: task.pti_numrunning > 0,
        activity: [
            // Counts, held signed: their bits change as they do.
            u64::from(task.pti_csw as u32),
            u64::from(task.pti_threadnum as u32),
            task.pti_total_user + task.pti_total_system,
        ],
    }
}

/// What the system reports of process `pid` now.
#[cfg(windows)]
pub(crate) fn look(pid: u32) -> Look {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{GetProcessHandleCount, GetProcessTimes, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    let ticks = |time: FILETIME| u64::from(time.dwHighDateTime) << 32 | u64::from(time.dwLowDateTime);
    // SAFETY: opens a handle closed below, or none.
    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
    if process.is_null() {
        return Look::default();
    }
    let none = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user, mut handles) = (none, none, none, none, 0u32);
    // SAFETY: each pointer is to a live local of the type the call writes.
    let timed = unsafe { GetProcessTimes(process, &raw mut created, &raw mut exited, &raw mut kernel, &raw mut user) };
    // SAFETY: as above.
    let counted = unsafe { GetProcessHandleCount(process, &raw mut handles) };
    // SAFETY: the handle was opened above and is not used again.
    unsafe { CloseHandle(process) };
    Look {
        running: false,
        activity: [
            if timed == 0 { 0 } else { ticks(kernel) },
            if timed == 0 { 0 } else { ticks(user) },
            if counted == 0 { 0 } else { u64::from(handles) },
        ],
    }
}

/// Real processes in the states the rule reads, for this module's tests and the memory
/// supervisor's.
#[cfg(all(test, unix))]
pub(crate) mod real {
    use std::process::{Child, Command, Stdio};

    /// A real process, killed and reaped as it is dropped.
    pub(crate) struct Process(Child);

    impl Process {
        pub(crate) fn pid(&self) -> u32 {
            self.0.id()
        }

        /// Kill it, returning once it is reaped.
        pub(crate) fn reap(mut self) {
            self.0.kill().unwrap();
            self.0.wait().unwrap();
        }
    }

    impl Drop for Process {
        fn drop(&mut self) {
            drop(self.0.kill());
            drop(self.0.wait());
        }
    }

    fn shell(script: &str) -> Process {
        Process(Command::new("sh").args(["-c", script]).stdin(Stdio::null()).spawn().unwrap())
    }

    /// One the system shows running throughout.
    pub(crate) fn running() -> Process {
        shell("while :; do :; done")
    }

    /// One the system shows doing nothing throughout: a sleeping shell.
    pub(crate) fn silent() -> Process {
        // `exec` leaves one process, and the wait sees it past its start's activity.
        let process = shell("exec sleep 3600");
        std::thread::sleep(std::time::Duration::from_millis(200));
        process
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADROOM: u64 = 1 << 30;

    /// Silent from the first look, it gives way until the bound and no longer; running, a change
    /// of count or memory coming back past its highest ends the silence, and memory coming
    /// back short of that does not.
    #[test]
    fn only_silence_past_the_bound_is_hung() {
        let start = Instant::now();
        let quiet = Look { running: false, activity: [7, 2, 0] };
        let just = SILENCE - Duration::from_millis(1);
        let mut dying = DyingProcess::new(1);
        assert!(dying.giving_way(quiet, start, HEADROOM));
        assert!(dying.giving_way(quiet, start + just, HEADROOM), "silent short of the bound");
        assert!(!dying.giving_way(quiet, start + SILENCE, HEADROOM), "silent to the bound");

        let mut at = start;
        let mut dying = DyingProcess::new(1);
        assert!(dying.giving_way(quiet, at, HEADROOM));
        at += just;
        assert!(dying.giving_way(Look { running: true, ..quiet }, at, HEADROOM), "running");
        at += just;
        assert!(dying.giving_way(Look { running: false, activity: [8, 2, 0] }, at, HEADROOM), "a switch");
        at += just;
        assert!(dying.giving_way(Look { running: false, activity: [8, 1, 0] }, at, HEADROOM), "a thread gone");
        at += just;
        let seen = Look { running: false, activity: [8, 1, 0] };
        assert!(dying.giving_way(seen, at, HEADROOM + 1), "memory coming back");
        assert!(!dying.giving_way(seen, at + SILENCE, HEADROOM), "memory falling back is no sign");
    }

    /// A process the system no longer reports: its going is one last sign, and nothing since is
    /// silence, so a look that fails never spares it for long.
    #[test]
    fn nothing_reported_is_silence() {
        let start = Instant::now();
        let mut dying = DyingProcess::new(1);
        assert!(dying.giving_way(Look { running: false, activity: [9, 1, 0] }, start, HEADROOM));
        assert!(dying.giving_way(Look::default(), start + SILENCE, HEADROOM), "its leaving is a sign");
        assert!(!dying.giving_way(Look::default(), start + SILENCE * 2, HEADROOM), "nothing since is silence");
    }

    /// The looks of real processes: a running one gives way however long it runs, a sleeping
    /// one is silent however long it sleeps, and a reaped one reads as nothing.
    #[cfg(unix)]
    #[test]
    fn real_processes_read_as_the_rule_expects() {
        let start = Instant::now();
        let running = real::running();
        let mut dying = DyingProcess::new(running.pid());
        for step in 0..5 {
            std::thread::sleep(Duration::from_millis(20));
            assert!(dying.giving_way(look(running.pid()), start + SILENCE * step, HEADROOM), "a running process, look {step}");
        }

        let silent = real::silent();
        let mut dying = DyingProcess::new(silent.pid());
        let first = look(silent.pid());
        assert_ne!(first, Look::default(), "the system reports a live process");
        assert!(!first.running, "{first:?}");
        assert!(dying.giving_way(first, start, HEADROOM));
        std::thread::sleep(Duration::from_millis(20));
        let later = look(silent.pid());
        assert_eq!(later, first, "a sleeping process shows no sign");
        assert!(!dying.giving_way(later, start + SILENCE, HEADROOM), "silent to the bound");

        let pid = silent.pid();
        silent.reap();
        assert_eq!(look(pid), Look::default(), "a reaped process reports nothing");
    }

    /// A process whose one thread is blocked in the kernel (a vfork's parent, in the
    /// uninterruptible state a teardown waiting on a driver is in) reads as silent. It is killable,
    /// as a teardown hung in a driver is not, and the rule reads nothing of that.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_thread_blocked_in_the_kernel_is_silent_on_linux() {
        extern "C" fn sleep_forever(_: *mut libc::c_void) -> libc::c_int {
            loop {
                // SAFETY: waits for a signal, touching no memory.
                unsafe { libc::pause() };
            }
        }
        const STACK: usize = 1 << 16;
        let mut stack = vec![0u8; STACK];
        let top = stack.as_mut_ptr().wrapping_add(STACK).cast::<libc::c_void>();
        // SAFETY: the child calls only what is safe after a fork, on memory set up before it.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            // SAFETY: as above. The clone's parent blocks in the kernel until its child execs or
            // ends, which it never does; the two are one process group, killed together.
            unsafe {
                libc::setpgid(0, 0);
                libc::clone(sleep_forever, top, libc::CLONE_VM | libc::CLONE_VFORK | libc::SIGCHLD, std::ptr::null_mut());
                libc::_exit(1);
            }
        }
        let state = || std::fs::read_to_string(format!("/proc/{pid}/stat")).ok().and_then(|stat| stat.rsplit(") ").next().and_then(|rest| rest.chars().next()));
        let deadline = Instant::now() + Duration::from_secs(5);
        while state() != Some('D') {
            assert!(Instant::now() < deadline, "never blocked in the kernel: {:?}", state());
            std::thread::sleep(Duration::from_millis(5));
        }
        let start = Instant::now();
        let mut dying = DyingProcess::new(pid as u32);
        let first = look(pid as u32);
        assert!(!first.running, "{first:?}");
        assert!(dying.giving_way(first, start, HEADROOM));
        std::thread::sleep(Duration::from_millis(20));
        assert!(!dying.giving_way(look(pid as u32), start + SILENCE, HEADROOM), "blocked in the kernel to the bound");
        // SAFETY: signals the group made above.
        assert_eq!(unsafe { libc::kill(-pid, libc::SIGKILL) }, 0);
        let mut status = 0;
        // SAFETY: waits for the child forked above.
        assert_eq!(unsafe { libc::waitpid(pid, &raw mut status, 0) }, pid);
    }
}
