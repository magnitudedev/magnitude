//! The process's report on demand: `kill -USR1 <pid>` prints every
//! thread's stack, then the state of every owner that registered one, on
//! stderr. It is the one view of a hang where no debugger attaches (a
//! container that refuses ptrace and other processes' `/proc` stacks).
//!
//! Under the rig, which names a directory in [`STACKS`], the report goes to
//! `stacks-<pid>.txt` there instead, created once the handler is in place:
//! that file tells the rig this process reports on SIGUSR1 (to a process
//! without the handler the signal is fatal), and holds what it reports.
//!
//! An owner registers a report ([`register`]) for as long as it holds the
//! returned [`Registered`]; a report never blocks on the owner (a stuck
//! owner reports that it is held, as the exchange watchdog's witnesses do).
//! A [`Phase`] is an owner whose report is the step it is in: what the
//! load is doing while no step runs.
//!
//! The signal's handler only writes a byte to a pipe, which a reporter
//! thread reads: nothing else runs in the handler. The reporter writes the
//! stacks first, allocating nothing and taking no lock a stuck thread may
//! hold ([`stacks`]), so they are out whatever holds the allocator; the
//! owners' reports, which allocate, follow.

#[cfg(unix)]
mod stacks;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

type Report = Arc<dyn Fn() -> String + Send + Sync>;

struct Owner {
    id: u64,
    name: &'static str,
    report: Report,
}

fn owners() -> &'static Mutex<Vec<Owner>> {
    static OWNERS: OnceLock<Mutex<Vec<Owner>>> = OnceLock::new();
    OWNERS.get_or_init(Mutex::default)
}

/// An owner's registration: its report is printed on demand until this
/// drops.
pub struct Registered(u64);

impl Drop for Registered {
    fn drop(&mut self) {
        owners().lock().expect("the diagnostics' owners").retain(|owner| owner.id != self.0);
    }
}

/// Register `name`'s `report`, printed on demand while the returned
/// registration lives. The report must not block on its owner.
pub fn register(name: &'static str, report: impl Fn() -> String + Send + Sync + 'static) -> Registered {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    owners().lock().expect("the diagnostics' owners").push(Owner { id, name, report: Arc::new(report) });
    Registered(id)
}

/// An owner whose report is the step it is in, as it enters each
/// ([`Self::enter`]), and since when.
pub struct Phase {
    current: Arc<Mutex<(String, std::time::Instant)>>,
    _registered: Registered,
}

impl Phase {
    /// `name`'s phase, entering `first`.
    pub fn new(name: &'static str, first: impl Into<String>) -> Self {
        let current = Arc::new(Mutex::new((first.into(), std::time::Instant::now())));
        let reported = current.clone();
        let registered = register(name, move || match reported.try_lock() {
            Ok(held) => format!("{} for {:?}", held.0, held.1.elapsed()),
            Err(_) => "held by its owner".into(),
        });
        Self { current, _registered: registered }
    }

    /// Enter step `step`.
    pub fn enter(&self, step: impl Into<String>) {
        *self.current.lock().expect("a phase") = (step.into(), std::time::Instant::now());
    }
}

/// Every owner's report, in registration order.
pub(crate) fn report() -> String {
    let owners = owners().lock().expect("the diagnostics' owners").iter().map(|owner| (owner.name, owner.report.clone())).collect::<Vec<_>>();
    let mut text = format!("magnitude-engine: report on demand (process {}):", std::process::id());
    for (name, report) in owners {
        text.push_str(&format!("\n  {name}: {}", report()));
    }
    text
}

/// The environment variable naming the directory a report goes to, the
/// rig's: `stacks-<pid>.txt` in it.
pub const STACKS: &str = "MAGNITUDE_STACKS";

/// Where a report goes: stderr, until [`STACKS`]' file is open.
#[cfg(unix)]
static OUT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(2);

/// Print every thread's stack on SIGUSR1, then every owner's report, for
/// the life of the process; installed once, at its start.
#[cfg(unix)]
pub fn install() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let mut fds = [0 as libc::c_int; 2];
        // SAFETY: a pipe of two descriptors this process owns from now on.
        if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
            panic!("the diagnostics pipe opens: {}", std::io::Error::last_os_error());
        }
        WAKE.store(fds[1] as u64, Ordering::Relaxed);
        let reader = fds[0];
        std::thread::Builder::new().name("diagnostics".into()).spawn(move || {
            let mut byte = 0u8;
            // SAFETY: reading one byte into a local from the pipe's read end.
            while unsafe { libc::read(reader, (&mut byte as *mut u8).cast(), 1) } == 1 {
                let out = OUT.load(Ordering::Acquire);
                stacks::write(out);
                write_all(out, format!("{}\n", report()).as_bytes());
            }
        })
        .expect("the diagnostics reporter thread starts");
        stacks::install();
        // SAFETY: a handler that only writes a byte (`wake`).
        let previous = unsafe { libc::signal(libc::SIGUSR1, wake as extern "C" fn(libc::c_int) as libc::sighandler_t) };
        assert_ne!(previous, libc::SIG_ERR, "the report handler installs: {}", std::io::Error::last_os_error());
        // Only now, the handler in place, may the rig find this process's file.
        if let Some(fd) = stacks_file() {
            OUT.store(fd, Ordering::Release);
        }
    });
}

/// `stacks-<pid>.txt` in the directory [`STACKS`] names, created for
/// appending; none outside the rig. A named directory whose file cannot be
/// created is the rig's own setup failing, which panics.
#[cfg(unix)]
fn stacks_file() -> Option<libc::c_int> {
    use std::os::fd::IntoRawFd;
    let directory = std::env::var_os(STACKS)?;
    let path = std::path::Path::new(&directory).join(format!("stacks-{}.txt", std::process::id()));
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .unwrap_or_else(|error| panic!("the stacks file {} opens: {error}", path.display()));
    Some(file.into_raw_fd())
}

/// Write all of `bytes` to `fd`, as far as it takes them.
#[cfg(unix)]
fn write_all(fd: libc::c_int, mut bytes: &[u8]) {
    while !bytes.is_empty() {
        // SAFETY: writes initialized bytes of the slice to an open descriptor.
        let wrote = unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) };
        if wrote <= 0 {
            return;
        }
        bytes = &bytes[wrote as usize..];
    }
}

#[cfg(not(unix))]
pub fn install() {}

/// The pipe's write end, which the SIGUSR1 handler writes a byte to.
#[cfg(unix)]
static WAKE: AtomicU64 = AtomicU64::new(0);

#[cfg(unix)]
extern "C" fn wake(_: libc::c_int) {
    let byte = 1u8;
    // SAFETY: `write` is async-signal-safe; the descriptor is the pipe's.
    unsafe { libc::write(WAKE.load(Ordering::Relaxed) as libc::c_int, (&byte as *const u8).cast(), 1) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The report holds every registered owner's report while it is
    /// registered, and a phase's current step.
    #[test]
    fn the_report_holds_every_registered_owner() {
        let registered = register("owner-under-test", || "its state".into());
        let phase = Phase::new("phase-under-test", "first");
        phase.enter("second");
        let text = report();
        assert!(text.contains("owner-under-test: its state"), "{text}");
        assert!(text.contains("phase-under-test: second for"), "{text}");
        drop((registered, phase));
        let text = report();
        assert!(!text.contains("owner-under-test") && !text.contains("phase-under-test"), "{text}");
    }
}
