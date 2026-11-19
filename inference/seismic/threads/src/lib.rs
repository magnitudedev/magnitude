//! The threads this process may run at once: its share of the machine.
//!
//! The process's auxiliary pools (memory fault-in, native-target workers,
//! repack, formation) size themselves by [`share`], never by the machine: a
//! process the service starts is handed its share at startup
//! ([`set_share`]), the serving worker its whole [`quota`] and background
//! work less, so its auxiliary pools leave the serving worker room. A
//! device's own pools (the CPU device's, the expert-compute pool) never read
//! the share: their size is part of the device's identity, which
//! a measurement sizes and serving then opens, so they take the machine as
//! the process sees it. A process handed none (a tool, a test) takes its
//! [`quota`].
//!
//! It also decides how a panic ends, the one panic policy of the process. A
//! panic is an invariant violated: the state it leaves is not one the
//! process can trust. A program ([`main`]), a thread nobody waits for
//! ([`spawn`]) and a task nobody waits for ([`detached`]) end the process at
//! the panic, before anything unwinds, so nothing drops, drains or answers
//! from that state: the panic's message and the process's stacks are its
//! report. Work done for a thread that waits for it (a scoped thread,
//! [`spawn_scoped`] and [`run`]; a pool's job, [`Submitter`]; a runtime's
//! task, [`runtime`]) ends a panic as that thread would. A panic unwinds to
//! a caller only on a thread whose owner answers for it (a test harness's),
//! or within a boundary declared [`recoverable`], and arrives there as a
//! [`Panic`], which carries its cause.

#![allow(
    clippy::disallowed_methods,
    reason = "the one place the engine starts a thread or a runtime"
)]

use std::any::Any;
use std::cell::Cell;
use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::task::{Context, Poll};

/// The share, fixed at its first read or its one setting, whichever is
/// first.
static SHARE: OnceLock<NonZeroUsize> = OnceLock::new();

/// The threads this process may run at once: what [`set_share`] set, or,
/// where nothing set it before this first read, the process's [`quota`].
pub fn share() -> NonZeroUsize {
    *SHARE.get_or_init(quota)
}

/// Set this process's share, at its startup, before any pool reads it.
///
/// # Panics
/// Once the share is set or read: a pool sized by another share runs
/// beside the pools sized by this one, and the process takes more of the
/// machine than it was handed.
pub fn set_share(share: NonZeroUsize) {
    if SHARE.set(share).is_err() {
        panic!("invariant violated: the process's thread share is set once, at startup, before any pool reads it");
    }
}

/// The CPUs the process may keep busy at once: its affinity, under a
/// cgroup's CPU quota (`cpu.max`), as `available_parallelism` reads them;
/// one where the OS reports none.
pub fn quota() -> NonZeroUsize {
    std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN)
}

/// The stack of every thread this process starts, and of every runtime
/// worker it builds ([`spawn`], [`spawn_scoped`], [`run`], `runtime`): the
/// one stack size the process decides, never one a platform or library
/// chose for a thread (2 MiB for Rust's and Tokio's, 1 MiB for a Windows
/// main thread).
///
/// It is derived from the two deepest needs a thread has, the larger taken
/// [`STACK_MARGIN`] times and rounded up to a MiB:
/// - the engine's own deepest, a load: a walk of a Seismic program
///   ([`PROGRAM_BYTES`]: decoding the checked program, or parsing,
///   checking, monomorphizing or encoding it) on the load path
///   ([`LOAD_PATH_BYTES`]), 1.9 MiB;
/// - a platform compiler's ([`COMPILER_PEAK_BYTES`]), 2.2 MiB.
///
/// So 9 MiB. It is address space reserved for the thread's life; only the
/// pages a call touches are committed.
pub const STACK_BYTES: usize = {
    let engine = PROGRAM_BYTES + LOAD_PATH_BYTES;
    let need = if engine > COMPILER_PEAK_BYTES { engine } else { COMPILER_PEAK_BYTES };
    (need * STACK_MARGIN).div_ceil(1 << 20) << 20
};

/// The stack one walk of a Seismic program may take: parsing, checking,
/// monomorphizing, encoding or decoding it. Every such walk recurses with
/// the program's nesting, which `seismic-lang` bounds by construction: its
/// parser refuses a body nested past `MAX_NESTING` statements and
/// expressions as it parses, before any other walk sees it, and its checker
/// refuses a checked body nested deeper. So each walk's need is its frames
/// at `MAX_NESTING` levels, `seismic-lang`'s `WALK_STACK_BYTES`: 128 levels
/// at the largest unoptimized frame any walk takes per level, 13 KiB, and
/// 64 KiB for the frames around them, asserted at compile time to stay
/// within this. A test parses, checks, monomorphizes, encodes and decodes
/// programs nested exactly `MAX_NESTING` deep on a thread of exactly this
/// stack (`seismic-lang`'s `nesting_tests`): a walk whose frames outgrow
/// this overflows it.
pub const PROGRAM_BYTES: usize = 1_728 << 10;

/// The load path's own stack beside the program's decode, in an unoptimized
/// build. Measured on 2026-11-17: an LFM3 load filled a 2 MiB thread
/// (2,138,096 B over 351 frames), about 1.9 MB of it six load-path frames
/// holding plans, definitions and the executor domain by value, which those
/// types now box where they are defined; of the remaining 0.24 MB, 147 KB
/// was the decode. This holds the rest with the boxed frames' remaining
/// locals.
pub const LOAD_PATH_BYTES: usize = 256 << 10;

/// The most stack one function's frame may take in an unoptimized build,
/// which clippy's `large_stack_frames`, denied workspace-wide, holds every
/// function to (clippy.toml's `stack-size-threshold`, which a test keeps
/// equal to this). The overflowing load was a few frames holding plans,
/// definitions and the executor domain by value (288 KB to 763 KB each);
/// one type that large held by value again is refused where it is written.
/// A quarter of [`LOAD_PATH_BYTES`]: no one frame takes the load path's
/// budget, which holds every frame of it.
pub const FRAME_BYTES: usize = LOAD_PATH_BYTES / 4;

/// A platform compiler's peak stack: external code whose recursion follows
/// the program and the driver, which nothing here bounds, so a measurement.
/// On an RTX 3090, driver 570.211.01, with its pipeline cache off
/// (2026-11-07): forming `attention_decode_k8v4` (kv 1, g 16, s 512) drove
/// NVIDIA's SPIR-V compiler (`libnvidia-glvkspirv`) through two functions
/// recursing in turn, 640 B and about 390 B a frame, to 2,260,992 B at its
/// peak, of which Seismic's frames above the driver were at most 9,136 B.
pub const COMPILER_PEAK_BYTES: usize = 2_260_992;

/// How many times the larger need [`STACK_BYTES`] holds: the compilers of
/// the drivers and the programs not measured (other vendors' Vulkan, Metal's,
/// a program deeper than `attention_decode_k8v4`), and the targets whose
/// unoptimized frames were not (x86-64), each within it.
pub const STACK_MARGIN: usize = 4;

/// Run `program`, the process's whole work, and return what it returned:
/// every binary's `main` is this and nothing before it but what must run
/// before any thread exists. The program runs on a thread of
/// [`STACK_BYTES`], never on the platform's main thread (whose stack the
/// platform sized), and a panic anywhere in it ends the process at the
/// panic, as on a thread nobody waits for: the process's owner answers for
/// the program by its exit alone.
pub fn main<R: Send>(name: impl Into<String>, program: impl FnOnce() -> R + Send) -> R {
    PANICS.with(|panics| panics.set(Panics::ENDS));
    install();
    std::thread::scope(|scope| {
        builder(name)
            .spawn_scoped(scope, move || Panics::ENDS.on(program))
            .unwrap_or_else(|error| panic!("invariant violated: the host starts the program's thread: {error}"))
            .join()
            .unwrap_or_else(|_| unreachable!("a panic in the program ends the process at the panic"))
    })
}

/// Start a thread named `name`, with [`STACK_BYTES`] of stack, running
/// `work`: every thread of the process starts here. Nobody waits for its
/// work as its own, so a panic on it ends the process.
pub fn spawn<T: Send + 'static>(
    name: impl Into<String>,
    work: impl FnOnce() -> T + Send + 'static,
) -> std::io::Result<std::thread::JoinHandle<T>> {
    install();
    builder(name).spawn(move || Panics::ENDS.on(work))
}

/// [`spawn`] in `scope`: the spawning thread waits for it, so a panic on
/// it ends as one on the spawner would.
pub fn spawn_scoped<'scope, 'env, T: Send + 'scope>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    name: impl Into<String>,
    work: impl FnOnce() -> T + Send + 'scope,
) -> std::io::Result<std::thread::ScopedJoinHandle<'scope, T>> {
    install();
    let panics = Panics::here();
    builder(name).spawn_scoped(scope, move || panics.on(work))
}

/// `work` on a thread of its own ([`spawn_scoped`]) while the caller waits:
/// work whose caller's stack is not the process's to size (a test harness's
/// thread, an embedder's) runs on one that is. A thread is started for each
/// call, which costs microseconds beside the work run this way. A panic in
/// `work` ends as one in the caller would: it continues in a caller it
/// unwinds to.
pub fn run<R: Send>(name: impl Into<String>, work: impl FnOnce() -> R + Send) -> R {
    std::thread::scope(|scope| {
        spawn_scoped(scope, name, work)
            .unwrap_or_else(|error| panic!("invariant violated: the host starts a thread: {error}"))
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
    })
}

fn builder(name: impl Into<String>) -> std::thread::Builder {
    std::thread::Builder::new().name(name.into()).stack_size(STACK_BYTES)
}

/// How a panic on the calling thread ends: the process at the panic, or an
/// unwind to the caller that waits for the work and answers for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Panics {
    ends_process: bool,
}

thread_local! {
    /// The calling thread's [`Panics`]: unwinding on a thread this crate
    /// did not start, whose owner (a test harness, an embedder) answers for
    /// its panics.
    static PANICS: Cell<Panics> = const { Cell::new(Panics::UNWINDS) };
}

impl Panics {
    /// The process ends at the panic.
    const ENDS: Self = Self { ends_process: true };
    /// The panic unwinds to a caller that answers for it.
    const UNWINDS: Self = Self { ends_process: false };

    /// The calling thread's.
    fn here() -> Self {
        PANICS.with(Cell::get)
    }

    /// `work` on the calling thread, a panic in it ending as `self` says.
    fn on<R>(self, work: impl FnOnce() -> R) -> R {
        struct Restore(Panics);
        impl Drop for Restore {
            fn drop(&mut self) {
                PANICS.with(|panics| panics.set(self.0));
            }
        }
        install();
        let _restore = Restore(PANICS.with(|panics| panics.replace(self)));
        work()
    }
}

/// The thread a job is submitted from and that waits for it: what a pool
/// captures as the job is submitted ([`Self::here`]) and runs the job for
/// on its own threads ([`Self::run`]), so the job's panic ends as one on
/// the submitter would. Where the submitter's panics end the process, the
/// job's does, at the panic; where they unwind, the job's arrives as a
/// [`Panic`] the pool hands its submitter to resume.
#[derive(Clone, Copy, Debug)]
pub struct Submitter(Panics);

impl Submitter {
    /// The calling thread, as a job's submitter.
    pub fn here() -> Self {
        Self(Panics::here())
    }

    /// `job`, for this submitter, on the calling thread: what it returned,
    /// or its panic, where the submitter's panics unwind.
    pub fn run<R>(self, job: impl FnOnce() -> R) -> Result<R, Panic> {
        self.0.on(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(job))).map_err(Panic::caught)
    }
}

/// `work`, its panic caught: a boundary that declares its work's panic its
/// own failure, which leaves nothing the caller reads (a sweep's row
/// reported as failed, a probe process's failure). The one way a panic on a
/// thread whose panics end the process is caught; every thread that works
/// for `work` (a scoped thread, a pool's job) unwinds as it does.
pub fn recoverable<R>(work: impl FnOnce() -> R) -> Result<R, Panic> {
    Submitter(Panics::UNWINDS).run(work)
}

/// A panic caught where it unwound to the thread that answers for it
/// ([`Submitter::run`], [`recoverable`]): its cause, and its payload, which
/// unwinds again on the first of its observers to resume it.
#[derive(Clone)]
pub struct Panic {
    message: Arc<str>,
    payload: Arc<Mutex<Option<Box<dyn Any + Send>>>>,
}

impl Panic {
    fn caught(payload: Box<dyn Any + Send>) -> Self {
        let message = match payload.downcast_ref::<&str>() {
            Some(message) => Arc::from(*message),
            None => match payload.downcast_ref::<String>() {
                Some(message) => Arc::from(message.as_str()),
                None => Arc::from("a panic without a message"),
            },
        };
        Self { message, payload: Arc::new(Mutex::new(Some(payload))) }
    }

    /// What the panic said.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Go on with the panic on the calling thread, as a panic there would:
    /// unwind with its payload (or, where another observer resumed that, a
    /// panic naming its cause) where the thread's panics unwind, and end the
    /// process at it where they end it.
    pub fn resume(&self) -> ! {
        let payload = self.payload.lock().expect("nothing panics holding a caught panic").take();
        match payload {
            Some(payload) if !Panics::here().ends_process => std::panic::resume_unwind(payload),
            Some(_) => panic!("{}", self.message),
            None => panic!("{}; another observer resumed this panic", self.message),
        }
    }
}

impl std::fmt::Display for Panic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::fmt::Debug for Panic {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("Panic").field(&self.message).finish()
    }
}

/// `future` as a task nobody waits for as its own (one that owns part of
/// the process's state), named `name`, whatever runtime polls it: a panic
/// polling it ends the process at the panic, as one on a thread [`spawn`]
/// started, its report naming the task.
pub fn detached<F: Future>(name: &'static str, future: F) -> Detached<F> {
    Detached { name, future: Box::pin(future) }
}

/// A future [`detached`] returns.
pub struct Detached<F> {
    name: &'static str,
    future: Pin<Box<F>>,
}

thread_local! {
    /// The [`detached`] task the calling thread is polling, if any.
    static TASK: Cell<Option<&'static str>> = const { Cell::new(None) };
}

impl<F: Future> Future for Detached<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<F::Output> {
        let this = self.get_mut();
        // A panic polling it ends the process, so nothing unwinds past this.
        let outer = TASK.with(|task| task.replace(Some(this.name)));
        let polled = Panics::ENDS.on(|| this.future.as_mut().poll(context));
        TASK.with(|task| task.set(outer));
        polled
    }
}

/// Why the process ends, once a panic ends it: the panic's thread, place
/// and message, as the hook reported it.
static ENDING: OnceLock<String> = OnceLock::new();

/// Why the process is ending, where a panic ends it: what a report of its
/// end (a fault report written as it aborts) closes with, so a reader of
/// only its last lines learns the panic.
pub fn ending() -> Option<&'static str> {
    ENDING.get().map(String::as_str)
}

/// The process's one panic hook, once: after the hook before it (which
/// reports the panic), a panic whose thread's [`Panics`] ends the process
/// records why ([`ending`]) and aborts it there, before anything unwinds. A
/// thread's local being gone (a panic as it exits) unwinds as one on a
/// thread this crate did not start.
fn install() {
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        let reported = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            reported(info);
            if PANICS.try_with(Cell::get).is_ok_and(|panics| panics.ends_process) {
                let task = match TASK.try_with(Cell::get) {
                    Ok(Some(task)) => format!(" (task {task})"),
                    _ => String::new(),
                };
                let ending = format!(
                    "invariant violated: thread {}{task} {info}; the process ends rather than run on the state it left",
                    std::thread::current().name().unwrap_or("unnamed")
                );
                eprintln!("{ending}");
                let _ = ENDING.set(ending);
                std::process::abort();
            }
        }));
    });
}

/// A multi-threaded Tokio runtime whose workers and blocking threads have
/// [`STACK_BYTES`] of stack: every runtime of the process is built here, so
/// work that blocks on one runs on the owner's stack too. Its threads work
/// for the thread that builds it, which blocks on it: a panic on them ends
/// as one there would, so within [`main`] a task's panic ends the process
/// at the panic.
#[cfg(feature = "tokio")]
pub fn runtime() -> std::io::Result<tokio::runtime::Runtime> {
    install();
    let panics = Panics::here();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(STACK_BYTES)
        .on_thread_start(move || PANICS.with(|thread| thread.set(panics)))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Work that touches `bytes` of its stack: what an unoptimized build's
    /// deepest load path does in frames of its own.
    fn deep(bytes: usize) -> u8 {
        #[inline(never)]
        fn frame(left: usize) -> u8 {
            // Within the frame bound (`FRAME_BYTES`), as every frame is.
            let mut page = [0u8; 16 << 10];
            page[left % page.len()] = left as u8;
            let below = if left > page.len() { frame(left - page.len()) } else { 0 };
            std::hint::black_box(&page)[0].wrapping_add(below)
        }
        frame(bytes)
    }

    /// Work that overflows a library's 2 MiB thread runs to its end on the
    /// owner's, whoever asks: a thread of 1 MiB here.
    #[test]
    fn work_deeper_than_a_default_thread_runs_on_the_owners_stack() {
        let wanted = STACK_BYTES / 2;
        assert!(wanted > 2 << 20);
        std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(move || {
                run("deep", move || deep(wanted));
                spawn("deep", move || deep(wanted)).unwrap().join().unwrap();
                std::thread::scope(|scope| spawn_scoped(scope, "deep", move || deep(wanted)).unwrap().join().unwrap());
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// The variable a test's own process runs it in as the child it checks.
    const CHILD: &str = "SEISMIC_THREADS_PANIC_CHILD";

    /// `test` run alone in a child process of this test binary, and how
    /// that process ended.
    fn child(test: &str) -> std::process::Output {
        child_as(test, "1")
    }

    /// [`child`], running case `case` of `test`.
    fn child_as(test: &str, case: &str) -> std::process::Output {
        std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture", "--test-threads", "1"])
            .env(CHILD, case)
            .output()
            .unwrap()
    }

    /// A panic on a thread nobody waits for ends the process at the panic,
    /// reported, before anything on the thread unwinds: no destructor runs
    /// on the state the panic left.
    #[test]
    fn a_panic_where_nobody_waits_ends_the_process_before_it_unwinds() {
        const NAME: &str = "tests::a_panic_where_nobody_waits_ends_the_process_before_it_unwinds";
        if std::env::var_os(CHILD).is_some() {
            struct Unwound;
            impl Drop for Unwound {
                fn drop(&mut self) {
                    eprintln!("unwound past the panic");
                }
            }
            let thread = spawn("doomed", || {
                let _unwound = Unwound;
                panic!("an invariant violated")
            })
            .unwrap();
            let _ = thread.join();
            eprintln!("the process outlived the panic");
            return;
        }
        let ended = child(NAME);
        let stderr = String::from_utf8_lossy(&ended.stderr);
        assert!(!ended.status.success(), "{stderr}");
        #[cfg(unix)]
        assert_eq!(std::os::unix::process::ExitStatusExt::signal(&ended.status), Some(6), "aborted: {stderr}");
        assert!(stderr.contains("an invariant violated"), "the panic is reported: {stderr}");
        assert!(stderr.contains("invariant violated: thread doomed panicked at"), "{stderr}");
        assert!(stderr.contains("the process ends rather than run on the state it left"), "{stderr}");
        assert!(!stderr.contains("unwound past the panic"), "nothing unwinds: {stderr}");
        assert!(!stderr.contains("outlived"), "{stderr}");
    }

    /// A panic on a thread its spawner waits for ends as one on the
    /// spawner would: on a test's thread, it unwinds to the join, as one in
    /// [`run`] does to its caller.
    #[test]
    fn a_waited_threads_panic_ends_as_its_spawners() {
        let joined = std::thread::scope(|scope| spawn_scoped(scope, "waited", || panic!("waited")).unwrap().join());
        assert!(joined.is_err());
        assert!(std::panic::catch_unwind(|| run("waited", || panic!("run"))).is_err());
    }

    /// A boundary declared recoverable catches a panic on a thread whose
    /// panics end the process, with its cause, and every thread it waits
    /// for, or that works for it, unwinds as it does.
    #[test]
    fn a_recoverable_boundary_catches_where_the_process_would_end() {
        let caught = spawn("recovering", || {
            assert_eq!(Panics::here(), Panics::ENDS);
            let alone = recoverable(|| panic!("a candidate's defect: {}", 7)).unwrap_err().message().to_owned();
            let waited = recoverable(|| run("helper", || panic!("its helper's defect"))).unwrap_err().message().to_owned();
            let pooled = recoverable(|| {
                let submitter = Submitter::here();
                let job = spawn("pool", move || submitter.run(|| panic!("a pool's job")).unwrap_err()).unwrap().join().unwrap();
                job.resume()
            })
            .unwrap_err()
            .message()
            .to_owned();
            (alone, waited, pooled, Panics::here())
        })
        .unwrap()
        .join()
        .unwrap();
        assert_eq!(
            caught,
            ("a candidate's defect: 7".into(), "its helper's defect".into(), "a pool's job".into(), Panics::ENDS),
            "each was caught with its cause, and the thread's own panics end the process again"
        );
    }

    /// A program's panic ends the process at the panic, before anything on
    /// its thread unwinds, as do a task's of its runtime and a detached
    /// task's on a runtime of a thread whose panics unwind.
    #[test]
    fn a_programs_panic_ends_the_process_before_it_unwinds() {
        const NAME: &str = "tests::a_programs_panic_ends_the_process_before_it_unwinds";
        if let Some(case) = std::env::var_os(CHILD) {
            struct Unwound;
            impl Drop for Unwound {
                fn drop(&mut self) {
                    eprintln!("unwound past the panic");
                }
            }
            match case.to_str() {
                Some("program") => main("program", || {
                    let _unwound = Unwound;
                    panic!("an invariant violated")
                }),
                #[cfg(feature = "tokio")]
                Some("task") => main("program", || {
                    let _unwound = Unwound;
                    let _ = runtime().unwrap().block_on(async {
                        tokio::spawn(async {
                            let _unwound = Unwound;
                            panic!("an invariant violated")
                        })
                        .await
                    });
                }),
                #[cfg(feature = "tokio")]
                Some("detached") => {
                    let _ = runtime().unwrap().block_on(async {
                        tokio::spawn(detached("an owner", async {
                            let _unwound = Unwound;
                            panic!("an invariant violated")
                        }))
                        .await
                    });
                }
                _ => return,
            }
            eprintln!("the process outlived the panic");
            return;
        }
        let cases: &[&str] = if cfg!(feature = "tokio") { &["program", "task", "detached"] } else { &["program"] };
        for case in cases {
            let ended = child_as(NAME, case);
            let stderr = String::from_utf8_lossy(&ended.stderr);
            assert!(!ended.status.success(), "{case}: {stderr}");
            #[cfg(unix)]
            assert_eq!(std::os::unix::process::ExitStatusExt::signal(&ended.status), Some(6), "{case} aborted: {stderr}");
            assert!(stderr.contains("an invariant violated"), "{case}: the panic is reported: {stderr}");
            assert!(stderr.contains("the process ends rather than run on the state it left"), "{case}: {stderr}");
            if *case == "detached" {
                assert!(stderr.contains("(task an owner) panicked at"), "the report names the task: {stderr}");
            }
            assert!(!stderr.contains("unwound past the panic"), "{case}: nothing unwinds: {stderr}");
            assert!(!stderr.contains("outlived"), "{case}: {stderr}");
        }
    }

    /// No frame may take more of the stack than [`FRAME_BYTES`]: clippy's
    /// `large_stack_frames` holds every crate to the threshold clippy.toml
    /// sets, which is this.
    #[test]
    fn the_frame_lint_holds_frames_to_the_frame_bound() {
        let clippy = include_str!("../../../clippy.toml");
        assert!(
            clippy
                .lines()
                .any(|line| line.trim() == format!("stack-size-threshold = {FRAME_BYTES}")),
            "clippy.toml's stack-size-threshold is FRAME_BYTES ({FRAME_BYTES})"
        );
    }

    #[cfg(feature = "tokio")]
    #[test]
    fn a_runtimes_threads_are_the_owners() {
        let wanted = STACK_BYTES / 2;
        let runtime = runtime().unwrap();
        runtime.block_on(async move {
            tokio::spawn(async move { deep(wanted) }).await.unwrap();
            tokio::task::spawn_blocking(move || deep(wanted)).await.unwrap();
        });
    }
}
