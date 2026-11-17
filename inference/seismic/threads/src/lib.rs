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

use std::num::NonZeroUsize;
use std::sync::OnceLock;

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

/// Start a thread named `name`, with [`STACK_BYTES`] of stack, running
/// `work`: every thread of the process starts here.
pub fn spawn<T: Send + 'static>(
    name: impl Into<String>,
    work: impl FnOnce() -> T + Send + 'static,
) -> std::io::Result<std::thread::JoinHandle<T>> {
    builder(name).spawn(work)
}

/// [`spawn`] in `scope`.
pub fn spawn_scoped<'scope, 'env, T: Send + 'scope>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    name: impl Into<String>,
    work: impl FnOnce() -> T + Send + 'scope,
) -> std::io::Result<std::thread::ScopedJoinHandle<'scope, T>> {
    builder(name).spawn_scoped(scope, work)
}

/// `work` on a thread of its own ([`spawn_scoped`]) while the caller waits:
/// work whose caller's stack is not the process's to size (a test harness's
/// thread, an embedder's) runs on one that is. A thread is started for each
/// call, which costs microseconds beside the work run this way. A panic in
/// `work` continues in the caller.
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

/// A multi-threaded Tokio runtime whose workers and blocking threads have
/// [`STACK_BYTES`] of stack: every runtime of the process is built here, so
/// work that blocks on one runs on the owner's stack too.
#[cfg(feature = "tokio")]
pub fn runtime() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread().enable_all().thread_stack_size(STACK_BYTES).build()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Work that touches `bytes` of its stack: what an unoptimized build's
    /// deepest load path does in frames of its own.
    fn deep(bytes: usize) -> u8 {
        #[inline(never)]
        fn frame(left: usize) -> u8 {
            let mut page = [0u8; 64 << 10];
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
