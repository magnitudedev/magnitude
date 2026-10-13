//! A bounded set of reused worker threads.
//!
//! Work that runs off its caller's thread (source rendering, compile calls,
//! module loads) allocates and frees much heap, and an allocator keeps what
//! a thread frees for that thread: glibc in an arena of the thread's own.
//! A thread for each unit of work left freed heap resident in as many
//! arenas as threads had ever lived at once (a GB10 host, 2026-10-10,
//! Qwen3.5-4B first load: 0.36 GiB in 112 arenas where a load that tunes
//! nothing has 10). So the work of one owner runs on one set of threads,
//! at most [`Workers::width`] of them, each reused for as long as the set
//! lives, and a thread returns the pages its work freed whenever it has
//! run the last job waiting ([`crate::process::release_freed_heap`]): what
//! the set's threads hold between bursts is what is in use, not each
//! thread's high-water mark.
//!
//! Work may itself hand out work ([`Workers::each`]): the thread that hands
//! it out takes part, so nested work never waits for a thread the set does
//! not have, and the threads at work are never more than the set's width
//! and the callers'.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct Workers {
    shared: Arc<Shared>,
    width: usize,
}

/// A set whose batches are held to fewer threads while this lives
/// ([`Workers::narrowed`]).
pub struct Narrowed {
    workers: Arc<Workers>,
    before: usize,
}

impl Drop for Narrowed {
    fn drop(&mut self) {
        self.workers.state().allowed = self.before;
        // Jobs that waited for a thread they were not allowed may run.
        self.workers.shared.wake.notify_all();
    }
}

struct Shared {
    state: Mutex<State>,
    /// A job waits, or the set is closing.
    wake: Condvar,
}

#[derive(Default)]
struct State {
    jobs: VecDeque<Job>,
    idle: usize,
    closing: bool,
    threads: Vec<std::thread::JoinHandle<()>>,
    /// The most threads at work at once, the caller that hands the work
    /// out among them ([`Workers::narrowed`]): the set's width and that
    /// caller unless its owner allows fewer.
    allowed: usize,
    /// The set's threads running a job now.
    running: usize,
}

impl Workers {
    /// A set of at most `width` threads, each started when work first
    /// waits for it.
    pub fn new(width: usize) -> Self {
        Self {
            shared: Arc::new(Shared {
                state: Mutex::default(),
                wake: Condvar::new(),
            }),
            width: width.max(1),
        }
        .whole()
    }

    /// The set with every thread allowed: its width and a caller.
    fn whole(self) -> Self {
        self.state().allowed = self.width + 1;
        self
    }

    /// Hold the set's work to `threads` threads at once, the caller that
    /// hands it out among them and never fewer than that caller, until
    /// what is returned is dropped: of everything handed out, batches
    /// within batches included, the set's own threads run at most one
    /// fewer than `threads` at a time. Work that runs at once holds its
    /// memory at once: the owner of a span of work that knows what the
    /// process may still take allows as many threads as that affords,
    /// instead of as many as the host has.
    pub fn narrowed(self: &Arc<Self>, threads: usize) -> Narrowed {
        let allowed = threads.clamp(1, self.width + 1);
        Narrowed {
            workers: self.clone(),
            before: std::mem::replace(&mut self.state().allowed, allowed),
        }
    }

    /// As many threads as the host runs in parallel: more at work than
    /// that only take longer each.
    pub fn of_the_host() -> Self {
        Self::new(
            std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(1),
        )
    }

    /// The most threads the set runs.
    pub fn width(&self) -> usize {
        self.width
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.shared
            .state
            .lock()
            .expect("no worker panics holding the set")
    }

    /// Run `job` on a thread of the set, when one is free. A job that
    /// panics is caught where it reports ([`Workers::each`] resumes it in
    /// its caller); the thread goes on.
    pub fn spawn(&self, job: impl FnOnce() + Send + 'static) {
        let mut state = self.state();
        state.jobs.push_back(Box::new(job));
        if state.jobs.len() > state.idle && state.threads.len() < self.width {
            let shared = self.shared.clone();
            state.threads.push(
                std::thread::Builder::new()
                    .name("seismic-worker".into())
                    .spawn(move || shared.work())
                    .expect("the host starts a worker thread"),
            );
        }
        drop(state);
        self.shared.wake.notify_one();
    }

    /// `form` each of `work`, as many at once as the set has threads free
    /// and the calling thread besides; what each formed, in order. The
    /// caller takes part, so this returns whether or not a thread of the
    /// set is ever free, and may be called from one.
    pub fn each<W: Sync, R: Send, F: Fn(&W) -> R + Sync>(&self, work: &[W], form: F) -> Vec<R> {
        if work.len() <= 1 {
            return work.iter().map(form).collect();
        }
        let batch = Batch {
            work,
            form: &form,
            next: AtomicUsize::new(0),
            formed: Mutex::new((0..work.len()).map(|_| None).collect()),
        };
        let gate = Arc::new(Gate::default());
        // The helpers borrow the batch on this stack. The gate is how that
        // is sound: a helper touches the batch only while it is counted in,
        // and this call does not return, by its end or by a panic, before
        // the gate is shut and every helper counted in is out.
        let shut = Shut(&gate);
        let erased = Erased(&batch as *const Batch<'_, W, R, F> as *const ());
        // A function pointer names the batch's types without their
        // lifetimes, so a job of the set can carry it.
        let run: unsafe fn(*const ()) = run::<W, R, F>;
        // The caller is one of the threads the set is allowed.
        let helpers = self.state().allowed - 1;
        for _ in 0..(work.len() - 1).min(helpers) {
            let gate = gate.clone();
            self.spawn(move || {
                let erased = erased;
                if !gate.enter() {
                    return;
                }
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // SAFETY: counted in through the gate, so the caller
                    // is still inside `each` and the batch, of the types
                    // `run` was named with, is alive.
                    unsafe { run(erased.0) };
                }));
                gate.leave(outcome.err());
            });
        }
        batch.run();
        drop(shut);
        if let Some(panic) = gate.state.lock().expect("gate lock").panic.take() {
            std::panic::resume_unwind(panic);
        }
        batch
            .formed
            .into_inner()
            .expect("no helper panics holding the results")
            .into_iter()
            .map(|formed| formed.expect("every item of a batch is formed before it ends"))
            .collect()
    }
}

/// The items of one [`Workers::each`], claimed one at a time by the caller
/// and its helpers.
struct Batch<'a, W, R, F> {
    work: &'a [W],
    form: &'a F,
    next: AtomicUsize,
    formed: Mutex<Vec<Option<R>>>,
}

impl<W: Sync, R: Send, F: Fn(&W) -> R + Sync> Batch<'_, W, R, F> {
    fn run(&self) {
        loop {
            let index = self.next.fetch_add(1, Ordering::Relaxed);
            let Some(item) = self.work.get(index) else {
                return;
            };
            let formed = (self.form)(item);
            self.formed.lock().expect("no helper panics holding the results")[index] =
                Some(formed);
        }
    }
}

/// A batch behind a pointer without its lifetime, for a helper job.
#[derive(Clone, Copy)]
struct Erased(*const ());

// SAFETY: the batch it points to is `Sync` (its work and its function are,
// and its results are behind a lock), and is reached only through the gate.
unsafe impl Send for Erased {}

/// Run the batch `erased` points to.
///
/// # Safety
/// `erased` points to a live `Batch<W, R, F>` of exactly these types.
unsafe fn run<W: Sync, R: Send, F: Fn(&W) -> R + Sync>(erased: *const ()) {
    // SAFETY: the caller's contract.
    unsafe { &*(erased as *const Batch<'_, W, R, F>) }.run();
}

/// Whether the helpers of one batch may still reach it, and how many are
/// in it now.
#[derive(Default)]
struct Gate {
    state: Mutex<GateState>,
    left: Condvar,
}

#[derive(Default)]
struct GateState {
    shut: bool,
    inside: usize,
    panic: Option<Box<dyn std::any::Any + Send>>,
}

impl Gate {
    fn enter(&self) -> bool {
        let mut state = self.state.lock().expect("gate lock");
        if state.shut {
            return false;
        }
        state.inside += 1;
        true
    }

    fn leave(&self, panic: Option<Box<dyn std::any::Any + Send>>) {
        let mut state = self.state.lock().expect("gate lock");
        state.inside -= 1;
        if state.panic.is_none() {
            state.panic = panic;
        }
        drop(state);
        self.left.notify_all();
    }
}

/// Shuts a gate and waits for every helper inside it, when dropped.
struct Shut<'a>(&'a Gate);

impl Drop for Shut<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().expect("gate lock");
        state.shut = true;
        while state.inside > 0 {
            state = self.0.left.wait(state).expect("gate lock");
        }
    }
}

impl Shared {
    /// One thread of the set: take each waiting job in turn until the set
    /// closes.
    fn work(&self) {
        loop {
            let job = {
                let mut state = self.state.lock().expect("no worker panics holding the set");
                loop {
                    // A job waits while the set's threads at work are as
                    // many as it is allowed beside the caller.
                    if state.running + 1 < state.allowed || state.closing {
                        if let Some(job) = state.jobs.pop_front() {
                            state.running += 1;
                            break job;
                        }
                    }
                    if state.closing {
                        return;
                    }
                    state.idle += 1;
                    state = self
                        .wake
                        .wait(state)
                        .expect("no worker panics holding the set");
                    state.idle -= 1;
                }
            };
            // A job reports its own panic to whoever waits for it.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
            // With no job waiting, this thread's burst of work is over. A
            // compile frees nearly all it took but leaves what it formed
            // at the far end of its thread's arena, so the allocator cannot
            // shrink the arena itself and the freed pages stay resident:
            // each thread's arena then sits at its own high-water mark, and
            // the process holds their sum, not what is in use at one time.
            // The thread that ran the burst is the one that knows it
            // ended, so it returns the pages here.
            let waiting = {
                let mut state = self.state.lock().expect("no worker panics holding the set");
                state.running -= 1;
                state.jobs.len()
            };
            // A job that waited for this thread's place may run.
            self.wake.notify_one();
            // What the job left resident is read before any of it is given
            // back: where the footprint's peak is the most of its readings,
            // these are the readings that see what jobs running at once
            // hold together.
            if crate::process::ProcessMemory::PEAK_IS_SAMPLED {
                crate::process::ProcessMemory::footprint_now();
            }
            if waiting == 0 {
                crate::process::release_freed_heap();
            }
        }
    }
}

impl Drop for Workers {
    /// The owner is done: every job handed out runs to its end, and the
    /// threads are joined.
    fn drop(&mut self) {
        let threads = {
            let mut state = self.state();
            state.closing = true;
            std::mem::take(&mut state.threads)
        };
        self.shared.wake.notify_all();
        for thread in threads {
            thread.join().expect("a worker thread catches its jobs' panics");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Whatever is handed out, no more threads than the set's width ever
    /// run it, each reused, and every job runs before the set is gone.
    #[test]
    fn a_set_runs_every_job_on_at_most_its_width_of_reused_threads() {
        let threads = Arc::new(Mutex::new(HashSet::new()));
        let ran = Arc::new(AtomicUsize::new(0));
        let workers = Workers::new(3);
        for _ in 0..200 {
            let (threads, ran) = (threads.clone(), ran.clone());
            workers.spawn(move || {
                threads.lock().unwrap().insert(std::thread::current().id());
                ran.fetch_add(1, Ordering::SeqCst);
            });
        }
        drop(workers);
        assert_eq!(ran.load(Ordering::SeqCst), 200);
        assert!(threads.lock().unwrap().len() <= 3);
    }

    /// Work handed out from inside work ends, in order, on no more threads
    /// than the set's and the caller's, even when every thread of the set
    /// is itself handing out work.
    #[test]
    fn nested_work_ends_without_a_thread_of_its_own() {
        let workers = Workers::new(2);
        let threads = Mutex::new(HashSet::new());
        let outer = (0..8usize).collect::<Vec<_>>();
        let formed = workers.each(&outer, |outer| {
            let inner = (0..5usize).collect::<Vec<_>>();
            workers
                .each(&inner, |inner| {
                    threads.lock().unwrap().insert(std::thread::current().id());
                    outer * 10 + inner
                })
                .into_iter()
                .sum::<usize>()
        });
        assert_eq!(
            formed,
            outer.iter().map(|outer| outer * 50 + 10).collect::<Vec<_>>()
        );
        assert!(threads.lock().unwrap().len() <= 3);
    }

    /// A narrowed set works a batch on no more threads than it was allowed,
    /// the caller among them, and on its whole width again afterwards.
    #[test]
    fn a_narrowed_set_works_a_batch_on_the_threads_allowed() {
        let workers = Arc::new(Workers::new(6));
        let work = (0..48usize).collect::<Vec<_>>();
        let distinct = |workers: &Workers| {
            let threads = Mutex::new(HashSet::new());
            workers.each(&work, |_| {
                threads.lock().unwrap().insert(std::thread::current().id());
                std::thread::sleep(std::time::Duration::from_millis(2));
            });
            threads.into_inner().unwrap()
        };
        {
            let _narrowed = workers.narrowed(2);
            assert!(distinct(&workers).len() <= 2);
            // Batches handed out from inside a batch share the allowance:
            // it is the set's, not each batch's.
            let nested = Mutex::new(HashSet::new());
            workers.each(&work[..6], |_| {
                workers.each(&work[..6], |_| {
                    nested.lock().unwrap().insert(std::thread::current().id());
                    std::thread::sleep(std::time::Duration::from_millis(2));
                });
            });
            assert!(nested.into_inner().unwrap().len() <= 2);
            let _alone = workers.narrowed(0);
            assert_eq!(
                distinct(&workers),
                HashSet::from([std::thread::current().id()]),
                "a batch always has its caller"
            );
        }
        assert!(distinct(&workers).len() > 2, "the set is whole again");
    }

    /// What a job allocated and freed on a worker thread is the system's
    /// again once that thread has no job waiting: the set holds no
    /// thread's high-water mark between bursts.
    #[test]
    #[cfg(any(target_os = "macos", all(target_os = "linux", target_env = "gnu")))]
    fn a_worker_returns_what_its_burst_freed() {
        use crate::process::ProcessMemory;
        const BLOCK: usize = 16 << 10;
        const BYTES: usize = 96 << 20;
        let _measuring = crate::process::MEASURING
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = ProcessMemory::read().footprint_bytes.unwrap();
        let held = Arc::new(Mutex::new(0u64));
        let workers = Workers::new(2);
        for _ in 0..2 {
            let held = held.clone();
            workers.spawn(move || {
                // Small blocks, with one that outlives the burst allocated
                // last: freed, the rest stay in the thread's arena.
                let blocks = (0..BYTES / BLOCK)
                    .map(|block| vec![(block % 255) as u8 + 1; BLOCK])
                    .collect::<Vec<_>>();
                let kept = vec![1u8; BLOCK];
                let mut held = held.lock().unwrap();
                *held = (*held).max(ProcessMemory::read().footprint_bytes.unwrap());
                assert!(std::hint::black_box(&blocks).iter().all(|block| block[0] > 0));
                drop(blocks);
                std::mem::forget(kept);
            });
        }
        // Joined: each thread has run its last job and what follows it.
        drop(workers);
        let held = *held.lock().unwrap();
        assert!(held >= before + BYTES as u64 / 2, "the burst held its blocks");
        let after = ProcessMemory::read().footprint_bytes.unwrap();
        let left = after as i64 - before as i64;
        assert!(left < BYTES as i64 / 4, "{left} bytes of freed heap stay resident");
    }

    /// A panic in handed-out work is its caller's panic, once every helper
    /// has left the caller's stack.
    #[test]
    fn a_panic_in_handed_out_work_is_the_callers() {
        let workers = Workers::new(4);
        let work = (0..64usize).collect::<Vec<_>>();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            workers.each(&work, |item| {
                assert!(*item != 40, "item forty fails");
                *item
            })
        }));
        assert!(outcome.is_err());
        // The set works on.
        assert_eq!(workers.each(&work, |item| item + 1)[63], 64);
    }
}
