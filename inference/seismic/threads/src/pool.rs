//! Persistent work-stealing thread pool sized by [`crate::share`].
//!
//! Spawning scoped OS threads ([`crate::spawn_scoped`]) on every parallel
//! fan-out pays repeated thread-stack `mmap`/`munmap` and
//! `pthread_create`/`pthread_join` overhead. This module owns a single
//! lazily-initialized pool of persistent workers sized by [`crate::share`]
//! and started through [`crate::spawn`], supporting arbitrary nested [`join`],
//! [`par_for_results`], and [`par_for_results_with`] calls without deadlock
//! and preserving [`crate::Submitter`] panic semantics.

use std::cell::UnsafeCell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

#[derive(Clone, Copy)]
struct Job {
    run: unsafe fn(*const ()),
    data: *const (),
}

// SAFETY: Every `Job` pushed to the pool points to a `Sync` task state whose
// stack frame is guaranteed by a drop-safe completion barrier to outlive
// `Job::run`, and `run` only performs synchronized accesses through `data`.
unsafe impl Send for Job {}

impl Job {
    fn run(self) {
        // SAFETY: Guaranteed by the submitter's drop-safe completion barrier
        // before pushing this `Job` to the pool queue.
        unsafe { (self.run)(self.data) };
    }
}

struct QueueState {
    jobs: VecDeque<Job>,
    /// Number of threads currently parked in [`WorkPool::wait_until`] waiting
    /// for a task to finish or for newly pushed nested work to steal.
    waiters: usize,
}

struct Shared {
    state: Mutex<QueueState>,
    /// Wakes idle pool workers when jobs are pushed.
    worker_wake: Condvar,
    /// Wakes waiting `join` / `par_for` callers when jobs are pushed or a task
    /// completes.
    waiter_wake: Condvar,
}

struct WorkPool {
    shared: Arc<Shared>,
    workers: usize,
}

static POOL: OnceLock<WorkPool> = OnceLock::new();

impl WorkPool {
    fn global() -> &'static Self {
        POOL.get_or_init(Self::new)
    }

    fn new() -> Self {
        let workers = crate::share().get().saturating_sub(1);
        let shared = Arc::new(Shared {
            state: Mutex::new(QueueState {
                jobs: VecDeque::new(),
                waiters: 0,
            }),
            worker_wake: Condvar::new(),
            waiter_wake: Condvar::new(),
        });
        for _ in 0..workers {
            let shared = Arc::clone(&shared);
            crate::spawn("seismic-pool", move || worker_loop(&shared)).unwrap_or_else(|error| {
                panic!("invariant violated: the host starts a work pool thread: {error}")
            });
        }
        Self { shared, workers }
    }

    fn push(&self, job: Job) {
        let mut state = self
            .shared
            .state
            .lock()
            .expect("work pool queue mutex poisoned");
        state.jobs.push_back(job);
        if state.waiters > 0 {
            self.shared.waiter_wake.notify_one();
        }
        self.shared.worker_wake.notify_one();
    }

    fn push_many(&self, count: usize, job: Job) {
        let mut state = self
            .shared
            .state
            .lock()
            .expect("work pool queue mutex poisoned");
        state.jobs.reserve(count);
        for _ in 0..count {
            state.jobs.push_back(job);
        }
        if count == 1 {
            if state.waiters > 0 {
                self.shared.waiter_wake.notify_one();
            }
            self.shared.worker_wake.notify_one();
        } else {
            if state.waiters > 0 {
                self.shared.waiter_wake.notify_all();
            }
            self.shared.worker_wake.notify_all();
        }
    }

    fn try_remove(&self, data: *const ()) -> bool {
        let mut state = self
            .shared
            .state
            .lock()
            .expect("work pool queue mutex poisoned");
        if let Some(pos) = state.jobs.iter().rposition(|job| job.data == data) {
            state.jobs.remove(pos);
            true
        } else {
            false
        }
    }

    fn remove_all(&self, data: *const ()) -> usize {
        let mut state = self
            .shared
            .state
            .lock()
            .expect("work pool queue mutex poisoned");
        let before = state.jobs.len();
        state.jobs.retain(|job| job.data != data);
        before - state.jobs.len()
    }

    /// Run `mark_done` while holding the queue mutex and wake parked waiters if
    /// it reports task completion. Updating the completion flag under the same
    /// mutex that [`Self::wait_until`] checks before parking eliminates lost
    /// wakeups.
    fn finish_and_notify(&self, mark_done: impl FnOnce() -> bool) {
        let state = self
            .shared
            .state
            .lock()
            .expect("work pool queue mutex poisoned");
        if mark_done() && state.waiters > 0 {
            self.shared.waiter_wake.notify_all();
        }
    }

    /// Help run queued work until `is_done()` returns true, parking on
    /// `waiter_wake` whenever the queue is empty and waking immediately on
    /// either task completion or newly pushed nested work.
    fn wait_until(&self, is_done: impl Fn() -> bool) {
        loop {
            if is_done() {
                return;
            }
            let job = {
                let mut state = self
                    .shared
                    .state
                    .lock()
                    .expect("work pool queue mutex poisoned");
                loop {
                    if is_done() {
                        return;
                    }
                    if let Some(job) = state.jobs.pop_back() {
                        break job;
                    }
                    state.waiters += 1;
                    state = self
                        .shared
                        .waiter_wake
                        .wait(state)
                        .expect("work pool queue mutex poisoned");
                    state.waiters -= 1;
                }
            };
            job.run();
        }
    }
}

fn worker_loop(shared: &Shared) {
    loop {
        let job = {
            let mut state = shared.state.lock().expect("work pool queue mutex poisoned");
            loop {
                if let Some(job) = state.jobs.pop_front() {
                    break job;
                }
                state = shared
                    .worker_wake
                    .wait(state)
                    .expect("work pool queue mutex poisoned");
            }
        };
        job.run();
    }
}

struct JoinTask<F, A> {
    submitter: crate::Submitter,
    /// Closure `a`, taken once by either the single worker that pops this job
    /// or the calling thread if it removes the job from the queue first.
    func: UnsafeCell<Option<F>>,
    /// Written once by the worker before release-storing `done = true`; read
    /// once by the calling thread after acquire-loading `done == true`.
    result: UnsafeCell<Option<Result<A, crate::Panic>>>,
    done: AtomicBool,
}

// SAFETY: `func` is accessed exclusively by whichever thread owns the job
// (either the single worker that popped it from the queue, or the calling
// thread after removing it from the queue). `result` is written once by that
// worker before release-storing `done = true`, and only read by the calling
// thread after acquire-loading `done == true`.
unsafe impl<F: Send, A: Send> Sync for JoinTask<F, A> {}

unsafe fn run_join_task<F, A>(data: *const ())
where
    F: FnOnce() -> A + Send,
    A: Send,
{
    // SAFETY: `JoinGuard` ensures `data` points to a live `JoinTask<F, A>`
    // until `task.done` is release-stored `true`.
    let task = unsafe { &*(data as *const JoinTask<F, A>) };
    // SAFETY: This job was popped from the queue by this thread alone, so no
    // other thread accesses `task.func`.
    let func = unsafe { (*task.func.get()).take() }
        .expect("invariant violated: join task executed twice");
    let outcome = task.submitter.run(func);
    // SAFETY: Only this thread writes `task.result`, and the caller does not
    // read it until observing `task.done == true` with `Ordering::Acquire`.
    unsafe { *task.result.get() = Some(outcome) };
    // `task` must not be touched after `done.store(true, Ordering::Release)`.
    WorkPool::global().finish_and_notify(|| {
        task.done.store(true, Ordering::Release);
        true
    });
}

struct JoinGuard<'a, F: Send, A: Send> {
    pool: &'static WorkPool,
    task: &'a JoinTask<F, A>,
    task_ptr: *const (),
    active: bool,
}

impl<F: Send, A: Send> JoinGuard<'_, F, A> {
    fn wait_remote(&self) {
        self.pool
            .wait_until(|| self.task.done.load(Ordering::Acquire));
    }
}

impl<F: FnOnce() -> A + Send, A: Send> JoinGuard<'_, F, A> {
    fn complete(mut self) -> A {
        self.active = false;
        if self.pool.try_remove(self.task_ptr) {
            // SAFETY: The job was removed from the queue before any worker
            // popped it, so the calling thread has exclusive access to `func`.
            let func = unsafe { (*self.task.func.get()).take() }
                .expect("invariant violated: join task closure already taken");
            return func();
        }
        self.wait_remote();
        // SAFETY: `wait_remote` observed `task.done == true` with
        // `Ordering::Acquire`, synchronizing-with the worker's write to
        // `task.result` and guaranteeing the worker will never touch `task`
        // again.
        let outcome = unsafe { (*self.task.result.get()).take() }
            .expect("invariant violated: completed join task produced no result");
        match outcome {
            Ok(value) => value,
            Err(panic) => panic.resume(),
        }
    }
}

impl<F: Send, A: Send> Drop for JoinGuard<'_, F, A> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // `b()` panicked: ensure no worker can observe `self.task` after this
        // stack frame unwinds.
        if self.pool.try_remove(self.task_ptr) {
            return;
        }
        self.wait_remote();
    }
}

/// Run `a` and `b` potentially in parallel on the shared work pool, returning
/// `(a(), b())`.
///
/// Safe under arbitrary nesting (the waiting caller helps run queued work) and
/// preserves [`crate::Submitter`] panic semantics.
pub fn join<A: Send, B, FA, FB>(a: FA, b: FB) -> (A, B)
where
    FA: FnOnce() -> A + Send,
    FB: FnOnce() -> B,
{
    let pool = WorkPool::global();
    if pool.workers == 0 {
        return (a(), b());
    }
    let task = JoinTask {
        submitter: crate::Submitter::here(),
        func: UnsafeCell::new(Some(a)),
        result: UnsafeCell::new(None),
        done: AtomicBool::new(false),
    };
    let task_ptr = (&task as *const JoinTask<FA, A>).cast::<()>();
    pool.push(Job {
        run: run_join_task::<FA, A>,
        data: task_ptr,
    });
    let guard = JoinGuard {
        pool,
        task: &task,
        task_ptr,
        active: true,
    };
    let b_val = b();
    let a_val = guard.complete();
    (a_val, b_val)
}

struct ParForState<'a, T, Init, F, R, E> {
    submitter: crate::Submitter,
    items: &'a [T],
    init: &'a Init,
    work: &'a F,
    next: AtomicUsize,
    /// Pre-allocated output slots indexed by input position `0..items.len()`.
    /// Each index is claimed at most once via `next.fetch_add(1)`.
    slots: Vec<UnsafeCell<Option<Result<R, E>>>>,
    panic: Mutex<Option<crate::Panic>>,
    remaining: AtomicUsize,
}

// SAFETY: `items`, `init`, and `work` are `Sync`; each `slots[idx]` is written
// at most once by the unique thread that claims `idx` from `next.fetch_add(1)`;
// the calling thread only reads `slots` after observing `remaining == 0` with
// `Ordering::Acquire`, which synchronizes-with every helper thread's release
// decrement.
unsafe impl<T: Sync, Init: Sync, F: Sync, R: Send, E: Send> Sync
    for ParForState<'_, T, Init, F, R, E>
{
}

unsafe fn run_par_for_helper<T, Init, S, F, R, E>(data: *const ())
where
    T: Sync,
    Init: Fn() -> S + Sync,
    F: Fn(&T) -> Result<R, E> + Sync,
    R: Send,
    E: Send,
{
    // SAFETY: `ParForGuard` guarantees `data` points to a live `ParForState`
    // until `state.remaining` reaches 0.
    let state = unsafe { &*(data as *const ParForState<'_, T, Init, F, R, E>) };
    let outcome = state.submitter.run(|| {
        let _scope = (state.init)();
        let n = state.items.len();
        loop {
            let idx = state.next.fetch_add(1, Ordering::Relaxed);
            if idx >= n {
                break;
            }
            let res = (state.work)(&state.items[idx]);
            // SAFETY: `idx` was uniquely claimed by `state.next.fetch_add(1)`
            // and is strictly less than `n == state.slots.len()`.
            unsafe { *state.slots[idx].get() = Some(res) };
        }
    });
    if let Err(panic) = outcome {
        state
            .next
            .fetch_max(state.items.len(), Ordering::Relaxed);
        let mut slot = state.panic.lock().expect("par_for panic mutex poisoned");
        if slot.is_none() {
            *slot = Some(panic);
        }
    }
    // `state` must not be touched after `remaining.fetch_sub(1, Ordering::Release)`.
    WorkPool::global().finish_and_notify(|| {
        state.remaining.fetch_sub(1, Ordering::Release) == 1
    });
}

struct ParForGuard<'s, 'a, T: Sync, Init: Sync, F: Sync, R: Send, E: Send> {
    pool: &'static WorkPool,
    state: &'s ParForState<'a, T, Init, F, R, E>,
    state_ptr: *const (),
    active: bool,
}

impl<T: Sync, Init: Sync, F: Sync, R: Send, E: Send> ParForGuard<'_, '_, T, Init, F, R, E> {
    fn wait_all(&mut self) {
        let removed = self.pool.remove_all(self.state_ptr);
        if removed > 0
            && self.state.remaining.fetch_sub(removed, Ordering::AcqRel) == removed
        {
            return;
        }
        self.pool
            .wait_until(|| self.state.remaining.load(Ordering::Acquire) == 0);
    }

    fn finish(mut self) {
        self.active = false;
        self.wait_all();
    }
}

impl<T: Sync, Init: Sync, F: Sync, R: Send, E: Send> Drop
    for ParForGuard<'_, '_, T, Init, F, R, E>
{
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // Calling thread panicked: stop helpers from claiming further items,
        // remove any still-queued helper jobs, and wait for in-flight helpers
        // before `state` drops.
        self.state
            .next
            .fetch_max(self.state.items.len(), Ordering::Relaxed);
        self.wait_all();
    }
}

/// Evaluate `work` on every item in `items` across the shared work pool using
/// dynamic atomic work stealing, calling `init` once on each participating
/// thread before its steal loop (and holding the returned guard until that
/// thread's steal loop exits), and returning results (or the first error in
/// input order) in exact input order.
pub fn par_for_results_with<T, R, E, S, Init, F>(
    items: &[T],
    init: Init,
    work: F,
) -> Result<Vec<R>, E>
where
    T: Sync,
    R: Send,
    E: Send,
    Init: Fn() -> S + Sync,
    F: Fn(&T) -> Result<R, E> + Sync,
{
    let n = items.len();
    let pool = WorkPool::global();
    if n <= 1 || pool.workers == 0 {
        let _scope = init();
        return items.iter().map(work).collect();
    }
    let helpers = (n - 1).min(pool.workers);
    let state = ParForState {
        submitter: crate::Submitter::here(),
        items,
        init: &init,
        work: &work,
        next: AtomicUsize::new(0),
        slots: (0..n).map(|_| UnsafeCell::new(None)).collect(),
        panic: Mutex::new(None),
        remaining: AtomicUsize::new(helpers),
    };
    let state_ptr = (&state as *const ParForState<'_, T, Init, F, R, E>).cast::<()>();
    pool.push_many(
        helpers,
        Job {
            run: run_par_for_helper::<T, Init, S, F, R, E>,
            data: state_ptr,
        },
    );
    let guard = ParForGuard {
        pool,
        state: &state,
        state_ptr,
        active: true,
    };
    {
        let _scope = init();
        loop {
            let idx = state.next.fetch_add(1, Ordering::Relaxed);
            if idx >= n {
                break;
            }
            let res = work(&items[idx]);
            // SAFETY: `idx` was uniquely claimed by `state.next.fetch_add(1)`
            // and is strictly less than `n == state.slots.len()`.
            unsafe { *state.slots[idx].get() = Some(res) };
        }
    }
    guard.finish();
    if let Some(panic) = state
        .panic
        .into_inner()
        .expect("par_for panic mutex poisoned")
    {
        panic.resume();
    }
    state
        .slots
        .into_iter()
        .map(|cell| {
            cell.into_inner()
                .expect("every input index produced a result")
        })
        .collect()
}

/// Evaluate `work` on every item in `items` across the shared work pool using
/// dynamic atomic work stealing, returning results (or the first error in
/// input order) in exact input order.
pub fn par_for_results<T: Sync, R: Send, E: Send, F>(
    items: &[T],
    work: F,
) -> Result<Vec<R>, E>
where
    F: Fn(&T) -> Result<R, E> + Sync,
{
    par_for_results_with(items, || (), work)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_runs_both_and_handles_deep_nesting() {
        fn fib(n: u32) -> u32 {
            if n <= 1 {
                return n;
            }
            let (a, b) = join(|| fib(n - 1), || fib(n - 2));
            a + b
        }
        assert_eq!(fib(12), 144);
    }

    #[test]
    fn join_propagates_panics_from_either_branch() {
        let from_a = std::panic::catch_unwind(|| join(|| -> i32 { panic!("boom a") }, || 2));
        assert!(from_a.is_err());

        let from_b = std::panic::catch_unwind(|| {
            join(
                || {
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    1
                },
                || -> i32 { panic!("boom b") },
            )
        });
        assert!(from_b.is_err());
    }

    #[test]
    fn par_for_results_preserves_order_errors_and_nesting() {
        let empty: Vec<i32> = par_for_results::<i32, i32, &str, _>(&[], |&x| Ok(x)).unwrap();
        assert!(empty.is_empty());

        let items: Vec<usize> = (0..64).collect();
        let nested = par_for_results(&items, |&i| -> Result<usize, &str> {
            let inner: Vec<usize> = (0..4).map(|j| i * 4 + j).collect();
            let vals = par_for_results(&inner, |&v| -> Result<usize, &str> { Ok(v + 1) })?;
            Ok(vals.into_iter().sum())
        })
        .unwrap();
        for (i, sum) in nested.into_iter().enumerate() {
            let expected: usize = (0..4).map(|j| i * 4 + j + 1).sum();
            assert_eq!(sum, expected);
        }

        let err = par_for_results(&items, |&i| {
            if i == 17 || i == 42 {
                Err(i)
            } else {
                Ok(i)
            }
        });
        assert_eq!(err, Err(17));
    }

    #[test]
    fn par_for_results_with_scopes_each_participating_thread() {
        use std::sync::atomic::AtomicUsize;

        struct Guard<'a>(&'a AtomicUsize);
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let entered = AtomicUsize::new(0);
        let exited = AtomicUsize::new(0);
        let items: Vec<usize> = (0..32).collect();
        let out = par_for_results_with(
            &items,
            || {
                entered.fetch_add(1, Ordering::Relaxed);
                Guard(&exited)
            },
            |&i| -> Result<usize, ()> { Ok(i * 2) },
        )
        .unwrap();
        assert_eq!(out, (0..32).map(|i| i * 2).collect::<Vec<_>>());
        assert!(entered.load(Ordering::Relaxed) >= 1);
        assert_eq!(
            entered.load(Ordering::Relaxed),
            exited.load(Ordering::Relaxed)
        );
    }

    #[test]
    fn par_for_results_propagates_panics() {
        let items: Vec<usize> = (0..32).collect();
        let caught = std::panic::catch_unwind(|| {
            let _ = par_for_results(&items, |&i| -> Result<usize, ()> {
                if i == 15 {
                    panic!("par_for boom");
                }
                Ok(i)
            });
        });
        assert!(caught.is_err());
    }
}
