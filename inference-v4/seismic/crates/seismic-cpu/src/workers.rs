//! The worker threads of one CPU device. A phase with several work items is published to
//! every worker; each claims items one at a time until none remain. The caller blocks until
//! all workers have left the phase, so the pointers it lends stay valid for its duration.
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// The compiled entry of a parallel phase: buffer table, scalar words, scratch, work item.
pub(crate) type PhaseEntry = unsafe extern "C" fn(*const *mut u8, *const u64, *mut u8, u64) -> i32;

#[derive(Clone)]
struct Job {
    entry: PhaseEntry,
    buffers: usize,
    scalars: usize,
    items: u64,
    scratch_bytes: usize,
    next: Arc<AtomicU64>,
    status: Arc<AtomicI32>,
}

#[derive(Default)]
struct State {
    generation: u64,
    job: Option<Job>,
    running: usize,
    shutdown: bool,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
}

impl Shared {
    fn lock(&self) -> Result<MutexGuard<'_, State>, String> {
        self.state.lock().map_err(|_| "a CPU worker panicked while holding the phase state".to_string())
    }
}

/// Status a worker reports when it cannot grow its scratch to a phase's size.
const SCRATCH_ALLOCATION_FAILED: i32 = -1;

pub struct Workers {
    shared: Arc<Shared>,
    threads: Vec<std::thread::JoinHandle<()>>,
    /// Scratch of the calling thread, for single-item phases.
    scratch: Vec<u64>,
}

fn grow(scratch: &mut Vec<u64>, bytes: usize) -> bool {
    let words = bytes.div_ceil(8);
    if scratch.len() >= words {
        return true;
    }
    if scratch.try_reserve_exact(words - scratch.len()).is_err() {
        return false;
    }
    scratch.resize(words, 0);
    true
}

fn work(shared: &Shared) {
    let mut scratch: Vec<u64> = Vec::new();
    let mut seen = 0u64;
    loop {
        let job = {
            let Ok(mut state) = shared.state.lock() else { return };
            loop {
                if state.shutdown {
                    return;
                }
                if state.generation != seen {
                    break;
                }
                state = match shared.wake.wait(state) {
                    Ok(state) => state,
                    Err(_) => return,
                };
            }
            seen = state.generation;
            state.job.clone()
        };
        if let Some(job) = job {
            if grow(&mut scratch, job.scratch_bytes) {
                loop {
                    let item = job.next.fetch_add(1, Ordering::Relaxed);
                    if item >= job.items {
                        break;
                    }
                    // The caller keeps the buffer table, the scalar words and every bound
                    // allocation alive and unaliased by the host until all workers report
                    // done. Each worker passes only its own scratch. Distinct pieces of a
                    // `parallel` domain write disjoint storage by the checked source effects.
                    let status = unsafe { (job.entry)(job.buffers as *const *mut u8, job.scalars as *const u64, scratch.as_mut_ptr().cast(), item) };
                    if status != 0 {
                        job.status.store(status, Ordering::Relaxed);
                    }
                }
            } else {
                job.status.store(SCRATCH_ALLOCATION_FAILED, Ordering::Relaxed);
            }
        }
        let Ok(mut state) = shared.state.lock() else { return };
        state.running -= 1;
        if state.running == 0 {
            shared.done.notify_all();
        }
    }
}

impl Workers {
    pub fn new(count: usize) -> Result<Self, String> {
        if count == 0 {
            return Err("a CPU device needs at least one worker".into());
        }
        let shared = Arc::new(Shared::default());
        let mut threads = Vec::with_capacity(count);
        for ordinal in 0..count {
            let shared = shared.clone();
            let thread = std::thread::Builder::new().name(format!("seismic-cpu-{ordinal}")).spawn(move || work(&shared)).map_err(|e| format!("CPU worker thread: {e}"))?;
            threads.push(thread);
        }
        Ok(Workers { shared, threads, scratch: Vec::new() })
    }

    /// One worker per unit of available parallelism.
    pub fn host() -> Result<Self, String> {
        Self::new(std::thread::available_parallelism().map_err(|e| format!("host parallelism: {e}"))?.get())
    }

    pub fn count(&self) -> usize {
        self.threads.len()
    }

    /// Execute `items` work items of one phase and return after all of them completed.
    /// A nonzero status is the first one a work item reported; remaining items still ran.
    pub(crate) fn run(&mut self, entry: PhaseEntry, buffers: &[*mut u8], scalars: &[u64], items: u64, scratch_bytes: usize) -> Result<i32, String> {
        if items == 0 {
            return Ok(0);
        }
        if items == 1 {
            if !grow(&mut self.scratch, scratch_bytes) {
                return Err(format!("CPU scratch allocation of {scratch_bytes} bytes failed"));
            }
            // Same contract as in `work`; the calling thread is the only participant.
            return Ok(unsafe { entry(buffers.as_ptr(), scalars.as_ptr(), self.scratch.as_mut_ptr().cast(), 0) });
        }
        let status = Arc::new(AtomicI32::new(0));
        let job = Job { entry, buffers: buffers.as_ptr() as usize, scalars: scalars.as_ptr() as usize, items, scratch_bytes, next: Arc::new(AtomicU64::new(0)), status: status.clone() };
        let mut state = self.shared.lock()?;
        state.generation += 1;
        state.job = Some(job);
        state.running = self.threads.len();
        self.shared.wake.notify_all();
        while state.running != 0 {
            state = self.shared.done.wait(state).map_err(|_| "a CPU worker panicked during a phase".to_string())?;
        }
        state.job = None;
        drop(state);
        match status.load(Ordering::Relaxed) {
            SCRATCH_ALLOCATION_FAILED => Err(format!("CPU scratch allocation of {scratch_bytes} bytes per worker failed")),
            status => Ok(status),
        }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        if let Ok(mut state) = self.shared.state.lock() {
            state.shutdown = true;
        }
        self.shared.wake.notify_all();
        for thread in self.threads.drain(..) {
            // A worker that panicked has already been reported by the phase it ran.
            let _ = thread.join();
        }
    }
}
