//! The worker pool of one CPU device: teams, work-item claiming, and the
//! team barrier a workgroup barrier compiles to.
//!
//! A launch publishes one job to every worker. Workers are grouped into
//! teams of `team_size` (the launch's workgroup thread count, bounded by
//! the worker count through the profile); each team claims workgroups from
//! the job's counter until none remain, its members running one thread of
//! the workgroup each. The submitting thread blocks until every worker has
//! left the job, so the frame, buffer table, words, and scratch it lends
//! stay valid for the job's duration.

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::buffer::{AllocationFailure, Buffer};
use crate::profile::SCRATCH_ALIGNMENT;

/// Restores the caller's floating environment after forcing the reference
/// recipe's primitive contract: round-to-nearest-even, gradual underflow and
/// non-canonicalized NaNs. Every worker enters this guard before native code.
struct StrictFloatEnvironment {
    saved: u64,
}

impl StrictFloatEnvironment {
    #[cfg(target_arch = "x86_64")]
    fn enter() -> Self {
        // MXCSR: DAZ=6, rounding-control=13..14, FTZ=15.
        let saved = unsafe { core::arch::x86_64::_mm_getcsr() };
        let strict = saved & !((1 << 6) | (3 << 13) | (1 << 15));
        unsafe { core::arch::x86_64::_mm_setcsr(strict) };
        Self {
            saved: u64::from(saved),
        }
    }
    #[cfg(target_arch = "aarch64")]
    fn enter() -> Self {
        let saved: u64;
        unsafe { core::arch::asm!("mrs {saved}, fpcr", saved = out(reg) saved) };
        // FPCR: FZ16=19, rounding-mode=22..23, FZ=24, DN=25.
        let strict = saved & !((1 << 19) | (3 << 22) | (1 << 24) | (1 << 25));
        unsafe { core::arch::asm!("msr fpcr, {strict}", strict = in(reg) strict) };
        Self { saved }
    }
}

impl Drop for StrictFloatEnvironment {
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_setcsr(self.saved as u32)
        };
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::asm!("msr fpcr, {saved}", saved = in(reg) self.saved)
        };
    }
}

/// What a compiled kernel receives: the launch's bound buffer table (one
/// address per binding slot), the word table (nat/scalar arguments, view
/// geometry, local geometry, grid and workgroup sizes), and the result
/// slot words the elected participant writes.
#[repr(C)]
pub struct LaunchFrame {
    pub buffers: *const *mut u8,
    pub words: *const u64,
    pub results: *mut u64,
}

/// The compiled entry of one kernel: frame, team barrier, linear workgroup
/// index, linear local index, and the workgroup, participant, and register
/// scratch bases.
pub type LaunchEntry = unsafe extern "C-unwind" fn(
    *const LaunchFrame,
    *const TeamBarrier,
    u64,
    u64,
    *mut u8,
    *mut u8,
    *mut u8,
);

// ---------------------------------------------------------------------------
// Team barrier
// ---------------------------------------------------------------------------

struct BarrierState {
    arrived: usize,
    generation: u64,
}

/// A reusable barrier over the members of one team.
pub struct TeamBarrier {
    state: Mutex<BarrierState>,
    released: Condvar,
    /// Members of the current workgroup; `1` short-circuits every wait.
    size: AtomicUsize,
    cancelled: AtomicBool,
}

impl TeamBarrier {
    fn new() -> Self {
        Self {
            state: Mutex::new(BarrierState {
                arrived: 0,
                generation: 0,
            }),
            released: Condvar::new(),
            size: AtomicUsize::new(1),
            cancelled: AtomicBool::new(false),
        }
    }

    fn lock(&self) -> MutexGuard<'_, BarrierState> {
        // Cancellation retains and rethrows the original panic on the
        // submitting thread. Recovering this bookkeeping guard is safe: a
        // cancelled generation is never resumed and reset replaces its
        // arrival count before reuse.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn reset(&self, members: usize) {
        self.cancelled.store(false, Ordering::Release);
        self.size.store(members, Ordering::Release);
        let mut state = self.lock();
        state.arrived = 0;
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.released.notify_all();
    }

    /// Returns false when another worker cancelled the current launch.
    fn wait(&self) -> bool {
        if self.cancelled.load(Ordering::Acquire) {
            return false;
        }
        if self.size.load(Ordering::Acquire) == 1 {
            return true;
        }
        let mut state = self.lock();
        if self.cancelled.load(Ordering::Acquire) {
            return false;
        }
        state.arrived += 1;
        if state.arrived >= self.size.load(Ordering::Acquire) {
            state.arrived = 0;
            state.generation += 1;
            self.released.notify_all();
            return true;
        }
        let generation = state.generation;
        while state.generation == generation && !self.cancelled.load(Ordering::Acquire) {
            // Cancellation retains the original panic payload on `Shared`.
            // Recovering a poisoned bookkeeping guard lets every peer
            // observe cancellation and leave instead of manufacturing a
            // second panic that could strand the team.
            state = self
                .released
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        !self.cancelled.load(Ordering::Acquire)
    }
}

/// The host entry a workgroup barrier compiles to.
pub(crate) extern "C" fn seismic_cpu_barrier(team: *const TeamBarrier) -> i32 {
    // The kernel received this pointer from the worker that runs it; the
    // team outlives the job.
    i32::from(!unsafe { &*team }.wait())
}

// ---------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------

struct Team {
    /// Synchronizes the claim step; never left.
    claim: TeamBarrier,
    /// The workgroup's barrier; reset per workgroup, left on early exit.
    kernel: TeamBarrier,
    current: AtomicU64,
}

#[derive(Clone)]
struct Job {
    entry: LaunchEntry,
    frame: usize,
    workgroups: u64,
    team_size: usize,
    teams: usize,
    /// Workgroup scratch address per team.
    team_scratch: Vec<usize>,
    /// Participant scratch address per worker.
    worker_scratch: Vec<usize>,
    /// Register-local scratch address per worker.
    register_scratch: Vec<usize>,
    next: Arc<AtomicU64>,
    cancelled: Arc<AtomicBool>,
}

#[derive(Default)]
struct State {
    generation: u64,
    job: Option<Job>,
    running: usize,
    shutdown: bool,
    panic: Option<Box<dyn Any + Send + 'static>>,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    done: Condvar,
    teams: Vec<Team>,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn cancel_job(&self, payload: Box<dyn Any + Send + 'static>) {
        {
            let mut state = self.lock();
            if let Some(job) = &state.job {
                job.cancelled.store(true, Ordering::Release);
            }
            if state.panic.is_none() {
                state.panic = Some(payload);
            }
        }
        for team in &self.teams {
            team.claim.cancel();
            team.kernel.cancel();
        }
        self.wake.notify_all();
    }
}

fn work(shared: &Shared, ordinal: usize) {
    let mut seen = 0u64;
    loop {
        let job = {
            let mut state = shared.lock();
            loop {
                if state.shutdown {
                    return;
                }
                if state.generation != seen {
                    break;
                }
                state = shared
                    .wake
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
            seen = state.generation;
            state.job.clone()
        };
        if let Some(job) = job {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _strict_float = StrictFloatEnvironment::enter();
                let team = ordinal / job.team_size;
                let member = (ordinal % job.team_size) as u64;
                if team >= job.teams {
                    return;
                }
                let frame = job.frame as *const LaunchFrame;
                let team_state = &shared.teams[team];
                let team_scratch = job.team_scratch[team] as *mut u8;
                let my_scratch = job.worker_scratch[ordinal] as *mut u8;
                let my_registers = job.register_scratch[ordinal] as *mut u8;
                // The submitter keeps the frame and every scratch alive and
                // unaliased by the host until all workers report done; the
                // kernel's addressing is the compiled plan's.
                if job.team_size == 1 {
                    while !job.cancelled.load(Ordering::Acquire) {
                        let workgroup = job.next.fetch_add(1, Ordering::Relaxed);
                        if workgroup >= job.workgroups {
                            break;
                        }
                        unsafe {
                            (job.entry)(
                                frame,
                                &team_state.kernel,
                                workgroup,
                                0,
                                team_scratch,
                                my_scratch,
                                my_registers,
                            )
                        };
                    }
                } else {
                    'groups: loop {
                        if !team_state.claim.wait() {
                            break;
                        }
                        if member == 0 {
                            team_state.kernel.reset(job.team_size);
                            team_state
                                .current
                                .store(job.next.fetch_add(1, Ordering::Relaxed), Ordering::Release);
                        }
                        if !team_state.claim.wait() {
                            break;
                        }
                        let workgroup = team_state.current.load(Ordering::Acquire);
                        if workgroup >= job.workgroups {
                            break;
                        }
                        unsafe {
                            (job.entry)(
                                frame,
                                &team_state.kernel,
                                workgroup,
                                member,
                                team_scratch,
                                my_scratch,
                                my_registers,
                            )
                        };
                        if job.cancelled.load(Ordering::Acquire) {
                            break 'groups;
                        }
                    }
                }
            }));
            if let Err(payload) = result {
                shared.cancel_job(payload);
            }
        }
        let mut state = shared.lock();
        state.running -= 1;
        if state.running == 0 {
            shared.done.notify_all();
        }
    }
}

fn grow(scratch: &mut Option<Buffer>, bytes: u64) -> Result<(), AllocationFailure> {
    if scratch
        .as_ref()
        .is_some_and(|scratch| scratch.len() >= bytes)
    {
        return Ok(());
    }
    *scratch = Some(Buffer::new(bytes, SCRATCH_ALIGNMENT)?);
    Ok(())
}

fn scratch_pointer(scratch: &Option<Buffer>) -> usize {
    scratch
        .as_ref()
        .map(|scratch| scratch.data_pointer() as usize)
        .unwrap_or(std::ptr::NonNull::<u8>::dangling().as_ptr() as usize)
}

/// The worker threads of one CPU device.
pub struct Workers {
    shared: Arc<Shared>,
    threads: Vec<std::thread::JoinHandle<()>>,
    team_scratch: Vec<Option<Buffer>>,
    worker_scratch: Vec<Option<Buffer>>,
    register_scratch: Vec<Option<Buffer>>,
}

/// Why a launch could not be submitted to the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchFailure {
    /// Workgroup or participant scratch could not be grown to the launch's
    /// size.
    Scratch(AllocationFailure),
}

impl Workers {
    /// A pool of `count` workers (at least one).
    pub fn new(count: usize) -> Result<Self, std::io::Error> {
        let count = count.max(1);
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            done: Condvar::new(),
            teams: (0..count)
                .map(|_| Team {
                    claim: TeamBarrier::new(),
                    kernel: TeamBarrier::new(),
                    current: AtomicU64::new(0),
                })
                .collect(),
        });
        let mut threads = Vec::with_capacity(count);
        for ordinal in 0..count {
            let shared = shared.clone();
            let thread = std::thread::Builder::new()
                .name(format!("seismic-cpu-{ordinal}"))
                .spawn(move || work(&shared, ordinal))?;
            threads.push(thread);
        }
        Ok(Workers {
            shared,
            threads,
            team_scratch: (0..count).map(|_| None).collect(),
            worker_scratch: (0..count).map(|_| None).collect(),
            register_scratch: (0..count).map(|_| None).collect(),
        })
    }

    /// One worker per unit of host parallelism.
    pub fn host() -> Result<Self, std::io::Error> {
        Self::new(std::thread::available_parallelism()?.get())
    }

    pub fn count(&self) -> usize {
        self.threads.len()
    }

    /// Runs `workgroups` workgroups of `team_size` threads each and returns
    /// after all complete. `team_size` is at most the worker count: the profile
    /// bounds every workgroup by it, so a larger value contradicts the
    /// private prepared-kernel constructor (§13.3.6).
    pub fn run(
        &mut self,
        entry: LaunchEntry,
        frame: &LaunchFrame,
        workgroups: u64,
        team_size: u64,
        workgroup_scratch_bytes: u64,
        participant_scratch_bytes: u64,
        register_scratch_bytes: u64,
    ) -> Result<(), LaunchFailure> {
        if workgroups == 0 || team_size == 0 {
            return Ok(());
        }
        let workers = self.threads.len();
        let team_size = match usize::try_from(team_size) {
            Ok(size) if size <= workers => size,
            _ => panic!(
                "PreparedKernel coverage invariant violated: a launch asks for {team_size} workgroup threads on a profile of {workers} workers"
            ),
        };
        let teams = workers / team_size;
        for scratch in self.team_scratch.iter_mut().take(teams) {
            grow(scratch, workgroup_scratch_bytes).map_err(LaunchFailure::Scratch)?;
        }
        for scratch in self.worker_scratch.iter_mut().take(teams * team_size) {
            grow(scratch, participant_scratch_bytes).map_err(LaunchFailure::Scratch)?;
        }
        for scratch in self.register_scratch.iter_mut().take(teams * team_size) {
            grow(scratch, register_scratch_bytes).map_err(LaunchFailure::Scratch)?;
        }
        for team in self.shared.teams.iter().take(teams) {
            team.claim.reset(team_size);
            team.kernel.reset(team_size);
            team.current.store(0, Ordering::Release);
        }
        let job = Job {
            entry,
            frame: frame as *const LaunchFrame as usize,
            workgroups,
            team_size,
            teams,
            team_scratch: self.team_scratch.iter().map(scratch_pointer).collect(),
            worker_scratch: self.worker_scratch.iter().map(scratch_pointer).collect(),
            register_scratch: self.register_scratch.iter().map(scratch_pointer).collect(),
            next: Arc::new(AtomicU64::new(0)),
            cancelled: Arc::new(AtomicBool::new(false)),
        };
        let mut state = self.shared.lock();
        // `run` always takes the retained payload before either returning or
        // resuming it on the submitter. There is no cross-launch panic state
        // to validate here: the state transition below owns the new job.
        state.generation += 1;
        state.job = Some(job);
        state.running = workers;
        self.shared.wake.notify_all();
        while state.running != 0 {
            state = self
                .shared
                .done
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        state.job = None;
        let panic = state.panic.take();
        drop(state);
        if let Some(payload) = panic {
            std::panic::resume_unwind(payload);
        }
        Ok(())
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        if let Ok(mut state) = self.shared.state.lock() {
            state.shutdown = true;
        }
        self.shared.wake.notify_all();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}
