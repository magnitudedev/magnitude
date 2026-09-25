//! The worker pool of one CPU device: its participants, job publication,
//! work-item claiming, and the barriers participants meet at.
//!
//! The pool has one participant per physical performance core
//! ([`Workers::host`]). The submitting thread is participant 0 and works
//! alongside the pool's threads (participants 1..): a job runs on every
//! participant and the submitter returns once every participant has left
//! it, so the frame, buffer table, words and scratch the job borrows stay
//! valid for its duration.
//!
//! Two kinds of job run on the pool:
//!
//! - one compiled launch ([`Workers::run`]): participants form teams of the
//!   launch's workgroup thread count; each team claims workgroups from the
//!   launch's counter, its members running one thread of the workgroup each;
//! - a sequence of authored native launches ([`Workers::run_native`]): every
//!   participant walks the steps in order; a step's participants (at most
//!   its item count) claim its work items from its counter and meet the next
//!   step's participants at a barrier after it. A whole native graph
//!   therefore wakes the pool once.
//!
//! Waiting participants spin for a short, fixed bound and then park. Job
//! state, counters and scratch are owned by the pool and reused.

use std::any::Any;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::Thread;

use crate::buffer::{AllocationFailure, Buffer};
use crate::profile::SCRATCH_ALIGNMENT;

/// Iterations a waiting participant spins on the condition.
const SPIN_LIMIT: u32 = 1 << 10;
/// Further checks, each after yielding the core, before it parks. Yielding
/// lets a descheduled participant, which every waiter depends on, run when
/// the host is oversubscribed.
const YIELD_LIMIT: u32 = 1 << 8;

/// Restores the caller's floating environment after forcing the reference
/// recipe's primitive contract: round-to-nearest-even, gradual underflow and
/// non-canonicalized NaNs. Every participant enters this guard before native
/// code.
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

    /// Returns false when another participant cancelled the current launch.
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
    // The kernel received this pointer from the participant that runs it;
    // the team outlives the job.
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

/// Work every participant runs its share of.
trait Job: Sync {
    /// Participant `ordinal`'s share. Returns when the participant's part of
    /// the job is complete or the job was cancelled.
    fn run(&self, shared: &Shared, ordinal: usize);
}

/// A job published to the pool's threads. The submitter keeps the pointee
/// alive until every thread has left it.
#[derive(Clone, Copy)]
struct Published(*const (dyn Job + 'static));

/// Bits of [`Shared::published`] holding the participant count of the job.
const PARTICIPANT_BITS: u32 = 16;

struct Shared {
    participants: usize,
    /// The latest job publication: its epoch (incremented once per
    /// published job) above [`PARTICIPANT_BITS`], and the participants it
    /// runs on below. One word, so a thread never pairs one job's epoch with
    /// another job's participant count.
    published: AtomicU64,
    /// The current job; written by the submitter before `published`
    /// advances and read by its participants after they observe the advance.
    job: UnsafeCell<Option<Published>>,
    /// Pool threads that have not yet left the current job.
    remaining: AtomicUsize,
    cancelled: AtomicBool,
    panic: Mutex<Option<Box<dyn Any + Send + 'static>>>,
    shutdown: AtomicBool,
    /// Whether each participant is parked, or about to park, in `wait_until`.
    parked: Vec<AtomicBool>,
    /// Participant 0's thread for the current job.
    submitter: Mutex<Option<Thread>>,
    /// Participants 1.. .
    threads: OnceLock<Vec<Thread>>,
    teams: Vec<Team>,
}

// SAFETY: `job` is written only by the submitter while no pool thread is in
// a job (`remaining == 0`), and published to the threads through the SeqCst
// `epoch` advance that follows the write. The pointee is `Sync`.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

impl Shared {
    /// Spins, then yields, until `ready`; then parks participant `ordinal`
    /// until `ready`. An idle pool thread waits the same way for the next
    /// job, so back-to-back jobs find it running where it ran before.
    fn wait_until(&self, ordinal: usize, ready: impl Fn() -> bool) {
        for _ in 0..SPIN_LIMIT {
            if ready() {
                return;
            }
            std::hint::spin_loop();
        }
        for _ in 0..YIELD_LIMIT {
            if ready() {
                return;
            }
            std::thread::yield_now();
        }
        self.park_until(ordinal, ready);
    }

    /// Parks participant `ordinal` until `ready`. Every writer of a condition
    /// a participant waits on publishes it with a SeqCst store and then
    /// wakes the participants that wait on it.
    fn park_until(&self, ordinal: usize, ready: impl Fn() -> bool) {
        loop {
            self.parked[ordinal].store(true, Ordering::SeqCst);
            if ready() {
                self.parked[ordinal].store(false, Ordering::SeqCst);
                return;
            }
            #[cfg(test)]
            tests::PARKS.fetch_add(1, Ordering::Relaxed);
            std::thread::park();
            if ready() {
                self.parked[ordinal].store(false, Ordering::SeqCst);
                return;
            }
        }
    }

    /// Unparks every parked participant.
    fn wake(&self) {
        self.wake_range(0..self.participants);
    }

    /// Unparks the parked participants among `ordinals`.
    fn wake_range(&self, ordinals: std::ops::Range<usize>) {
        for ordinal in ordinals {
            if self.parked[ordinal].swap(false, Ordering::SeqCst) {
                self.thread(ordinal).unpark();
            }
        }
    }

    fn thread(&self, ordinal: usize) -> Thread {
        if ordinal == 0 {
            self.submitter
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone()
                .expect("participant 0 parks only inside a job")
        } else {
            self.threads
                .get()
                .expect("pool threads are registered before any job")[ordinal - 1]
                .clone()
        }
    }

    fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Records the first panic of the job and makes every participant leave.
    fn cancel(&self, payload: Box<dyn Any + Send + 'static>) {
        {
            let mut panic = self
                .panic
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if panic.is_none() {
                *panic = Some(payload);
            }
        }
        self.cancelled.store(true, Ordering::SeqCst);
        for team in &self.teams {
            team.claim.cancel();
            team.kernel.cancel();
        }
        self.wake();
    }

    /// Runs participant `ordinal`'s share of `job` in the strict floating
    /// environment, cancelling the job if it panics.
    fn participate(&self, job: &dyn Job, ordinal: usize) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _strict_float = StrictFloatEnvironment::enter();
            job.run(self, ordinal);
        }));
        if let Err(payload) = result {
            self.cancel(payload);
        }
    }
}

fn work(shared: &Shared, ordinal: usize) {
    prefer_performance_cores();
    let mut seen = 0u64;
    loop {
        shared.wait_until(ordinal, || {
            shared.published.load(Ordering::SeqCst) >> PARTICIPANT_BITS != seen
                || shared.shutdown.load(Ordering::SeqCst)
        });
        if shared.shutdown.load(Ordering::SeqCst) {
            return;
        }
        let published = shared.published.load(Ordering::SeqCst);
        seen = published >> PARTICIPANT_BITS;
        if ordinal as u64 >= published & ((1 << PARTICIPANT_BITS) - 1) {
            // Not a participant of this job; it was not woken for it.
            continue;
        }
        // SAFETY: the submitter wrote the job before publishing it and
        // keeps it alive until `remaining`, which counts this thread,
        // reaches zero.
        let Published(job) = unsafe { *shared.job.get() }.expect("a publication carries a job");
        shared.participate(unsafe { &*job }, ordinal);
        if shared.remaining.fetch_sub(1, Ordering::SeqCst) == 1 {
            shared.wake_range(0..1);
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

fn scratch_pointer(scratch: &Option<Buffer>) -> *mut u8 {
    scratch
        .as_ref()
        .map(|scratch| scratch.data_pointer())
        .unwrap_or(std::ptr::NonNull::<u8>::dangling().as_ptr())
}

/// One participant's scratch of each class, with its current base address.
struct Scratch {
    buffer: Option<Buffer>,
    base: *mut u8,
}

impl Scratch {
    fn new() -> Self {
        Self {
            buffer: None,
            base: std::ptr::NonNull::<u8>::dangling().as_ptr(),
        }
    }

    fn grow(&mut self, bytes: u64) -> Result<(), AllocationFailure> {
        grow(&mut self.buffer, bytes)?;
        self.base = scratch_pointer(&self.buffer);
        Ok(())
    }
}

/// One launch of a sequence of authored native launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeStep {
    /// Independent work items.
    pub items: u64,
    /// Private bytes each item receives.
    pub shared_bytes: u64,
    /// Participants that may claim the step's items, in `1..=count()`. No
    /// more participants than items take part.
    pub workers: usize,
}

impl NativeStep {
    /// Participants `0..active()` claim the step's items.
    fn active(&self) -> usize {
        self.workers
            .min(usize::try_from(self.items).unwrap_or(usize::MAX))
            .max(1)
    }
}

/// A sequence of authored native launches, each depending on the ones
/// before it.
pub trait NativeSteps: Sync {
    fn count(&self) -> usize;
    fn step(&self, index: usize) -> NativeStep;
    /// Runs work item `item` of step `index` with its private bytes.
    fn run(&self, index: usize, item: u64, shared: &mut [u8]);
}

/// Per-step state of a native job, reset before each job.
struct StepState {
    /// Next unclaimed item.
    claimed: AtomicU64,
    /// Participants that have arrived at the boundary after the step.
    arrived: AtomicUsize,
}

struct NativeJob<'a> {
    steps: &'a dyn NativeSteps,
    state: &'a [StepState],
    /// Private-bytes base of each participant, grown to every step's size.
    scratch: &'a [*mut u8],
}

// SAFETY: participant `ordinal` alone writes through `scratch[ordinal]`.
unsafe impl Sync for NativeJob<'_> {}

impl NativeJob<'_> {
    /// The boundary after step `index`: the participants of that step and
    /// of the next one meet, so the next step's participants see every
    /// write of that step. Others pass by; each boundary has its own count,
    /// so a participant that passes one cannot be counted at another. Every
    /// boundary includes participant 0, which chains the boundaries in step
    /// order. Returns false when the job was cancelled.
    fn boundary(&self, shared: &Shared, ordinal: usize, index: usize, active: usize) -> bool {
        let meeting = active.max(self.steps.step(index + 1).active());
        if ordinal >= meeting {
            return !shared.cancelled();
        }
        let arrived = &self.state[index].arrived;
        if arrived.fetch_add(1, Ordering::SeqCst) + 1 == meeting {
            shared.wake_range(0..meeting);
        } else {
            shared.wait_until(ordinal, || {
                arrived.load(Ordering::SeqCst) == meeting || shared.cancelled()
            });
        }
        !shared.cancelled()
    }
}

impl Job for NativeJob<'_> {
    fn run(&self, shared: &Shared, ordinal: usize) {
        let count = self.steps.count();
        for index in 0..count {
            let step = self.steps.step(index);
            let active = step.active();
            if ordinal < active {
                // SAFETY: the pool grew this participant's scratch to at
                // least `shared_bytes` before publishing the job.
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        self.scratch[ordinal],
                        step.shared_bytes as usize,
                    )
                };
                // Participant `ordinal` owns item `ordinal`; the items past
                // the first `active` are claimed. A step with no more items
                // than participants touches no shared counter.
                let mut item = ordinal as u64;
                while item < step.items && !shared.cancelled() {
                    self.steps.run(index, item, bytes);
                    item =
                        active as u64 + self.state[index].claimed.fetch_add(1, Ordering::Relaxed);
                }
            }
            if index + 1 < count && !self.boundary(shared, ordinal, index, active) {
                return;
            }
        }
    }
}

struct LaunchJob {
    entry: LaunchEntry,
    frame: *const LaunchFrame,
    workgroups: u64,
    team_size: usize,
    teams: usize,
    next: AtomicU64,
    team_scratch: *const *mut u8,
    worker_scratch: *const *mut u8,
    register_scratch: *const *mut u8,
}

// SAFETY: the frame and scratch tables outlive the job; each participant
// writes only its own participant and register scratch and its team's
// workgroup scratch, as the compiled plan addresses them.
unsafe impl Sync for LaunchJob {}

impl Job for LaunchJob {
    fn run(&self, shared: &Shared, ordinal: usize) {
        let team = ordinal / self.team_size;
        let member = (ordinal % self.team_size) as u64;
        if team >= self.teams {
            return;
        }
        let team_state = &shared.teams[team];
        let (team_scratch, my_scratch, my_registers) = unsafe {
            (
                *self.team_scratch.add(team),
                *self.worker_scratch.add(ordinal),
                *self.register_scratch.add(ordinal),
            )
        };
        if self.team_size == 1 {
            while !shared.cancelled() {
                let workgroup = self.next.fetch_add(1, Ordering::Relaxed);
                if workgroup >= self.workgroups {
                    break;
                }
                unsafe {
                    (self.entry)(
                        self.frame,
                        &team_state.kernel,
                        workgroup,
                        0,
                        team_scratch,
                        my_scratch,
                        my_registers,
                    )
                };
            }
            return;
        }
        loop {
            if !team_state.claim.wait() {
                break;
            }
            if member == 0 {
                team_state.kernel.reset(self.team_size);
                team_state
                    .current
                    .store(self.next.fetch_add(1, Ordering::Relaxed), Ordering::Release);
            }
            if !team_state.claim.wait() {
                break;
            }
            let workgroup = team_state.current.load(Ordering::Acquire);
            if workgroup >= self.workgroups {
                break;
            }
            unsafe {
                (self.entry)(
                    self.frame,
                    &team_state.kernel,
                    workgroup,
                    member,
                    team_scratch,
                    my_scratch,
                    my_registers,
                )
            };
            if shared.cancelled() {
                break;
            }
        }
    }
}

/// The participants of one CPU device.
pub struct Workers {
    shared: Arc<Shared>,
    threads: Vec<std::thread::JoinHandle<()>>,
    team_scratch: Vec<Scratch>,
    worker_scratch: Vec<Scratch>,
    register_scratch: Vec<Scratch>,
    /// Base addresses of the scratch above, in participant order.
    team_bases: Vec<*mut u8>,
    worker_bases: Vec<*mut u8>,
    register_bases: Vec<*mut u8>,
    /// Per-step state for the largest native job so far.
    steps: Vec<StepState>,
}

// SAFETY: the base tables mirror the owned scratch buffers; the pool is
// driven through `&mut self` by one submitter at a time.
unsafe impl Send for Workers {}

/// Why a launch could not be submitted to the pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchFailure {
    /// Workgroup or participant scratch could not be grown to the launch's
    /// size.
    Scratch(AllocationFailure),
}

impl Workers {
    /// A pool of `count` participants (at least one): the submitting thread
    /// and `count - 1` pool threads.
    pub fn new(count: usize) -> Result<Self, std::io::Error> {
        let count = count.clamp(1, (1 << PARTICIPANT_BITS) - 1);
        let shared = Arc::new(Shared {
            participants: count,
            published: AtomicU64::new(0),
            job: UnsafeCell::new(None),
            remaining: AtomicUsize::new(0),
            cancelled: AtomicBool::new(false),
            panic: Mutex::new(None),
            shutdown: AtomicBool::new(false),
            parked: (0..count).map(|_| AtomicBool::new(false)).collect(),
            submitter: Mutex::new(None),
            threads: OnceLock::new(),
            teams: (0..count)
                .map(|_| Team {
                    claim: TeamBarrier::new(),
                    kernel: TeamBarrier::new(),
                    current: AtomicU64::new(0),
                })
                .collect(),
        });
        let mut threads = Vec::with_capacity(count - 1);
        for ordinal in 1..count {
            let shared = shared.clone();
            let thread = std::thread::Builder::new()
                .name(format!("seismic-cpu-{ordinal}"))
                .spawn(move || work(&shared, ordinal))?;
            threads.push(thread);
        }
        shared
            .threads
            .set(
                threads
                    .iter()
                    .map(|thread| thread.thread().clone())
                    .collect(),
            )
            .unwrap_or_else(|_| unreachable!("registered once, here"));
        let dangling = || vec![std::ptr::NonNull::<u8>::dangling().as_ptr(); count];
        Ok(Workers {
            shared,
            threads,
            team_scratch: (0..count).map(|_| Scratch::new()).collect(),
            worker_scratch: (0..count).map(|_| Scratch::new()).collect(),
            register_scratch: (0..count).map(|_| Scratch::new()).collect(),
            team_bases: dangling(),
            worker_bases: dangling(),
            register_bases: dangling(),
            steps: Vec::new(),
        })
    }

    /// One participant per physical performance core of the host.
    pub fn host() -> Result<Self, std::io::Error> {
        Self::new(performance_cores()?)
    }

    /// Participants, the submitting thread included.
    pub fn count(&self) -> usize {
        self.shared.participants
    }

    /// Runs `job` on participants `0..participants` (the submitter and the
    /// first `participants - 1` pool threads, which alone are woken); returns
    /// after all have left it, resuming the first panic of any participant.
    fn execute(&self, job: &dyn Job, participants: usize) {
        let shared = &*self.shared;
        debug_assert!((1..=shared.participants).contains(&participants));
        shared.cancelled.store(false, Ordering::SeqCst);
        *shared
            .submitter
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(std::thread::current());
        let pool = participants > 1;
        if pool {
            // SAFETY: no pool thread is in a job (`remaining` is zero), and
            // `job` outlives the wait for `remaining` below. Only the
            // lifetime is erased; the pointer is cleared before returning.
            let job: *const (dyn Job + '_) = job;
            let job: *const (dyn Job + 'static) = unsafe { std::mem::transmute(job) };
            unsafe { *shared.job.get() = Some(Published(job)) };
            shared.remaining.store(participants - 1, Ordering::SeqCst);
            let epoch =
                (shared.published.load(Ordering::SeqCst) >> PARTICIPANT_BITS).wrapping_add(1);
            shared.published.store(
                epoch << PARTICIPANT_BITS | participants as u64,
                Ordering::SeqCst,
            );
            shared.wake_range(1..participants);
        }
        {
            let _class = pool.then(SubmitterClass::enter);
            shared.participate(job, 0);
        }
        if pool {
            shared.wait_until(0, || shared.remaining.load(Ordering::SeqCst) == 0);
            unsafe { *shared.job.get() = None };
        }
        let panic = shared
            .panic
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        if let Some(payload) = panic {
            std::panic::resume_unwind(payload);
        }
    }

    /// Runs `workgroups` workgroups of `team_size` threads each and returns
    /// after all complete. `team_size` is at most the participant count: the
    /// profile bounds every workgroup by it, so a larger value contradicts
    /// the private prepared-kernel constructor (§13.3.6).
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
        let participants = self.count();
        let team_size = match usize::try_from(team_size) {
            Ok(size) if size <= participants => size,
            _ => panic!(
                "PreparedKernel coverage invariant violated: a launch asks for {team_size} workgroup threads on a profile of {participants} workers"
            ),
        };
        let teams =
            (participants / team_size).min(usize::try_from(workgroups).unwrap_or(usize::MAX));
        for (scratch, base) in self
            .team_scratch
            .iter_mut()
            .zip(&mut self.team_bases)
            .take(teams)
        {
            scratch
                .grow(workgroup_scratch_bytes)
                .map_err(LaunchFailure::Scratch)?;
            *base = scratch.base;
        }
        let members = teams * team_size;
        for (scratch, base) in self
            .worker_scratch
            .iter_mut()
            .zip(&mut self.worker_bases)
            .take(members)
        {
            scratch
                .grow(participant_scratch_bytes)
                .map_err(LaunchFailure::Scratch)?;
            *base = scratch.base;
        }
        for (scratch, base) in self
            .register_scratch
            .iter_mut()
            .zip(&mut self.register_bases)
            .take(members)
        {
            scratch
                .grow(register_scratch_bytes)
                .map_err(LaunchFailure::Scratch)?;
            *base = scratch.base;
        }
        for team in self.shared.teams.iter().take(teams) {
            team.claim.reset(team_size);
            team.kernel.reset(team_size);
            team.current.store(0, Ordering::Release);
        }
        self.execute(
            &LaunchJob {
                entry,
                frame,
                workgroups,
                team_size,
                teams,
                next: AtomicU64::new(0),
                team_scratch: self.team_bases.as_ptr(),
                worker_scratch: self.worker_bases.as_ptr(),
                register_scratch: self.register_bases.as_ptr(),
            },
            members,
        );
        Ok(())
    }

    /// Runs `steps` in order as one job and returns after all complete: the
    /// participants of each step claim its items and meet the next step's
    /// participants after it. Only the participants some step uses are
    /// woken; a job whose steps each have one runs on the submitter alone.
    pub fn run_native(&mut self, steps: &dyn NativeSteps) -> Result<(), LaunchFailure> {
        let count = steps.count();
        if count == 0 {
            return Ok(());
        }
        let participants = self.count();
        let mut shared_bytes = 0;
        for index in 0..count {
            let step = steps.step(index);
            assert!(
                (1..=participants).contains(&step.workers),
                "a native step asks for {} workers on a pool of {participants}",
                step.workers
            );
            shared_bytes = shared_bytes.max(step.shared_bytes);
        }
        for (scratch, base) in self.worker_scratch.iter_mut().zip(&mut self.worker_bases) {
            scratch.grow(shared_bytes).map_err(LaunchFailure::Scratch)?;
            *base = scratch.base;
        }
        if self.steps.len() < count {
            self.steps.resize_with(count, || StepState {
                claimed: AtomicU64::new(0),
                arrived: AtomicUsize::new(0),
            });
        }
        for state in &self.steps[..count] {
            state.claimed.store(0, Ordering::Relaxed);
            state.arrived.store(0, Ordering::Relaxed);
        }
        let participants = (0..count)
            .map(|index| steps.step(index).active())
            .max()
            .unwrap_or(1);
        self.execute(
            &NativeJob {
                steps,
                state: &self.steps[..count],
                scratch: &self.worker_bases,
            },
            participants,
        );
        Ok(())
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        self.shared.wake();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

/// Physical performance cores available to this process: SMT siblings count
/// once and efficiency cores are excluded, since a second hardware thread on
/// a saturated core or a slower core delays every barrier.
fn performance_cores() -> Result<usize, std::io::Error> {
    let available = std::thread::available_parallelism()?.get();
    Ok(physical_performance_cores().map_or(available, |cores| cores.min(available)))
}

#[cfg(target_os = "macos")]
fn physical_performance_cores() -> Option<usize> {
    fn sysctl(name: &std::ffi::CStr) -> Option<usize> {
        let mut value: i32 = 0;
        let mut size = std::mem::size_of::<i32>();
        // SAFETY: `value` and `size` describe a writable i32 as the named
        // integer sysctls require.
        let status = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&mut value as *mut i32).cast(),
                &mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        (status == 0 && value > 0).then_some(value as usize)
    }
    // `perflevel0` is the performance cluster on hybrid parts; Intel Macs
    // have no performance levels.
    sysctl(c"hw.perflevel0.physicalcpu").or_else(|| sysctl(c"hw.physicalcpu"))
}

#[cfg(target_os = "linux")]
fn physical_performance_cores() -> Option<usize> {
    use std::collections::BTreeSet;
    let read = |path: &str| std::fs::read_to_string(path).ok();
    // Hybrid Intel parts list their performance cores here.
    let performance = read("/sys/devices/cpu_core/cpus").map(|list| parse_cpu_list(list.trim()));
    let online = parse_cpu_list(read("/sys/devices/system/cpu/online")?.trim());
    let mut cores = BTreeSet::new();
    for cpu in online {
        if performance
            .as_ref()
            .is_some_and(|performance| !performance.contains(&cpu))
        {
            continue;
        }
        let siblings = read(&format!(
            "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        ))?;
        cores.insert(siblings.trim().to_owned());
    }
    (!cores.is_empty()).then_some(cores.len())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn physical_performance_cores() -> Option<usize> {
    None
}

/// Asks the scheduler to keep the calling pool thread on performance cores.
/// On macOS a thread of default quality of service may run on an efficiency
/// core, and every barrier would wait for it.
#[cfg(target_os = "macos")]
fn prefer_performance_cores() {
    // SAFETY: sets the calling thread's own class; no pointers.
    unsafe {
        libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0)
    };
}

#[cfg(not(target_os = "macos"))]
fn prefer_performance_cores() {}

/// Keeps the submitting thread, participant 0, on performance cores for the
/// duration of a job and restores its class afterwards.
struct SubmitterClass {
    #[cfg(target_os = "macos")]
    saved: Option<libc::qos_class_t>,
}

impl SubmitterClass {
    #[cfg(target_os = "macos")]
    fn enter() -> Self {
        let mut current = libc::qos_class_t::QOS_CLASS_UNSPECIFIED;
        let mut priority = 0;
        // SAFETY: reads the calling thread's own class into locals.
        unsafe {
            libc::pthread_get_qos_class_np(libc::pthread_self(), &mut current, &mut priority)
        };
        if matches!(current, libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE) {
            return Self { saved: None };
        }
        prefer_performance_cores();
        Self {
            saved: Some(current),
        }
    }

    #[cfg(not(target_os = "macos"))]
    fn enter() -> Self {
        Self {}
    }
}

impl Drop for SubmitterClass {
    fn drop(&mut self) {
        #[cfg(target_os = "macos")]
        if let Some(saved) = self.saved {
            // SAFETY: restores the calling thread's own class.
            unsafe { libc::pthread_set_qos_class_self_np(saved, 0) };
        }
    }
}

/// CPU numbers of a sysfs list such as `0-3,8,10-11`.
#[cfg(target_os = "linux")]
fn parse_cpu_list(list: &str) -> Vec<usize> {
    list.split(',')
        .filter(|range| !range.is_empty())
        .flat_map(|range| {
            let (first, last) = range.split_once('-').unwrap_or((range, range));
            match (first.parse::<usize>(), last.parse::<usize>()) {
                (Ok(first), Ok(last)) => first..=last,
                _ => 1..=0,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) static PARKS: AtomicU64 = AtomicU64::new(0);

    struct Counting {
        steps: Vec<NativeStep>,
        hits: Vec<Vec<AtomicU64>>,
        /// Items of the previous step seen complete when each step starts.
        ordered: AtomicBool,
    }

    impl NativeSteps for Counting {
        fn count(&self) -> usize {
            self.steps.len()
        }
        fn step(&self, index: usize) -> NativeStep {
            self.steps[index]
        }
        fn run(&self, index: usize, item: u64, shared: &mut [u8]) {
            assert_eq!(shared.len() as u64, self.steps[index].shared_bytes);
            shared.fill(item as u8);
            if index > 0 {
                let previous = &self.hits[index - 1];
                if previous.iter().any(|hit| hit.load(Ordering::SeqCst) != 1) {
                    self.ordered.store(false, Ordering::SeqCst);
                }
            }
            self.hits[index][item as usize].fetch_add(1, Ordering::SeqCst);
        }
    }

    fn counting(workers: &Workers, sizes: &[u64]) -> Counting {
        Counting {
            steps: sizes
                .iter()
                .enumerate()
                .map(|(index, items)| NativeStep {
                    items: *items,
                    shared_bytes: 16 * index as u64,
                    workers: 1 + index % workers.count(),
                })
                .collect(),
            hits: sizes
                .iter()
                .map(|items| (0..*items).map(|_| AtomicU64::new(0)).collect())
                .collect(),
            ordered: AtomicBool::new(true),
        }
    }

    #[test]
    fn a_native_job_runs_every_item_once_in_step_order() {
        let mut workers = Workers::new(4).expect("pool");
        for round in 0..50 {
            let job = counting(&workers, &[37, 0, 5, 128, 1, 64 + round]);
            workers.run_native(&job).expect("run");
            assert!(job.ordered.load(Ordering::SeqCst));
            for hits in &job.hits {
                assert!(hits.iter().all(|hit| hit.load(Ordering::SeqCst) == 1));
            }
        }
    }

    #[test]
    fn a_single_participant_pool_runs_on_the_submitter() {
        let mut workers = Workers::new(1).expect("pool");
        let job = counting(&workers, &[3, 4]);
        workers.run_native(&job).expect("run");
        assert!(job
            .hits
            .iter()
            .flatten()
            .all(|hit| hit.load(Ordering::SeqCst) == 1));
    }

    #[test]
    fn a_panicking_item_is_resumed_on_the_submitter_and_the_pool_recovers() {
        struct Panicking;
        impl NativeSteps for Panicking {
            fn count(&self) -> usize {
                3
            }
            fn step(&self, _: usize) -> NativeStep {
                NativeStep {
                    items: 64,
                    shared_bytes: 0,
                    workers: 3,
                }
            }
            fn run(&self, index: usize, item: u64, _: &mut [u8]) {
                if index == 1 && item == 7 {
                    panic!("item failed");
                }
            }
        }
        let mut workers = Workers::new(3).expect("pool");
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            workers.run_native(&Panicking).expect("run")
        }));
        let payload = caught.expect_err("the panic reaches the submitter");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"item failed"));
        let job = counting(&workers, &[9, 9, 9]);
        workers.run_native(&job).expect("run");
        assert!(job
            .hits
            .iter()
            .flatten()
            .all(|hit| hit.load(Ordering::SeqCst) == 1));
    }

    #[test]
    fn parked_participants_are_woken_for_the_next_job() {
        let mut workers = Workers::new(4).expect("pool");
        for _ in 0..3 {
            // Long enough for every pool thread to exhaust its spin and park.
            std::thread::sleep(std::time::Duration::from_millis(20));
            let job = counting(&workers, &[16, 16]);
            workers.run_native(&job).expect("run");
            assert!(job
                .hits
                .iter()
                .flatten()
                .all(|hit| hit.load(Ordering::SeqCst) == 1));
        }
    }

    #[test]
    fn an_idle_pool_parks_every_thread() {
        let mut workers = Workers::new(4).expect("pool");
        let job = counting(&workers, &[16, 16]);
        workers.run_native(&job).expect("run");
        let parked = || {
            workers.shared.parked[1..]
                .iter()
                .all(|parked| parked.load(Ordering::SeqCst))
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !parked() {
            assert!(
                std::time::Instant::now() < deadline,
                "an idle pool thread kept running"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    #[test]
    #[ignore = "measurement; run with --ignored --nocapture"]
    fn step_boundary_cost() {
        struct Tiny {
            steps: usize,
            items: u64,
            workers: usize,
        }
        impl NativeSteps for Tiny {
            fn count(&self) -> usize {
                self.steps
            }
            fn step(&self, _: usize) -> NativeStep {
                NativeStep {
                    items: self.items,
                    shared_bytes: 0,
                    workers: self.workers,
                }
            }
            fn run(&self, _: usize, item: u64, _: &mut [u8]) {
                std::hint::black_box(item);
            }
        }
        let mut workers = Workers::host().expect("pool");
        let count = workers.count();
        for items in [2, 3, 4, 6, 8, 12] {
            // Samples of about 5 ms, as the tuner takes them.
            let job = Tiny {
                steps: 20_000,
                items,
                workers: count,
            };
            let mut samples = (0..15)
                .map(|_| {
                    let started = std::time::Instant::now();
                    workers.run_native(&job).expect("run");
                    started.elapsed().as_secs_f64()
                })
                .collect::<Vec<_>>();
            samples.sort_by(f64::total_cmp);
            let median = samples[7];
            let mut deviations = samples
                .iter()
                .map(|sample| (sample - median).abs())
                .collect::<Vec<_>>();
            deviations.sort_by(f64::total_cmp);
            eprintln!(
                "{items} items, 20000 steps: median {:.2} ms, MAD {:.1}%, range {:.2}..{:.2} ms",
                median * 1e3,
                deviations[7] / median * 100.0,
                samples[0] * 1e3,
                samples[14] * 1e3
            );
        }
        for items in [1, 2, 3, 4, 5, 6, 8, 12, 64] {
            let job = Tiny {
                steps: 2_000,
                items,
                workers: count,
            };
            let parks = PARKS.load(Ordering::Relaxed);
            let mut samples = (0..12)
                .map(|_| {
                    let started = std::time::Instant::now();
                    workers.run_native(&job).expect("run");
                    started.elapsed().as_secs_f64() / job.steps as f64 * 1e6
                })
                .collect::<Vec<_>>();
            samples.sort_by(f64::total_cmp);
            eprintln!(
                "{count} participants, {items:>2} items: per step min {:.2} median {:.2} max {:.2} us; parks per job {:.1}",
                samples[0],
                samples[6],
                samples[11],
                (PARKS.load(Ordering::Relaxed) - parks) as f64 / 12.0
            );
        }
    }

    #[test]
    fn the_host_pool_counts_physical_performance_cores() {
        let cores = performance_cores().expect("host parallelism");
        assert!(cores >= 1);
        assert!(
            cores
                <= std::thread::available_parallelism()
                    .expect("host parallelism")
                    .get()
        );
    }
}
