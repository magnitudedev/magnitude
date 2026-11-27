use super::queue::{Piece, Queue, Returned};
use crate::disk::{VolumeReads, LEAST_READ};
#[cfg(doc)]
use crate::Reader;
use crate::{Buffer, Error, Totals, Volume};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

/// A reader's requests made and not yet ended ([`Reader::outstanding`]).
#[derive(Clone)]
pub struct Outstanding(pub(crate) Arc<std::sync::atomic::AtomicUsize>);

impl Outstanding {
    /// How many of the reader's requests have not yet ended.
    pub fn requests(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// A piece a worker of the engine is reading: its file and the volume it
/// lies on, its bytes there, the depth its link holds pieces of its kind
/// to (the due depth for a due piece, the knee for one read ahead), and
/// since when.
#[derive(Clone, Debug)]
pub struct Flight {
    pub path: Arc<Path>,
    pub volume: Volume,
    pub offset: u64,
    pub bytes: u64,
    pub window: u64,
    pub since: Instant,
}

/// The read engine's work as a waiter on a read sees it at one look: the
/// pieces it has ended so far, read or failed, each piece its workers are
/// reading now, and when it last ended a piece on each volume (none for a
/// volume it has ended none on). Whether that is progress is the waiter's
/// to judge, by how long each volume may go ending none of its pieces in
/// flight.
#[derive(Clone, Debug)]
pub struct EngineWork {
    pub ended: u64,
    pub reading: Vec<Flight>,
    pub last_ended: HashMap<Volume, Instant>,
}

/// The process's read engine's work now ([`EngineWork`]): its pieces ended
/// are every volume's requests, and each volume's last end its book's
/// ([`crate::accounting`]). A worker counts its piece ended before it stops
/// reporting it in flight, and the two are read in the other order, so no
/// look misses a piece between them.
pub fn engine_work() -> EngineWork {
    let engine = engine();
    let mut reading: Vec<Flight> = engine
        .flights
        .lock()
        .expect("read workers' flights")
        .iter()
        .filter_map(|flight| flight.lock().expect("a read worker's flight").clone())
        .collect();
    #[cfg(target_os = "linux")]
    if let Backend::Uring(ctl) = &engine.backend {
        ctl.snapshot_flights(&mut reading);
    }
    let queue = engine.queue();
    let ended = queue.links.values().map(|link| link.book.requests()).sum();
    let last_ended = queue.links.iter().filter_map(|(volume, link)| Some((*volume, link.book.last_ended()?))).collect();
    EngineWork { ended, reading, last_ended }
}

/// The counts of `volume` at `now` ([`crate::accounting::totals`]), none
/// before a piece of it was taken.
pub(crate) fn totals(volume: Volume, now: Instant) -> Totals {
    engine().queue().links.get_mut(&volume).map_or_else(Totals::default, |link| link.book.totals_at(now))
}

/// The piece a worker is reading, if any: held only to set, clear or copy
/// it, never across a read.
type InFlight = Arc<Mutex<Option<Flight>>>;

/// A worker's registered slot in [`Engine::flights`]: inserted before the
/// worker starts and removed when it exits (whether retiring or unwinding),
/// so a retiring worker can never race ahead of registration and leak its
/// slot, and an active worker's piece is always visible to [`engine_work`].
struct FlightSlot {
    engine: &'static Engine,
    flight: InFlight,
}

impl FlightSlot {
    fn register(engine: &'static Engine) -> Self {
        let flight = InFlight::default();
        engine.flights.lock().expect("read workers' flights").push(flight.clone());
        Self { engine, flight }
    }

    fn set(&self, value: Option<Flight>) {
        *self.flight.lock().expect("a read worker's flight") = value;
    }
}

impl Drop for FlightSlot {
    fn drop(&mut self) {
        self.engine
            .flights
            .lock()
            .expect("read workers' flights")
            .retain(|other| !Arc::ptr_eq(other, &self.flight));
    }
}

/// Dispatch backend for [`Engine`]: Linux `io_uring` (1 batched ring thread)
/// when supported by the kernel, otherwise the blocking `pread` worker pool.
enum Backend {
    ThreadPool,
    #[cfg(target_os = "linux")]
    Uring(super::uring::UringCtl),
}

/// The process's pool: its queue, the condition its workers wait on, its
/// workers and what its readers and probes need of them ([`Pool`]), each
/// worker's piece in flight, and the buffers pieces that do not land straight
/// are read into.
pub(crate) struct Engine {
    /// Single lock synchronizing queue mutations and [`Self::work`] condvar
    /// notifications so no worker wakeup can be lost between predicate check
    /// and wait.
    queue: Mutex<Queue>,
    pub(crate) work: Condvar,
    /// Taken under the queue's lock where both are held, never the other
    /// way.
    pool: Mutex<Pool>,
    /// Each worker's piece in flight ([`engine_work`]).
    pub(crate) flights: Mutex<Vec<InFlight>>,
    pub(crate) buffers: Buffers,
    backend: Backend,
}

/// Whether the process's read engine is currently using Linux `io_uring`.
#[doc(hidden)]
pub fn io_uring_active() -> bool {
    #[cfg(target_os = "linux")]
    {
        matches!(engine().backend, Backend::Uring(_))
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// The bytes the process's read engine keeps in buffers between pieces:
/// none once no reader is open ([`Buffers`]).
pub fn read_buffers() -> u64 {
    engine().buffers.bytes()
}

/// A hold on the engine's read buffers for a reader that opens
/// ([`Buffers::reading`]).
pub(crate) fn reading() -> Reading {
    engine().buffers.reading()
}

/// A claim on the process's pool for reads of `volume` ([`Need`]).
pub(crate) fn need(volume: &VolumeReads) -> Arc<Need> {
    engine().need(volume)
}

/// The workers the process's pool holds now. Each reads one piece at a
/// time and, when it reads a request's last piece, runs that request's end
/// before it takes another.
#[doc(hidden)]
pub fn read_workers() -> usize {
    engine().pool.lock().expect("read pool").started
}

/// The pool, its workers started for the reads its readers and probes
/// submit ([`Need`]).
pub(crate) fn engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    let engine = ENGINE.get_or_init(|| Engine::with_plane(Arc::clone(crate::global_plane())));
    static HOOK: OnceLock<crate::BlockingClearedHookGuard> = OnceLock::new();
    let _ = HOOK.get_or_init(|| {
        crate::global_plane().register_blocking_cleared_hook(
            crate::LINK_STORAGE_NVME,
            Arc::new(move || {
                // Acquire & release `engine.queue()` first so no storage worker
                // can slip between checking `queue.take() == None` and entering
                // `engine.work.wait(queue)`. Deadlock-free because the hook only
                // fires when a *different* link clears `Blocking`.
                drop(engine.queue());
                engine.wake_all();
            }),
        )
    });
    engine
}

impl Engine {
    /// A pool with no workers and nothing queued (isolated `IoPlane` for tests).
    #[cfg(test)]
    pub(crate) fn new() -> Self {
        Self::with_plane_and_backend(crate::IoPlane::new_default(), Backend::ThreadPool)
    }

    pub(crate) fn with_plane(plane: Arc<crate::IoPlane>) -> Self {
        #[cfg(target_os = "linux")]
        let backend = match super::uring::UringCtl::try_new() {
            Some(ctl) => Backend::Uring(ctl),
            None => Backend::ThreadPool,
        };
        #[cfg(not(target_os = "linux"))]
        let backend = Backend::ThreadPool;
        Self::with_plane_and_backend(plane, backend)
    }

    fn with_plane_and_backend(plane: Arc<crate::IoPlane>, backend: Backend) -> Self {
        Engine {
            queue: Mutex::new(Queue::with_plane(plane)),
            work: Condvar::new(),
            pool: Mutex::new(Pool::default()),
            flights: Mutex::new(Vec::new()),
            buffers: Buffers::default(),
            backend,
        }
    }

    /// Lock the read queue.
    pub(crate) fn queue(&self) -> MutexGuard<'_, Queue> {
        self.queue.lock().expect("read queue")
    }

    /// End `returned` under `self.queue`'s lock and notify a waiting worker
    /// while holding the lock so no wakeup can be lost between a worker's
    /// empty-queue check and `work.wait(queue)`. Drops the returned
    /// `Arc<Need>` strictly outside `self.queue`'s lock so `Need::drop` can
    /// never re-enter `self.queue`.
    pub(crate) fn read_returned(&self, returned: Returned) {
        let need = {
            let mut queue = self.queue();
            let (wake, need) = queue.ended(returned);
            if wake {
                self.work.notify_one();
            }
            need
        };
        drop(need);
    }

    /// End `returned` from the `io_uring` ring thread (no condvar notify needed
    /// because the ring thread loops directly back to `queue.take()`).
    #[cfg(target_os = "linux")]
    pub(crate) fn read_returned_uring(&self, returned: Returned) {
        let need = {
            let mut queue = self.queue();
            let (_, need) = queue.ended(returned);
            need
        };
        drop(need);
    }

    /// Wake `count` workers (or signal the `io_uring` ring thread once).
    pub(crate) fn wake(&self, count: usize) {
        if count == 0 {
            return;
        }
        match &self.backend {
            Backend::ThreadPool => {
                for _ in 0..count {
                    self.work.notify_one();
                }
            }
            #[cfg(target_os = "linux")]
            Backend::Uring(ctl) if !ctl.is_failed() => ctl.wake(),
            #[cfg(target_os = "linux")]
            Backend::Uring(_) => {
                for _ in 0..count {
                    self.work.notify_one();
                }
            }
        }
    }

    /// Wake all idle workers (or signal the `io_uring` ring thread).
    pub(crate) fn wake_all(&self) {
        match &self.backend {
            Backend::ThreadPool => self.work.notify_all(),
            #[cfg(target_os = "linux")]
            Backend::Uring(ctl) if !ctl.is_failed() => ctl.wake(),
            #[cfg(target_os = "linux")]
            Backend::Uring(_) => self.work.notify_all(),
        }
    }

    /// Request async cancellation of any in-flight pieces of `request_id`
    /// (effective on Linux `io_uring`; no-op on the blocking `pread` pool).
    pub(crate) fn cancel_inflight(&self, request_id: u64) {
        match &self.backend {
            Backend::ThreadPool => {
                let _ = request_id;
            }
            #[cfg(target_os = "linux")]
            Backend::Uring(ctl) if !ctl.is_failed() => ctl.cancel_inflight(request_id),
            #[cfg(target_os = "linux")]
            Backend::Uring(_) => {
                let _ = request_id;
            }
        }
    }

    /// A claim on the pool for reads of `volume`, which holds no worker
    /// until reads are submitted under it ([`Need::submitted`]).
    pub(crate) fn need(&'static self, volume: &VolumeReads) -> Arc<Need> {
        let live = Arc::new(AtomicUsize::new(0));
        let (id, generation, budget) = {
            let mut pool = self.pool.lock().expect("read pool");
            let id = pool.claim(volume, live.clone());
            let (generation, budget) = pool.budget_update();
            (id, generation, budget)
        };
        self.buffers.set_budget(generation, budget);
        Arc::new(Need { engine: self, id, live })
    }

    /// Start the workers the pool counted out for its target under its
    /// lock (`pool`, released here), each past it ([`Pool::starting`]).
    fn grow(&'static self, mut pool: MutexGuard<'_, Pool>) -> Result<(), Error> {
        match &self.backend {
            #[cfg(target_os = "linux")]
            Backend::Uring(ctl) if !ctl.is_failed() => {
                if pool.started == 0 && pool.target() > 0 {
                    let ring = ctl.open_ring().map_err(Error::ThreadStart)?;
                    pool.started = 1;
                    pool.named += 1;
                    let name = pool.named;
                    drop(pool);
                    let started = seismic_threads::spawn(format!("seismic-uring-{name}"), move || {
                        super::uring::run_ring(self, ctl, ring);
                    });
                    if let Err(error) = started {
                        self.pool.lock().expect("read pool").started = 0;
                        return Err(Error::ThreadStart(error));
                    }
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Uring(_) => self.grow_thread_pool(pool),
            Backend::ThreadPool => self.grow_thread_pool(pool),
        }
    }

    #[cfg(target_os = "linux")]
    pub(crate) fn fallback_from_uring(&'static self) {
        let mut pool = self.pool.lock().expect("read pool");
        pool.started = 0;
        if pool.target() > 0 {
            let _ = self.grow_thread_pool(pool);
        }
    }

    fn grow_thread_pool(&'static self, mut pool: MutexGuard<'_, Pool>) -> Result<(), Error> {
        let names = pool.starting();
        drop(pool);
        for (index, name) in names.clone().enumerate() {
            let slot = FlightSlot::register(self);
            let started = seismic_threads::spawn(format!("seismic-read-{name}"), move || {
                crate::waited_on();
                work(self, slot)
            });
            if let Err(error) = started {
                self.pool.lock().expect("read pool").started -= names.len() - index;
                return Err(Error::ThreadStart(error));
            }
        }
        Ok(())
    }

    /// Retire the single `io_uring` ring thread once all claims have ended and
    /// no pieces remain queued or in flight. Caller holds `self.queue`'s lock.
    #[cfg(target_os = "linux")]
    pub(crate) fn retire_uring(&self, _queue: &Queue) -> bool {
        let mut pool = self.pool.lock().expect("read pool");
        if pool.target() == 0 {
            pool.started = 0;
            true
        } else {
            false
        }
    }

    /// Whether a worker that found nothing to take leaves the pool: while
    /// it holds more workers than its target. The worker counted out
    /// leaves at once.
    fn retire(&self) -> bool {
        let mut pool = self.pool.lock().expect("read pool");
        let leaves = pool.started > pool.target();
        if leaves {
            pool.started -= 1;
        }
        leaves
    }
}

/// The workers the pool holds, the claims on them and the workers named
/// so far.
///
/// A worker reads one piece at a time, its read blocking, so the pool holds
/// a worker for every piece a device's links can have in flight at once,
/// up to the device's queue depth ([`Device`]). A worker waits for work
/// while its device has a live claim; only a device's last claim ending
/// shrinks the target, and only a shrunk target retires a worker.
#[derive(Default)]
struct Pool {
    started: usize,
    named: usize,
    claims: u64,
    budget_gen: u64,
    needs: HashMap<u64, Claim>,
    devices: HashMap<Volume, Device>,
}

/// One claim on the pool ([`Need`]): its volume (a device), its own device
/// shape and peak piece count, and its pieces queued or in flight, shared
/// with its need.
struct Claim {
    volume: Volume,
    device: Device,
    live: Arc<AtomicUsize>,
}

/// What the pool holds for one device while any claim on it lives: its
/// links' deepest due depth and knee, the smallest piece its claims read
/// in (their read sizes, or any smaller piece submitted), its queue depth
/// in pieces (each link's due depth and knee in the read sizes it was
/// measured with, and one piece past them, the deepest of its claims') and
/// the most pieces its claims have had queued or in flight at once.
#[derive(Clone, Copy)]
struct Device {
    due: u64,
    knee: u64,
    smallest: u64,
    depth: u64,
    most: usize,
}

/// Per-volume worker ceiling: Apple Silicon NVMe saturates well below 16
/// concurrent threads, and Linux/Windows NVMe saturates well below 32
/// concurrent threads; spawning 65–256 blocking threads oversubscribes
/// runtime cores and stalls decode.
#[cfg(target_os = "macos")]
const MAX_WORKERS: usize = 16;
#[cfg(not(target_os = "macos"))]
const MAX_WORKERS: usize = 32;

impl Device {
    /// A device read by a link of `shape`, nothing yet submitted.
    fn of(shape: &crate::Shape) -> Self {
        Device {
            due: shape.due,
            knee: shape.knee,
            smallest: shape.due_read.min(shape.ahead_read),
            depth: shape.queue_depth(),
            most: 0,
        }
    }

    /// Fold another claim's bounds into this device's aggregate bounds.
    fn merge(&mut self, other: &Device) {
        self.due = self.due.max(other.due);
        self.knee = self.knee.max(other.knee);
        self.smallest = self.smallest.min(other.smallest);
        self.depth = self.depth.max(other.depth);
    }

    /// The pieces its links can hold in flight at once: their due pieces
    /// join the due bytes in flight up to the due depth and their pieces
    /// read ahead join those up to the knee, each kind's first past it
    /// where none of its kind is in flight, so at most the due depth and
    /// knee in its smallest piece, and one piece more; clamped to its queue
    /// depth; and never more pieces than its claims have had queued or in
    /// flight at once, which holds steady while they read as they did.
    /// Capped per volume to `MAX_WORKERS` (16 on macOS, 32 on Linux/Windows).
    fn workers(&self) -> usize {
        let admitted = u128::from(self.due.div_ceil(self.smallest)) + u128::from(self.knee.div_ceil(self.smallest)) + 1;
        let admitted = admitted.min(u128::from(self.depth));
        let workers = usize::try_from(admitted).map_or(self.most, |admitted| admitted.min(self.most));
        workers.min(MAX_WORKERS)
    }
}

impl Pool {
    /// A claim on `volume`'s device, its pieces counted in `live`; the
    /// device takes the deepest of its claims' links.
    fn claim(&mut self, volume: &VolumeReads, live: Arc<AtomicUsize>) -> u64 {
        self.claims += 1;
        self.budget_gen += 1;
        let id = self.claims;
        let claimed = Device::of(&volume.shape);
        self.devices
            .entry(volume.id)
            .and_modify(|device| device.merge(&claimed))
            .or_insert(claimed);
        self.needs.insert(id, Claim { volume: volume.id, device: claimed, live });
        id
    }

    /// Claim `id` submitted pieces, the smallest `smallest` bytes: its
    /// device's most pieces queued or in flight at once, over its claims.
    fn submitted(&mut self, id: u64, smallest: Option<u64>) {
        let claim = self.needs.get_mut(&id).expect("a live need's claim");
        let volume = claim.volume;
        let claim_live = claim.live.load(Ordering::Acquire);
        claim.device.most = claim.device.most.max(claim_live);
        if let Some(smallest) = smallest {
            claim.device.smallest = claim.device.smallest.min(smallest);
        }
        let live = self
            .needs
            .values()
            .filter(|claim| claim.volume == volume)
            .map(|claim| claim.live.load(Ordering::Acquire))
            .fold(0, usize::saturating_add);
        let device = self.devices.get_mut(&volume).expect("a claimed device");
        device.most = device.most.max(live);
        if let Some(smallest) = smallest {
            device.smallest = device.smallest.min(smallest);
        }
    }

    /// Claim `id` ended: recompute its device from the remaining claims on
    /// that volume (or remove it if none remain). Returns whether `target()`
    /// shrank.
    fn release(&mut self, id: u64) -> bool {
        let before = self.target();
        let volume = self.needs.remove(&id).expect("a live need's claim").volume;
        let old_most = self.devices.remove(&volume).expect("a claimed device").most;
        let mut remaining = self.needs.values().filter(|claim| claim.volume == volume);
        if let Some(first) = remaining.next() {
            let mut merged = first.device;
            let mut sum_most = first.device.most;
            for claim in remaining {
                merged.merge(&claim.device);
                sum_most = sum_most.saturating_add(claim.device.most);
            }
            merged.most = sum_most.min(old_most);
            self.devices.insert(volume, merged);
        }
        self.budget_gen += 1;
        self.target() < before
    }

    /// Monotonic generation and current retained-buffer ceiling paired under
    /// `Engine::pool` so out-of-order `Buffers::set_budget` calls discard
    /// stale ceilings.
    fn budget_update(&self) -> (u64, u64) {
        (self.budget_gen, self.buffer_budget())
    }

    /// The workers the devices' links can hold in flight at once
    /// ([`Device::workers`]), never more than [`Pool::cap`].
    fn target(&self) -> usize {
        self.devices.values().map(Device::workers).sum()
    }

    /// The devices' queue depths, summed: the most workers the pool may hold.
    fn cap(&self) -> usize {
        self.devices.values().map(|device| usize::try_from(device.depth).unwrap_or(usize::MAX)).fold(0, usize::saturating_add)
    }

    /// Total bytes in flight the active devices' links can hold (`due + knee`
    /// summed across claimed volumes): the ceiling for retained bounce buffers.
    fn buffer_budget(&self) -> u64 {
        self.devices
            .values()
            .map(|device| device.due.saturating_add(device.knee))
            .fold(0u64, u64::saturating_add)
    }

    /// The names of the workers to start for the target, counted started
    /// now. A worker past the queue depths is an invariant violation.
    fn starting(&mut self) -> std::ops::Range<usize> {
        let (target, cap) = (self.target(), self.cap());
        assert!(target <= cap, "the read pool's target {target} exceeds its devices' queue depths {cap}");
        let starting = target.saturating_sub(self.started);
        if starting > 0 {
            self.started += starting;
            assert!(self.started <= cap, "{} read workers exceed their devices' queue depths {cap}", self.started);
        }
        let first = self.named;
        self.named += starting;
        first..self.named
    }
}

/// A reader's or probe's claim on the pool: workers for the pieces it
/// submits, as many as its volume's link can hold of them in flight
/// ([`Pool::target`]), held while it lives and every request it made
/// lives. Dropped, the workers it held beyond what the others claim leave
/// as they find nothing to take.
pub(crate) struct Need {
    engine: &'static Engine,
    id: u64,
    /// Its pieces queued or in flight, which its device's count reads.
    live: Arc<AtomicUsize>,
}

impl std::fmt::Debug for Need {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("Need").field(&self.id).finish()
    }
}

impl Need {
    /// `pieces` are to be queued under it: the pool holds the workers they
    /// can need before any is queued.
    pub(crate) fn submitted(&self, pieces: &[Piece]) -> Result<(), Error> {
        let mut pool = self.engine.pool.lock().expect("read pool");
        self.live.fetch_add(pieces.len(), Ordering::AcqRel);
        pool.submitted(self.id, pieces.iter().map(|piece| piece.end - piece.start).min());
        let result = self.engine.grow(pool);
        if result.is_err() {
            self.live.fetch_sub(pieces.len(), Ordering::AcqRel);
        }
        result
    }

    /// `count` of its pieces ended: each once, where the queue counts it
    /// ended ([`Queue::ended`]) or cancelled ([`Queue::cancel`]).
    pub(crate) fn ended(&self, count: usize) {
        let live = self.live.fetch_sub(count, Ordering::AcqRel);
        assert!(live >= count, "{count} pieces ended of a claim with {live} queued or in flight");
    }
}

impl Drop for Need {
    fn drop(&mut self) {
        let (shrunk, generation, budget) = {
            let mut pool = self.engine.pool.lock().expect("read pool");
            let shrunk = pool.release(self.id);
            let (generation, budget) = pool.budget_update();
            (shrunk, generation, budget)
        };
        self.engine.buffers.set_budget(generation, budget);
        // Taking the queue's lock first, every idle worker is waiting on
        // the condition, or will see the target as it now is.
        if shrunk {
            drop(self.engine.queue());
            self.engine.wake_all();
        }
    }
}

/// A worker: the most urgent piece a link may start, read, until the
/// process ends. One that takes a piece wakes another while a link may
/// start more, so what a change frees fills one worker after another.
/// One that finds nothing to take while the pool holds more workers than
/// its claims need leaves it.
fn work(engine: &'static Engine, slot: FlightSlot) {
    // Each worker reuses its own aligned bounce buffer across back-to-back
    // pieces (zero `Buffers` mutex contention on the hot path) and returns it
    // to `engine.buffers` whenever the queue goes idle or the worker retires,
    // so no buffer outlives the last open reader.
    let mut scratch: Option<Buffer> = None;
    loop {
        let taken = {
            let mut queue = engine.queue();
            loop {
                if let Some(taken) = queue.take() {
                    if queue.ready() {
                        engine.work.notify_one();
                    }
                    debug_assert!(
                        queue.taken() <= engine.pool.lock().expect("read pool").target(),
                        "a piece in flight beyond the pool's workers: a read the line admits would wait for a thread"
                    );
                    break taken;
                }
                if let Some(buffer) = scratch.take() {
                    engine.buffers.give(buffer);
                }
                if engine.retire() {
                    return;
                }
                queue = engine.work.wait(queue).expect("read queue");
            }
        };
        let piece = &taken.piece;
        slot.set(Some(Flight {
            path: taken.file.path.clone(),
            volume: taken.volume.id,
            offset: piece.start,
            bytes: piece.end - piece.start,
            window: if taken.standing.due() { taken.volume.shape.due } else { taken.volume.shape.knee },
            since: Instant::now(),
        }));
        // A panic is the request's, ending as one on its submitter would:
        // where that unwinds, its observer resumes it and the worker serves
        // on. A piece serving several parts that fails may have failed in bytes
        // only some of them need, or in a gap between them none needs: each
        // part's own span is read again, and only those that fail fail.
        let file = &taken.file;
        let read = taken.state.submitter.run(|| match file.fetch(piece, &mut scratch, &engine.buffers) {
            Ok(fetched) => Ok(fetched),
            Err(error) if piece.parts.len() == 1 => Err(vec![(piece.parts[0].range, Arc::new(error))]),
            Err(_) => Err(piece
                .parts_alone()
                .filter_map(|alone| {
                    let range = alone.parts[0].range;
                    file.read_piece(&alone, &mut scratch, &engine.buffers).err().map(|error| (range, Arc::new(error)))
                })
                .collect()),
        });
        // The piece leaves its link's flight as its read from the volume
        // returns, before its bytes are copied out of a buffer and before its
        // request's end runs the request's hook (a landing), neither of which
        // reads: the depths and the book count reads alone, at the instant
        // the read returned, and the room it frees admits the next piece
        // while this one lands. Where another holds the queue's lock the
        // piece is left for the lock's next holder to end, and lands without
        // waiting for it. It is counted ended, or left to be, before it stops
        // being reported in flight ([`engine_work`]).
        let now = Instant::now();
        crate::timeline::note_at(now, crate::timeline::Kind::Read, 0, (piece.start as i64, (piece.end - piece.start) as i64, 0));
        let direct = matches!(read, Ok(Ok(crate::disk::Fetched::Landed)));
        engine.read_returned(taken.returned(now, direct));
        slot.set(None);
        let read = read.and_then(|read| {
            taken.state.submitter.run(|| match read {
                Ok(fetched) => {
                    fetched.land(piece, &mut scratch);
                    Vec::new()
                }
                Err(failed) => failed,
            })
        });
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match read {
            Ok(failed) => taken.state.pieces_ended(1, failed, None),
            Err(panic) => taken.state.pieces_ended(1, [], Some(panic)),
        }));
    }
}

/// The buffers pieces that do not land straight are read into, kept for
/// the pieces after them while any reader is open ([`Self::reading`]), each
/// the least power of two of [`LEAST_READ`]s that holds its piece and
/// reused only for a piece of that size. Those in use hold at most a link's
/// depths (or one read) of each size, since no more is in flight, so those
/// kept never hold more. How many that is depends on how the pieces
/// happened to overlap, so none outlives the last reader: once it closes,
/// the process holds no read buffer, whatever its reads were.
#[derive(Default)]
pub(crate) struct Buffers(Mutex<Kept>);

/// The buffers kept, by size, the readers open, total retained bytes,
/// checked-out worker scratch bytes, and the active volumes' `due + knee`
/// retained-byte ceiling (`None` when unbounded in standalone unit tests).
#[derive(Default)]
struct Kept {
    readers: usize,
    bytes: u64,
    checked_out: u64,
    budget_gen: u64,
    budget: Option<u64>,
    sizes: HashMap<usize, Vec<Buffer>>,
}

/// One open reader's hold on the buffers ([`Buffers::reading`]): the last
/// one dropped releases every buffer kept.
pub(crate) struct Reading(&'static Buffers);

impl std::fmt::Debug for Reading {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Reading")
    }
}

impl Drop for Reading {
    fn drop(&mut self) {
        // Released past the lock.
        let _released = {
            let mut kept = self.0 .0.lock().expect("read buffers");
            kept.readers -= 1;
            if kept.readers == 0 {
                kept.bytes = 0;
                Some(std::mem::take(&mut kept.sizes))
            } else {
                None
            }
        };
    }
}

impl Buffers {
    /// Hold the buffers for a reader that opens: kept until the last
    /// reader's hold is dropped.
    pub(crate) fn reading(&'static self) -> Reading {
        self.0.lock().expect("read buffers").readers += 1;
        Reading(self)
    }

    /// Update the retained-byte ceiling (derived from active volumes'
    /// `due + knee`) and immediately release any excess buffers outside the
    /// lock, largest size classes first. Updates with a generation at or below
    /// the last applied generation are stale and discarded.
    pub(crate) fn set_budget(&self, generation: u64, budget: u64) {
        let _released = {
            let mut guard = self.0.lock().expect("read buffers");
            let kept = &mut *guard;
            if generation <= kept.budget_gen {
                return;
            }
            kept.budget_gen = generation;
            kept.budget = Some(budget);
            let mut released = Vec::new();
            if kept.bytes > budget {
                let mut sizes: Vec<usize> = kept.sizes.keys().copied().collect();
                sizes.sort_unstable_by(|a, b| b.cmp(a));
                for size in sizes {
                    if kept.bytes <= budget {
                        break;
                    }
                    if let Some(vec) = kept.sizes.get_mut(&size) {
                        while kept.bytes > budget {
                            let Some(buf) = vec.pop() else { break };
                            kept.bytes = kept.bytes.saturating_sub(size as u64);
                            released.push(buf);
                        }
                    }
                    if kept.sizes.get(&size).is_some_and(Vec::is_empty) {
                        kept.sizes.remove(&size);
                    }
                }
            }
            released
        };
    }

    /// Total live bounce buffer bytes: both retained in the pool and currently
    /// checked out by workers (`scratch` / in-flight held buffers).
    fn bytes(&self) -> u64 {
        let kept = self.0.lock().expect("read buffers");
        kept.bytes.saturating_add(kept.checked_out)
    }

    /// A buffer of at least `length` bytes, accounted in `checked_out` until
    /// returned via [`Self::give`].
    pub(crate) fn take(&self, length: usize) -> Buffer {
        let size = length.max(LEAST_READ as usize).next_power_of_two();
        let kept = {
            let mut kept = self.0.lock().expect("read buffers");
            let buf = kept.sizes.get_mut(&size).and_then(Vec::pop);
            if buf.is_some() {
                kept.bytes = kept.bytes.saturating_sub(size as u64);
            }
            kept.checked_out = kept.checked_out.saturating_add(size as u64);
            buf
        };
        kept.unwrap_or_else(|| Buffer::new(size))
    }

    /// Return `buffer` from a worker: decrements `checked_out` and retains it
    /// only while a reader is open and retained bytes stay within the active
    /// `due + knee` budget; excess buffers drop outside the lock.
    pub(crate) fn give(&self, buffer: Buffer) {
        let size = buffer.len() as u64;
        let dropped = {
            let mut kept = self.0.lock().expect("read buffers");
            kept.checked_out = kept.checked_out.saturating_sub(size);
            let fits = kept.readers > 0 && kept.budget.is_none_or(|budget| kept.bytes.saturating_add(size) <= budget);
            if fits {
                kept.bytes = kept.bytes.saturating_add(size);
                kept.sizes.entry(buffer.len()).or_default().push(buffer);
                None
            } else {
                Some(buffer)
            }
        };
        drop(dropped);
    }
}

#[cfg(test)]
mod tests {
    use super::{work, Buffers, Engine, FlightSlot, Pool};
    use crate::disk::{VolumeReads, LEAST_READ};
    use crate::{Shape, Volume};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;

    /// Expected per-device worker count for `decode`: 71 (`64 + 6 + 1`), capped
    /// at `MAX_WORKERS` (16 on macOS, 32 on Linux/Windows).
    const DECODE_WORKERS: usize = super::MAX_WORKERS;

    /// A measured M4 Pro decode link: due 32 MiB in 512 KiB reads, a knee
    /// of 5200 KiB in 2 MiB reads (64 + 6 + 1 pieces with 1 MiB QD cap).
    fn decode(volume: u64) -> VolumeReads {
        VolumeReads {
            id: Volume(volume),
            shape: Shape { due: 32 * MIB, knee: 5200 * KIB, due_read: 512 * KIB, ahead_read: 2 * MIB, cores: 0.25 },
        }
    }

    /// Claim `id` submits `pieces` pieces of `bytes` each, its count of
    /// them queued or in flight `live`; then the workers the pool starts.
    fn submit(pool: &mut Pool, id: u64, live: &Arc<AtomicUsize>, pieces: usize, bytes: u64) -> usize {
        live.fetch_add(pieces, Ordering::AcqRel);
        pool.submitted(id, Some(bytes));
        pool.starting().len()
    }

    /// The pool holds what a device's links can have in flight: its due
    /// depth and knee in its smallest pieces and one more, never more than
    /// its claims have had queued at once; claims on one device share it,
    /// a second device adds its own, and a device whose claims all ended
    /// holds none.
    #[test]
    fn the_pool_holds_what_its_devices_links_can_have_in_flight() {
        let mut pool = Pool::default();
        let (first, second) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let a = pool.claim(&decode(1), first.clone());
        let b = pool.claim(&decode(1), second.clone());
        assert_eq!(submit(&mut pool, a, &first, 1_000, 512 * KIB), DECODE_WORKERS, "the decode link in its own reads");
        assert_eq!(submit(&mut pool, b, &second, 1_000, 512 * KIB), 0, "claims on one device share its workers");
        let other = Arc::new(AtomicUsize::new(0));
        let c = pool.claim(&decode(2), other.clone());
        assert_eq!(submit(&mut pool, c, &other, 3, 512 * KIB), 3, "a device with three pieces queued at once holds three");
        assert!(pool.release(c), "the second device's last claim ending shrinks the target");
        assert_eq!(pool.target(), DECODE_WORKERS);
        assert!(!pool.release(a), "a claim ending beside another on its device shrinks nothing");
        assert!(pool.release(b));
        assert_eq!(pool.target(), 0, "a device whose claims ended holds no workers");
    }

    /// A piece far smaller than the reads its device was measured with
    /// grows the pool to the device's worker cap and no further (the
    /// sizing by the smallest piece alone asked for 8192 + 1300 + 1
    /// workers), and a burst of submits on the device's shards starts no
    /// more workers than that target: a claim that ends while another on
    /// the device lives retires none, and a new one starts none.
    #[test]
    fn a_burst_of_small_pieces_starts_no_more_than_its_devices_queue_depth() {
        let mut pool = Pool::default();
        let shards = (0..4).map(|_| Arc::new(AtomicUsize::new(0))).collect::<Vec<_>>();
        let ids = shards.iter().map(|live| pool.claim(&decode(1), live.clone())).collect::<Vec<_>>();
        let mut started = 0;
        for round in 0..1_000 {
            let shard = round % 4;
            started += submit(&mut pool, ids[shard], &shards[shard], 1, 4 * KIB);
        }
        assert_eq!(pool.cap(), 71, "four shards on one device share its depth");
        assert_eq!(pool.target(), DECODE_WORKERS, "4 KiB pieces hold the device's worker cap");
        assert_eq!(started, DECODE_WORKERS, "the burst started its target's workers once");
        assert!(!pool.release(ids[0]), "a claim ending beside its device's others shrinks nothing");
        let live = Arc::new(AtomicUsize::new(0));
        let next = pool.claim(&decode(1), live.clone());
        assert_eq!(submit(&mut pool, next, &live, 1, 4 * KIB), 0, "a new claim on the device starts none");
        assert_eq!(pool.started, DECODE_WORKERS);
    }

    /// Buffers are kept between pieces while a reader is open, and none
    /// outlives the last: a buffer given back with none open is released.
    #[test]
    fn no_read_buffer_outlives_the_last_reader() {
        let buffers: &'static Buffers = Box::leak(Box::default());
        let (first, second) = (buffers.reading(), buffers.reading());
        buffers.give(buffers.take(1));
        buffers.give(buffers.take(1 << 20));
        let both = LEAST_READ + (1 << 20);
        assert_eq!(buffers.bytes(), both);
        drop(first);
        assert_eq!(buffers.bytes(), both, "a reader is still open");
        drop(second);
        assert_eq!(buffers.bytes(), 0);
        buffers.give(buffers.take(1));
        assert_eq!(buffers.bytes(), 0, "no reader is open");
    }

    /// Releasing a deeper claim on a shared volume shrinks `target()` to what
    /// the remaining smaller claim actually needs rather than staying pinned
    /// at the departed claim's depth.
    #[test]
    fn releasing_a_deeper_claim_shrinks_target_while_smaller_claim_remains() {
        let mut pool = Pool::default();
        let small_volume = VolumeReads {
            id: Volume(1),
            shape: Shape { due: 2 * MIB, knee: 2 * MIB, due_read: 512 * KIB, ahead_read: 2 * MIB, cores: 0.25 },
        };
        let (big_live, small_live) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let big = pool.claim(&decode(1), big_live.clone());
        let small = pool.claim(&small_volume, small_live.clone());
        assert_eq!(submit(&mut pool, big, &big_live, 100, 512 * KIB), DECODE_WORKERS);
        assert_eq!(submit(&mut pool, small, &small_live, 10, 512 * KIB), 0);
        assert_eq!(pool.target(), DECODE_WORKERS);
        assert!(pool.release(big), "dropping the deeper claim must shrink target");
        assert_eq!(pool.target(), 7, "remaining claim needs 4 due + 2 knee + 1 = 7 workers");
        assert!(pool.release(small));
        assert_eq!(pool.target(), 0);
    }

    /// `Device::workers()` caps per-volume workers at `MAX_WORKERS` (16 on
    /// macOS, 32 on Linux/Windows) so large queue depths never oversubscribe
    /// runtime cores with blocking threads.
    #[test]
    fn device_workers_capped_at_platform_max() {
        let mut pool = Pool::default();
        let live = Arc::new(AtomicUsize::new(0));
        let claim = pool.claim(&decode(1), live.clone());
        let started = submit(&mut pool, claim, &live, 500, 256 * KIB);
        assert!(started <= super::MAX_WORKERS, "per-volume workers must be <= {}, got {started}", super::MAX_WORKERS);
        assert_eq!(started, super::MAX_WORKERS);
        assert_eq!(pool.target(), super::MAX_WORKERS);
    }

    /// `Buffers` bounds retained bounce buffers to the active volumes'
    /// `due + knee` budget and immediately drains excess buffers when the
    /// budget shrinks (e.g. after model import completes before bank sizing).
    #[test]
    fn buffers_bounded_by_due_plus_knee_and_trimmed_on_shrink() {
        let buffers: &'static Buffers = Box::leak(Box::default());
        let _reading = buffers.reading();
        let mut pool = Pool::default();
        let live = Arc::new(AtomicUsize::new(0));
        let small = VolumeReads {
            id: Volume(1),
            shape: Shape { due: 2 * MIB, knee: 2 * MIB, due_read: 512 * KIB, ahead_read: 2 * MIB, cores: 0.25 },
        };
        let claim = pool.claim(&small, live);
        let (gen1, budget1) = pool.budget_update();
        assert_eq!(budget1, 4 * MIB);
        buffers.set_budget(gen1, budget1);

        // Returning 5 × 1 MiB concurrent buffers retains at most 4 MiB (due + knee).
        let bufs: Vec<_> = (0..5).map(|_| buffers.take(MIB as usize)).collect();
        for buf in bufs {
            buffers.give(buf);
        }
        assert_eq!(buffers.bytes(), 4 * MIB, "retained bytes must not exceed due + knee");

        // Releasing the claim drops budget to 0 and immediately drains all kept buffers.
        pool.release(claim);
        let (gen2, budget2) = pool.budget_update();
        assert_eq!(budget2, 0);
        buffers.set_budget(gen2, budget2);
        assert_eq!(buffers.bytes(), 0, "excess buffers must be drained when claim releases");
    }

    /// Out-of-order budget publications from concurrent claim/release
    /// operations discard stale generations so an older larger ceiling can
    /// never overwrite a newer smaller ceiling.
    #[test]
    fn stale_budget_publication_is_discarded() {
        let buffers: &'static Buffers = Box::leak(Box::default());
        let _reading = buffers.reading();
        let mut pool = Pool::default();
        let vol_a = VolumeReads {
            id: Volume(1),
            shape: Shape { due: 2 * MIB, knee: 2 * MIB, due_read: 512 * KIB, ahead_read: 2 * MIB, cores: 0.25 },
        };
        let vol_b = VolumeReads {
            id: Volume(2),
            shape: Shape { due: 4 * MIB, knee: 4 * MIB, due_read: 512 * KIB, ahead_read: 2 * MIB, cores: 0.25 },
        };

        let claim_a = pool.claim(&vol_a, Arc::new(AtomicUsize::new(0)));
        let (gen_a, budget_a) = pool.budget_update();
        assert_eq!(budget_a, 4 * MIB);
        buffers.set_budget(gen_a, budget_a);

        // Claim B raises budget to 12 MiB (gen_b), then release B drops it back
        // to 4 MiB (gen_b_rel). Publish gen_b_rel first, then stale gen_b.
        let claim_b = pool.claim(&vol_b, Arc::new(AtomicUsize::new(0)));
        let (gen_b, budget_b) = pool.budget_update();
        assert_eq!(budget_b, 12 * MIB);
        pool.release(claim_b);
        let (gen_b_rel, budget_b_rel) = pool.budget_update();
        assert_eq!(budget_b_rel, 4 * MIB);

        buffers.set_budget(gen_b_rel, budget_b_rel);
        buffers.set_budget(gen_b, budget_b);

        // Returning 6 × 1 MiB buffers must remain capped at 4 MiB, not 12 MiB.
        let bufs: Vec<_> = (0..6).map(|_| buffers.take(MIB as usize)).collect();
        for buf in bufs {
            buffers.give(buf);
        }
        assert_eq!(buffers.bytes(), 4 * MIB, "stale larger budget must not overwrite newer smaller ceiling");

        // Release A drops to 0; publishing stale gen_a afterwards keeps 0.
        pool.release(claim_a);
        let (gen_zero, budget_zero) = pool.budget_update();
        assert_eq!(budget_zero, 0);
        buffers.set_budget(gen_zero, budget_zero);
        buffers.set_budget(gen_a, budget_a);
        assert_eq!(buffers.bytes(), 0);
        let buf = buffers.take(MIB as usize);
        buffers.give(buf);
        assert_eq!(buffers.bytes(), 0, "stale non-zero budget must not reopen retention after zero");
    }

    /// A worker's flight slot is registered before the worker begins and
    /// removed when it retires (or if dropped before running, as on a failed
    /// spawn), so a retiring worker can never leak its slot into `flights`.
    #[test]
    fn flight_slot_registers_before_work_and_deregisters_on_retire_or_drop() {
        let engine: &'static Engine = Box::leak(Box::new(Engine::new()));
        // Direct drop (e.g. failed spawn) removes the slot.
        let aborted = FlightSlot::register(engine);
        assert_eq!(engine.flights.lock().unwrap().len(), 1);
        drop(aborted);
        assert!(engine.flights.lock().unwrap().is_empty());

        // A worker that retires immediately deregisters its slot cleanly.
        engine.pool.lock().unwrap().started = 1;
        let slot = FlightSlot::register(engine);
        assert_eq!(engine.flights.lock().unwrap().len(), 1);
        std::thread::spawn(move || work(engine, slot)).join().unwrap();
        assert_eq!(engine.pool.lock().unwrap().started, 0);
        assert!(engine.flights.lock().unwrap().is_empty());
    }
}
