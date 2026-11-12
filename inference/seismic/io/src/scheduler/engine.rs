use super::queue::{Piece, Queue, Returned};
use super::Panic;
use crate::disk::{VolumeReads, LEAST_READ};
#[cfg(doc)]
use crate::Reader;
use crate::{Buffer, Error, Totals, Volume};
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, TryLockError};
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
    let reading = engine
        .flights
        .lock()
        .expect("read workers' flights")
        .iter()
        .filter_map(|flight| flight.lock().expect("a read worker's flight").clone())
        .collect();
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

/// The process's pool: its queue, the pieces whose reads returned while
/// another held it, the condition its workers wait on, its workers and
/// what its readers and probes need of them ([`Pool`]), each worker's piece
/// in flight, and the buffers pieces that do not land straight are read
/// into.
pub(crate) struct Engine {
    /// Held only through [`Engine::queue`], which ends `returned` first.
    queue: Mutex<Queue>,
    /// Pieces whose reads returned while the queue's lock was held, each
    /// ended by the lock's next holder before it looks at the queue: a
    /// returning read lands without waiting for the lock.
    returned: Mutex<Vec<Returned>>,
    /// How many `returned` holds, set under its lock, so a holder that finds
    /// none takes no lock for them: one left as a holder looked is ended by
    /// its worker's own next look at the queue.
    left: AtomicUsize,
    pub(crate) work: Condvar,
    /// Taken under the queue's lock where both are held, never the other
    /// way.
    pool: Mutex<Pool>,
    /// Each worker's piece in flight ([`engine_work`]).
    pub(crate) flights: Mutex<Vec<InFlight>>,
    pub(crate) buffers: Buffers,
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
    ENGINE.get_or_init(Engine::new)
}

impl Engine {
    /// A pool with no workers and nothing queued.
    pub(crate) fn new() -> Self {
        Engine {
            queue: Mutex::new(Queue::default()),
            returned: Mutex::new(Vec::new()),
            left: AtomicUsize::new(0),
            work: Condvar::new(),
            pool: Mutex::new(Pool::default()),
            flights: Mutex::new(Vec::new()),
            buffers: Buffers::default(),
        }
    }

    /// The queue, every piece whose read returned ended in it first.
    pub(crate) fn queue(&self) -> MutexGuard<'_, Queue> {
        let mut queue = self.queue.lock().expect("read queue");
        self.end_returned(&mut queue);
        queue
    }

    /// End `returned` now if the queue's lock is free, or leave it for the
    /// lock's next holder to end ([`Engine::queue`]).
    pub(crate) fn read_returned(&self, returned: Returned) {
        match self.queue.try_lock() {
            Ok(mut queue) => {
                self.end_returned(&mut queue);
                self.end(&mut queue, returned);
            }
            Err(TryLockError::WouldBlock) => {
                let mut left = self.returned.lock().expect("returned reads");
                left.push(returned);
                self.left.store(left.len(), Ordering::Release);
            }
            Err(TryLockError::Poisoned(_)) => panic!("read queue poisoned"),
        }
    }

    /// End in `queue` the pieces whose reads returned while its lock was
    /// held, in the order they were left.
    fn end_returned(&self, queue: &mut Queue) {
        if self.left.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut returned = self.returned.lock().expect("returned reads");
        self.left.store(0, Ordering::Relaxed);
        for returned in returned.drain(..) {
            self.end(queue, returned);
        }
    }

    /// End `returned` in `queue`: room it frees on a link with a request
    /// queued wakes a worker to take its next piece.
    fn end(&self, queue: &mut Queue, returned: Returned) {
        if queue.ended(returned) {
            self.work.notify_one();
        }
    }

    /// Wake `count` workers.
    pub(crate) fn wake(&self, count: usize) {
        for _ in 0..count {
            self.work.notify_one();
        }
    }

    /// A claim on the pool for reads of `volume`, which holds no worker
    /// until reads are submitted under it ([`Need::submitted`]).
    pub(crate) fn need(&'static self, volume: &VolumeReads) -> Arc<Need> {
        let live = Arc::new(AtomicUsize::new(0));
        let id = self.pool.lock().expect("read pool").claim(volume, live.clone());
        Arc::new(Need { engine: self, id, live })
    }

    /// Start the workers the pool counted out for its target under its
    /// lock (`pool`, released here), each past it ([`Pool::starting`]).
    fn grow(&'static self, mut pool: MutexGuard<'_, Pool>) -> Result<(), Error> {
        let names = pool.starting();
        drop(pool);
        for (index, name) in names.clone().enumerate() {
            let flight = InFlight::default();
            let started = std::thread::Builder::new()
                .name(format!("seismic-read-{name}"))
                .spawn({
                    let flight = flight.clone();
                    move || {
                        crate::waited_on();
                        work(self, &flight)
                    }
                });
            if let Err(error) = started {
                self.pool.lock().expect("read pool").started -= names.len() - index;
                return Err(Error::ThreadStart(error));
            }
            self.flights.lock().expect("read workers' flights").push(flight);
        }
        Ok(())
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
    needs: HashMap<u64, Claim>,
    devices: HashMap<Volume, Device>,
}

/// One claim on the pool ([`Need`]): its volume (a device) and its pieces
/// queued or in flight, shared with its need.
struct Claim {
    volume: Volume,
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

impl Device {
    /// A device read by a link of `shape`, nothing yet submitted.
    fn of(shape: &crate::Shape) -> Self {
        Device {
            due: shape.due,
            knee: shape.knee,
            smallest: shape.due_read.min(shape.ahead_read),
            depth: shape.due.div_ceil(shape.due_read) + shape.knee.div_ceil(shape.ahead_read) + 1,
            most: 0,
        }
    }

    /// The pieces its links can hold in flight at once: their due pieces
    /// join the due bytes in flight up to the due depth and their pieces
    /// read ahead join those up to the knee, each kind's first past it
    /// where none of its kind is in flight, so at most the due depth and
    /// knee in its smallest piece, and one piece more; clamped to its queue
    /// depth; and never more pieces than its claims have had queued or in
    /// flight at once, which holds steady while they read as they did.
    fn workers(&self) -> usize {
        let admitted = u128::from(self.due.div_ceil(self.smallest)) + u128::from(self.knee.div_ceil(self.smallest)) + 1;
        let admitted = admitted.min(u128::from(self.depth));
        usize::try_from(admitted).map_or(self.most, |admitted| admitted.min(self.most))
    }
}

impl Pool {
    /// A claim on `volume`'s device, its pieces counted in `live`; the
    /// device takes the deepest of its claims' links.
    fn claim(&mut self, volume: &VolumeReads, live: Arc<AtomicUsize>) -> u64 {
        self.claims += 1;
        let id = self.claims;
        let claimed = Device::of(&volume.shape);
        self.devices
            .entry(volume.id)
            .and_modify(|device| {
                device.due = device.due.max(claimed.due);
                device.knee = device.knee.max(claimed.knee);
                device.smallest = device.smallest.min(claimed.smallest);
                device.depth = device.depth.max(claimed.depth);
            })
            .or_insert(claimed);
        self.needs.insert(id, Claim { volume: volume.id, live });
        id
    }

    /// Claim `id` submitted pieces, the smallest `smallest` bytes: its
    /// device's most pieces queued or in flight at once, over its claims.
    fn submitted(&mut self, id: u64, smallest: Option<u64>) {
        let volume = self.needs.get(&id).expect("a live need's claim").volume;
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

    /// Claim `id` ended: its device, with its last claim, holds no workers.
    /// Whether the target shrank: only a device's last claim ending, of a
    /// device that held workers, shrinks it.
    fn release(&mut self, id: u64) -> bool {
        let volume = self.needs.remove(&id).expect("a live need's claim").volume;
        let last = !self.needs.values().any(|claim| claim.volume == volume);
        last && self.devices.remove(&volume).is_some_and(|device| device.workers() > 0)
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

    /// The names of the workers to start for the target, counted started
    /// now. A worker past the queue depths is an invariant violation.
    fn starting(&mut self) -> std::ops::Range<usize> {
        let (target, cap) = (self.target(), self.cap());
        assert!(target <= cap, "the read pool's target {target} exceeds its devices' queue depths {cap}");
        let starting = target.saturating_sub(self.started);
        self.started += starting;
        assert!(self.started <= cap, "{} read workers exceed their devices' queue depths {cap}", self.started);
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
        self.engine.grow(pool)
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
        let shrunk = {
            self.engine.pool.lock().expect("read pool").release(self.id)
        };
        // Taking the queue's lock first, every idle worker is waiting on
        // the condition, or will see the target as it now is.
        if shrunk {
            drop(self.engine.queue());
            self.engine.work.notify_all();
        }
    }
}

/// A worker: the most urgent piece a link may start, read, until the
/// process ends. One that takes a piece wakes another while a link may
/// start more, so what a change frees fills one worker after another.
/// One that finds nothing to take while the pool holds more workers than
/// its claims need leaves it.
fn work(engine: &'static Engine, flight: &InFlight) {
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
                if engine.retire() {
                    engine.flights.lock().expect("read workers' flights").retain(|other| !Arc::ptr_eq(other, flight));
                    return;
                }
                queue = engine.work.wait(queue).expect("read queue");
                engine.end_returned(&mut queue);
            }
        };
        let piece = &taken.piece;
        *flight.lock().expect("a read worker's flight") = Some(Flight {
            path: taken.file.path.clone(),
            volume: taken.volume.id,
            offset: piece.start,
            bytes: piece.end - piece.start,
            window: if taken.standing.due() { taken.volume.shape.due } else { taken.volume.shape.knee },
            since: Instant::now(),
        });
        // A panic is the request's, whose observer resumes it: the worker serves
        // on. A piece serving several parts that fails may have failed in bytes
        // only some of them need, or in a gap between them none needs: each
        // part's own span is read again, and only those that fail fail.
        let file = &taken.file;
        let read = std::panic::catch_unwind(AssertUnwindSafe(|| match file.read_piece(piece, &engine.buffers) {
            Ok(()) => Vec::new(),
            Err(error) if piece.parts.len() == 1 => vec![(piece.parts[0].range, Arc::new(error))],
            Err(_) => piece
                .parts_alone()
                .filter_map(|alone| {
                    let range = alone.parts[0].range;
                    file.read_piece(&alone, &engine.buffers).err().map(|error| (range, Arc::new(error)))
                })
                .collect(),
        }));
        // The piece leaves its link's flight as its read returns, before its
        // request's end runs the request's hook (a landing), which reads
        // nothing: the depths and the book count reads alone, at the instant
        // the read returned. Where another holds the queue's lock the piece
        // is left for the lock's next holder to end, and lands without
        // waiting for it. It is counted ended, or left to be, before it stops
        // being reported in flight ([`engine_work`]).
        let now = Instant::now();
        crate::timeline::note_at(now, crate::timeline::Kind::Read, 0, (piece.start as i64, (piece.end - piece.start) as i64, 0));
        engine.read_returned(taken.returned(now));
        *flight.lock().expect("a read worker's flight") = None;
        match read {
            Ok(failed) => taken.state.pieces_ended(1, failed, None),
            Err(payload) => taken.state.pieces_ended(1, [], Some(Panic::new(payload))),
        }
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

/// The buffers kept, by size, and the readers open.
#[derive(Default)]
struct Kept {
    readers: usize,
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
            (kept.readers == 0).then(|| std::mem::take(&mut kept.sizes))
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

    /// The bytes of the buffers kept.
    fn bytes(&self) -> u64 {
        let kept = self.0.lock().expect("read buffers");
        kept.sizes.iter().map(|(size, buffers)| (size * buffers.len()) as u64).sum()
    }

    /// A buffer of at least `length` bytes.
    pub(crate) fn take(&self, length: usize) -> Buffer {
        let size = length.max(LEAST_READ as usize).next_power_of_two();
        let kept = self.0.lock().expect("read buffers").sizes.get_mut(&size).and_then(Vec::pop);
        kept.unwrap_or_else(|| Buffer::new(size))
    }

    /// Keep `buffer` for a piece after, while a reader is open; a piece of
    /// a request that outlived every reader releases its own.
    pub(crate) fn give(&self, mut buffer: Buffer) {
        let mut kept = self.0.lock().expect("read buffers");
        if kept.readers > 0 {
            kept.sizes.entry(buffer.bytes_mut().len()).or_default().push(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Buffers, Pool};
    use crate::disk::{VolumeReads, LEAST_READ};
    use crate::{Shape, Volume};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;

    /// A measured M4 Pro decode link: due 32 MiB in 512 KiB reads, a knee
    /// of 5200 KiB in 2 MiB reads (64 + 3 + 1 pieces).
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
        assert_eq!(submit(&mut pool, a, &first, 1_000, 512 * KIB), 68, "the decode link in its own reads");
        assert_eq!(submit(&mut pool, b, &second, 1_000, 512 * KIB), 0, "claims on one device share its workers");
        let other = Arc::new(AtomicUsize::new(0));
        let c = pool.claim(&decode(2), other.clone());
        assert_eq!(submit(&mut pool, c, &other, 3, 512 * KIB), 3, "a device with three pieces queued at once holds three");
        assert!(pool.release(c), "the second device's last claim ending shrinks the target");
        assert_eq!(pool.target(), 68);
        assert!(!pool.release(a), "a claim ending beside another on its device shrinks nothing");
        assert!(pool.release(b));
        assert_eq!(pool.target(), 0, "a device whose claims ended holds no workers");
    }

    /// A piece far smaller than the reads its device was measured with
    /// grows the pool to the device's queue depth and no further (the
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
        assert_eq!(pool.cap(), 68, "four shards on one device share its depth");
        assert_eq!(pool.target(), 68, "4 KiB pieces hold the device's queue depth");
        assert_eq!(started, 68, "the burst started its target's workers once");
        assert!(!pool.release(ids[0]), "a claim ending beside its device's others shrinks nothing");
        let live = Arc::new(AtomicUsize::new(0));
        let next = pool.claim(&decode(1), live.clone());
        assert_eq!(submit(&mut pool, next, &live, 1, 4 * KIB), 0, "a new claim on the device starts none");
        assert_eq!(pool.started, 68);
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
}
