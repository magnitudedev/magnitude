use super::queue::{Queue, Taken};
use super::Panic;
use crate::disk::LEAST_READ;
#[cfg(doc)]
use crate::Reader;
use crate::{Buffer, Error, Totals, Volume};
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

/// A reader's jobs submitted and not yet ended ([`Reader::outstanding`]).
#[derive(Clone)]
pub struct Outstanding(pub(crate) Arc<std::sync::atomic::AtomicUsize>);

impl Outstanding {
    /// How many of the reader's jobs have not yet ended.
    pub fn jobs(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// A piece a worker of the engine is reading: its file and the volume it
/// lies on, its bytes there, the bytes its volume's window holds in flight
/// for its class, and since when.
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
    let queue = engine.queue.lock().expect("read queue");
    let ended = queue.volumes.values().map(|flight| flight.book.requests()).sum();
    let last_ended = queue.volumes.iter().filter_map(|(volume, flight)| Some((*volume, flight.book.last_ended()?))).collect();
    EngineWork { ended, reading, last_ended }
}

/// The counts of `volume` at `now` ([`crate::accounting::totals`]), none
/// before a piece of it was taken.
pub(crate) fn totals(volume: Volume, now: Instant) -> Totals {
    engine().queue.lock().expect("read queue").volumes.get_mut(&volume).map_or_else(Totals::default, |flight| flight.book.totals_at(now))
}

/// The piece a worker is reading, if any: held only to set, clear or copy
/// it, never across a read.
type InFlight = Arc<Mutex<Option<Flight>>>;

/// The process's pool: its queue, the condition its workers wait on, how
/// many it has started, each one's piece in flight, and the buffers pieces
/// that do not land straight are read into.
pub(crate) struct Engine {
    pub(crate) queue: Mutex<Queue>,
    pub(crate) work: Condvar,
    pub(crate) workers: Mutex<usize>,
    /// Each worker's piece in flight ([`engine_work`]).
    pub(crate) flights: Mutex<Vec<InFlight>>,
    pub(crate) buffers: Buffers,
}

/// Start workers until the pool holds `count` ([`Engine::deepen`]).
pub(crate) fn deepen(count: usize) -> Result<(), Error> {
    engine().deepen(count)
}

/// The pool, its workers started by the readers that need them
/// ([`Engine::deepen`]).
pub(crate) fn engine() -> &'static Engine {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    ENGINE.get_or_init(|| Engine {
        queue: Mutex::new(Queue::default()),
        work: Condvar::new(),
        workers: Mutex::new(0),
        flights: Mutex::new(Vec::new()),
        buffers: Buffers::default(),
    })
}

impl Engine {
    /// Start workers until the pool holds `count`: a reader of a volume
    /// that needs them reads at its full window or is not opened.
    pub(crate) fn deepen(&'static self, count: usize) -> Result<(), Error> {
        let mut workers = self.workers.lock().expect("read workers");
        while *workers < count {
            let flight = InFlight::default();
            self.flights.lock().expect("read workers' flights").push(flight.clone());
            std::thread::Builder::new()
                .name(format!("seismic-read-{workers}"))
                .spawn(move || {
                    crate::waited_on();
                    work(self, &flight)
                })
                .map_err(Error::ThreadStart)?;
            *workers += 1;
        }
        Ok(())
    }
}

/// A worker: the most urgent piece it may take, read, until the process
/// ends.
fn work(engine: &'static Engine, flight: &Mutex<Option<Flight>>) {
    loop {
        let taken = {
            let mut queue = engine.queue.lock().expect("read queue");
            loop {
                if let Some(taken) = queue.take() {
                    break taken;
                }
                queue = engine.work.wait(queue).expect("read queue");
            }
        };
        let piece = &taken.piece;
        *flight.lock().expect("a read worker's flight") = Some(Flight {
            path: taken.file.path.clone(),
            volume: taken.volume.id,
            offset: piece.start,
            bytes: piece.end - piece.start,
            window: taken.volume.window_for(taken.class),
            since: Instant::now(),
        });
        // A panic is the job's, whose observer resumes it: the worker serves
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
        // The piece leaves its volume's flight as its read returns, before
        // its job's end runs the job's hook (a landing), which reads
        // nothing: the window and the book count reads alone. Room it frees
        // for a held-back piece wakes a worker to take it. It is counted
        // ended before it stops being reported in flight ([`engine_work`]).
        let now = Instant::now();
        crate::timeline::note_at(now, crate::timeline::Kind::Read, 0, (piece.start as i64, (piece.end - piece.start) as i64, 0));
        let held_back = engine.queue.lock().expect("read queue").ended(&taken, now);
        *flight.lock().expect("a read worker's flight") = None;
        if held_back {
            engine.work.notify_one();
        }
        match read {
            Ok(failed) => taken.state.pieces_ended(1, failed, None),
            Err(payload) => taken.state.pieces_ended(1, [], Some(Panic::new(payload))),
        }
    }
}

/// The buffers pieces that do not land straight are read into, kept for
/// the pieces after them, each the least power of two of [`LEAST_READ`]s
/// that holds its piece and reused only for a piece of that size. Those in
/// use hold at most a volume's window (or one read) of each size, since no
/// more is in flight, so those kept never hold more.
#[derive(Default)]
pub(crate) struct Buffers(Mutex<HashMap<usize, Vec<Buffer>>>);

impl Buffers {
    /// A buffer of at least `length` bytes.
    pub(crate) fn take(&self, length: usize) -> Buffer {
        let size = length.max(LEAST_READ as usize).next_power_of_two();
        let kept = self.0.lock().expect("read buffers").get_mut(&size).and_then(Vec::pop);
        kept.unwrap_or_else(|| Buffer::new(size))
    }

    /// Keep `buffer` for a piece after.
    pub(crate) fn give(&self, mut buffer: Buffer) {
        self.0.lock().expect("read buffers").entry(buffer.bytes_mut().len()).or_default().push(buffer);
    }
}
