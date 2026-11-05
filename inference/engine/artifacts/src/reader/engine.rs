use super::probe::LEAST_READ;
use super::queue::Queue;
use super::volume::VolumeKey;
#[cfg(doc)]
use super::ArtifactReader;
use super::{Panic, ALIGNMENT};
use crate::Error;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::Instant;

/// A reader's jobs submitted and not yet ended ([`ArtifactReader::outstanding`]).
#[derive(Clone)]
pub struct Outstanding(pub(super) Arc<std::sync::atomic::AtomicUsize>);

impl Outstanding {
    /// How many of the reader's jobs have not yet ended.
    pub fn jobs(&self) -> usize {
        self.0.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// A piece a worker of the engine is reading: its file and the device it
/// lies on (`st_dev` on unix, the volume's serial number on Windows), its
/// bytes there, and since when.
#[derive(Clone, Debug)]
pub struct Flight {
    pub path: Arc<Path>,
    pub device: u64,
    pub offset: u64,
    pub bytes: u64,
    pub since: Instant,
}

/// The read engine's work as a waiter on a read sees it at one look: the
/// pieces it has ended so far, read or failed, and each piece its workers
/// are reading now. Whether that is progress is the waiter's to judge, by
/// how long each piece in flight may take.
#[derive(Clone, Debug)]
pub struct EngineWork {
    pub ended: u64,
    pub reading: Vec<Flight>,
}

/// The pieces the process's pool has ended ([`engine_work`]).
static PIECES_ENDED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The process's read engine's work now ([`EngineWork`]). A worker counts
/// its piece ended before it stops reporting it in flight, and the two are
/// read in the other order, so no look misses a piece between them.
pub fn engine_work() -> EngineWork {
    let reading = engine()
        .flights
        .lock()
        .expect("read workers' flights")
        .iter()
        .filter_map(|flight| flight.lock().expect("a read worker's flight").clone())
        .collect();
    let ended = PIECES_ENDED.load(std::sync::atomic::Ordering::Acquire);
    EngineWork { ended, reading }
}

/// The piece a worker is reading, if any: held only to set, clear or copy
/// it, never across a read.
type InFlight = Arc<Mutex<Option<Flight>>>;

/// The process's pool: its queue, the condition its workers wait on, how
/// many it has started, each one's piece in flight, and the buffers pieces
/// that do not land straight are read into.
pub(super) struct Engine {
    pub(super) queue: Mutex<Queue>,
    pub(super) work: Condvar,
    pub(super) workers: Mutex<usize>,
    /// Each worker's piece in flight ([`engine_work`]).
    pub(super) flights: Mutex<Vec<InFlight>>,
    pub(super) buffers: Buffers,
}

/// The pool, its workers started by the readers that need them
/// ([`Engine::deepen`]).
pub(super) fn engine() -> &'static Engine {
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
    pub(super) fn deepen(&'static self, count: usize) -> Result<(), Error> {
        let mut workers = self.workers.lock().expect("read workers");
        while *workers < count {
            let flight = InFlight::default();
            self.flights.lock().expect("read workers' flights").push(flight.clone());
            std::thread::Builder::new()
                .name(format!("artifact-read-{workers}"))
                .spawn(move || work(self, &flight))
                .map_err(Error::ThreadStart)?;
            *workers += 1;
        }
        Ok(())
    }
}

/// A worker: the most urgent piece it may take, read, until the process
/// ends.
fn work(engine: &'static Engine, flight: &Mutex<Option<Flight>>) {
    let mut read: Option<(VolumeKey, u64)> = None;
    loop {
        let (state, file, volume, piece) = {
            let mut queue = engine.queue.lock().expect("read queue");
            // An ended piece frees room a held-back piece may be waiting
            // for, where this worker takes another volume's piece.
            if let Some((volume, bytes)) = read.take() {
                let held_back = queue.ended(volume, bytes, Instant::now());
                if held_back {
                    engine.work.notify_one();
                }
            }
            loop {
                if let Some(taken) = queue.take() {
                    break taken;
                }
                queue = engine.work.wait(queue).expect("read queue");
            }
        };
        read = Some((volume.key, piece.end - piece.start));
        *flight.lock().expect("a read worker's flight") =
            Some(Flight {
                path: file.path.clone(),
                device: volume.key.0,
                offset: piece.start,
                bytes: piece.end - piece.start,
                since: Instant::now(),
            });
        // A panic is the job's, whose observer resumes it: the worker serves
        // on. A piece serving several parts that fails may have failed in bytes
        // only some of them need, or in a gap between them none needs: each
        // part's own span is read again, and only those that fail fail.
        match std::panic::catch_unwind(AssertUnwindSafe(|| match file.read_piece(&piece, &engine.buffers) {
            Ok(()) => Vec::new(),
            Err(error) if piece.parts.len() == 1 => vec![(piece.parts[0].range, Arc::new(error))],
            Err(_) => piece
                .parts_alone()
                .filter_map(|alone| {
                    let range = alone.parts[0].range;
                    file.read_piece(&alone, &engine.buffers).err().map(|error| (range, Arc::new(error)))
                })
                .collect(),
        })) {
            Ok(failed) => state.pieces_ended(1, failed, None),
            Err(payload) => state.pieces_ended(1, [], Some(Panic::new(payload))),
        }
        PIECES_ENDED.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        *flight.lock().expect("a read worker's flight") = None;
    }
}

/// A heap buffer whose first byte is [`ALIGNMENT`]-aligned.
pub(super) struct Aligned {
    storage: Vec<u8>,
    first: usize,
    length: usize,
}

impl Aligned {
    pub(super) fn new(length: usize) -> Self {
        let storage = vec![0u8; length + ALIGNMENT as usize];
        let first = storage.as_ptr().align_offset(ALIGNMENT as usize);
        Self {
            storage,
            first,
            length,
        }
    }

    pub(super) fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.storage[self.first..self.first + self.length]
    }
}

/// The buffers pieces that do not land straight are read into, kept for
/// the pieces after them, each the least power of two of [`LEAST_READ`]s
/// that holds its piece and reused only for a piece of that size. Those in
/// use hold at most a volume's window (or one read) of each size, since no
/// more is in flight, so those kept never hold more.
#[derive(Default)]
pub(super) struct Buffers(Mutex<HashMap<usize, Vec<Aligned>>>);

impl Buffers {
    /// A buffer of at least `length` bytes.
    pub(super) fn take(&self, length: usize) -> Aligned {
        let size = length.max(LEAST_READ as usize).next_power_of_two();
        let kept = self.0.lock().expect("read buffers").get_mut(&size).and_then(Vec::pop);
        kept.unwrap_or_else(|| Aligned::new(size))
    }

    /// Keep `buffer` for a piece after.
    pub(super) fn give(&self, buffer: Aligned) {
        self.0.lock().expect("read buffers").entry(buffer.length).or_default().push(buffer);
    }
}
