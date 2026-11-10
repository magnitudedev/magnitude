use super::*;
use crate::accounting::{Book, Wait};
#[cfg(unix)]
use crate::disk::drop_pages;
use crate::disk::{full_rate, grid, Request, Shape, VolumeLines, VolumeReads, LEAST_READ, WIDEST_WINDOW};
use crate::scheduler::{pieces, Engine, Part, Piece, Queue, Queued, RequestState, Span, Standing};
use std::collections::HashMap;
#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::path::Path;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A volume's curve is the process's one, shared by every reader of the
/// volume: a test that probes or adopts one on the temporary directory's
/// volume shapes every reader opened there after it. A test that probes
/// or adopts, or reads the bulk it depends on, holds the volume.
static TEMPORARY_VOLUME: Mutex<()> = Mutex::new(());

/// Hold [`TEMPORARY_VOLUME`] for this test.
fn temporary_volume() -> std::sync::MutexGuard<'static, ()> {
    TEMPORARY_VOLUME.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Expected an hour from now: read ahead.
fn ahead() -> Want {
    Want::expected(Instant::now() + Duration::from_secs(3600), 0.)
}

fn temporary(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "magnitude-reader-{name}-{}.bin",
        std::process::id()
    ));
    std::fs::write(&path, bytes).unwrap();
    path
}

/// A due read beside bulk from other readers of its volume (two loads
/// keeping 4 MiB requests queued and an urgent reader keeping its reads
/// ahead queued) waits behind at most the pieces in flight, never behind
/// the bulk queued: within 0.5 s each, on any device a test runs on (a
/// 4 KiB read behind a few MiB in flight). On an RTX 3090 host: p50
/// 0.16 ms, max 0.41; beside 136 busy threads on its 34 cores, max 9.8.
#[test]
fn a_due_read_beside_bulk_waits_only_for_the_pieces_in_flight() {
    let _volume = temporary_volume();
    let path = temporary("demand-beside-bulk", &vec![5u8; 16 << 20]);
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let urgent = Urgent { parts: vec![4096], records: 1 };
    let flood = |reader: Reader, want: fn() -> Want, length: usize| {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let sink = Arc::new(Buffer::new(length));
            let mut at = 0u64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let range = Range { offset: at, length, into: sink.clone(), at: 0 };
                // SAFETY: the sink is this thread's alone, one request at a time, waited for here.
                let ended = unsafe { reader.request(vec![range], want()) }.unwrap().wait();
                assert!(matches!(ended, Ended::Read { ref failed } if failed.is_empty()), "{ended:?}");
                at = (at + length as u64) % (reader.size() - length as u64);
            }
        })
    };
    let floods = [
        flood(Reader::open(&path, CachePolicy::Uncached).unwrap(), || Want::undated(0.), 4 << 20),
        flood(Reader::open(&path, CachePolicy::Uncached).unwrap(), || Want::undated(0.), 4 << 20),
        flood(Reader::open_urgent(&path, CachePolicy::Uncached, &urgent).unwrap(), ahead, 1 << 20),
    ];
    let reader = Reader::open_urgent(&path, CachePolicy::Uncached, &urgent).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));
    let sink = Arc::new(Buffer::new(4096));
    let longest = (0..100u64)
        .map(|sample| {
            let range = Range { offset: (sample * 7919 * 4096) % (reader.size() - 4096), length: 4096, into: sink.clone(), at: 0 };
            let began = Instant::now();
            // SAFETY: the sink is read by one request at a time, waited for here.
            let ended = unsafe { reader.request(vec![range], Want::now()) }.unwrap().wait();
            assert!(matches!(ended, Ended::Read { ref failed } if failed.is_empty()), "{ended:?}");
            began.elapsed().as_secs_f64() * 1e3
        })
        .fold(0., f64::max);
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for thread in floods {
        thread.join().unwrap();
    }
    std::fs::remove_file(path).unwrap();
    assert!(longest < 500., "a due read waited {longest:.1} ms behind bulk");
}

/// Each range of `ranges`, `(offset, length)`, read as one due request
/// into a buffer of its own: its bytes, or the error that kept them
/// from landing.
fn read(source: &Reader, ranges: &[(u64, usize)]) -> Vec<Result<Vec<u8>, String>> {
    let buffers = ranges
        .iter()
        .map(|(_, length)| Arc::new(Buffer::new(*length)))
        .collect::<Vec<_>>();
    let landing = ranges
        .iter()
        .zip(&buffers)
        .map(|((offset, length), buffer)| Range {
            offset: *offset,
            length: *length,
            into: buffer.clone(),
            at: 0,
        })
        .collect();
    let ended = unsafe { source.request(landing, Want::now()) }.unwrap().wait();
    let Ended::Read { failed } = ended else {
        panic!("the request ended {ended:?}")
    };
    let failed = failed.into_iter().collect::<HashMap<_, _>>();
    buffers
        .into_iter()
        .enumerate()
        .map(|(index, buffer)| match failed.get(&index) {
            Some(error) => Err(error.to_string()),
            // SAFETY: the request has ended.
            None => Ok(unsafe { buffer.bytes() }.to_vec()),
        })
        .collect()
}

/// Under either policy every range lands exactly as stored: through a
/// cached reader's worker buffers, and an uncached one's straight reads
/// where they align.
#[test]
fn ranges_read_as_stored_whatever_their_alignment_and_order() {
    // A file whose size is no multiple of the block, nor of a piece.
    let bytes = (0..3 * LEAST_READ as usize + 777)
        .map(|index| (index * 31 % 251) as u8)
        .collect::<Vec<_>>();
    let path = temporary("ranges", &bytes);
    for policy in [CachePolicy::Uncached, CachePolicy::Cached] {
        let source = Reader::open(&path, policy).unwrap();
        assert_eq!(source.size(), bytes.len() as u64);
        let ranges = [
            (5000u64, 90usize),
            (3, 90),
            (4090, 90),
            (bytes.len() as u64 - 90, 90),
            (5000, 90),
            (0, bytes.len()),
            (2 * ALIGNMENT, 0),
            (LEAST_READ - 10, 2 * LEAST_READ as usize + 20),
        ];
        let read = read(&source, &ranges);
        for (index, (offset, length)) in ranges.iter().enumerate() {
            let offset = *offset as usize;
            assert_eq!(
                read[index],
                Ok(bytes[offset..offset + length].to_vec()),
                "{policy:?} range {index}"
            );
        }
    }
    std::fs::remove_file(path).unwrap();
}

/// A read's end is its volume's last, whoever reads next: what a waiter on
/// a read judges the volume's progress by.
#[test]
fn the_engines_work_shows_each_volumes_last_end() {
    let path = temporary("last-end", &vec![4u8; 4 * LEAST_READ as usize]);
    let reader = Reader::open(&path, CachePolicy::Cached).unwrap();
    let began = Instant::now();
    let sink = Arc::new(Buffer::new(LEAST_READ as usize));
    let range = Range { offset: 0, length: LEAST_READ as usize, into: sink, at: 0 };
    // SAFETY: the sink is this request's alone, waited for here.
    let ended = unsafe { reader.request(vec![range], Want::now()) }.unwrap().wait();
    assert!(matches!(ended, Ended::Read { ref failed } if failed.is_empty()), "{ended:?}");
    let last = *engine_work().last_ended.get(&reader.volume()).expect("the volume ended a piece");
    assert!(last >= began && last <= Instant::now(), "the end is the read's");
    std::fs::remove_file(&path).unwrap();
}

/// A request's hook runs once, with how it ended: registered before it
/// ends, on the thread that ends it; registered after, at once. A request
/// of no piece ends at its submission. Each hook is waited for through
/// what it sends, never through time.
#[test]
fn a_requests_hook_runs_once_with_how_it_ended() {
    let path = temporary("hook", &vec![3u8; 4 * LEAST_READ as usize]);
    let reader = Reader::open(&path, CachePolicy::Uncached).unwrap();
    let sink = Arc::new(Buffer::new(3 * LEAST_READ as usize));
    let range = Range { offset: 0, length: 3 * LEAST_READ as usize, into: sink.clone(), at: 0 };
    let (sent, ran) = std::sync::mpsc::channel::<Ended>();
    let job = unsafe { reader.request(vec![range], Want::now()) }.unwrap();
    let send = sent.clone();
    job.on_end(move |ended| send.send(ended.clone()).unwrap());
    assert!(matches!(job.wait(), Ended::Read { .. }));
    assert!(matches!(ran.recv().unwrap(), Ended::Read { failed } if failed.is_empty()));
    let send = sent.clone();
    job.on_end(move |ended| send.send(ended.clone()).unwrap());
    assert!(ran.try_recv().is_ok(), "a hook after the end runs at once");
    let outside = Range { offset: 8 * LEAST_READ, length: 1, into: sink, at: 0 };
    let failed = unsafe { reader.request(vec![outside], Want::now()) }.unwrap();
    failed.on_end(move |ended| sent.send(ended.clone()).unwrap());
    assert!(matches!(ran.try_recv().unwrap(), Ended::Read { failed } if failed.len() == 1), "a request of no piece ended at its submission");
    assert!(ran.try_recv().is_err(), "each hook ran once");
    drop((job, failed));
    std::fs::remove_file(path).unwrap();
}

/// The request's end is set before its hook is announced: whatever a
/// hook wakes finds the request ended. The interleaving that hung the
/// suite (residency woken by a conveyor read's hook, finding the read still
/// in flight and staying asked, with no event left to wake it) is driven
/// by construction: the hook itself asks whether the request runs, the
/// question a woken `Conveyor::quiet` asks through `Ticket::running`. It
/// finds it ended, for a request a worker ends, for one of no piece, and
/// for one cancelled (asked through its reader's requests outstanding, as
/// the handle is being dropped).
#[test]
fn a_hook_finds_its_request_ended() {
    let path = temporary("ended-first", &vec![4u8; 4 * LEAST_READ as usize]);
    let reader = Reader::open(&path, CachePolicy::Uncached).unwrap();
    let sink = Arc::new(Buffer::new(LEAST_READ as usize));
    let (sent, ran) = std::sync::mpsc::channel::<bool>();
    // Whether the request runs, asked from inside its hook.
    let ask = |request: Arc<Handle>, sent: std::sync::mpsc::Sender<bool>| {
        let asked = request.clone();
        request.on_end(move |_| sent.send(asked.wait_until(Instant::now()).is_none()).unwrap());
    };
    let range = |sink: &Arc<Buffer>, offset| Range { offset, length: LEAST_READ as usize, into: sink.clone(), at: 0 };
    // Ended by the worker that reads its piece.
    let read = Arc::new(unsafe { reader.request(vec![range(&sink, 0)], Want::now()) }.unwrap());
    ask(read.clone(), sent.clone());
    assert!(!ran.recv().unwrap(), "a hook saw its read still running");
    // Ended at its submission (no piece in the file).
    let outside = Arc::new(unsafe { reader.request(vec![range(&sink, 8 * LEAST_READ)], Want::now()) }.unwrap());
    ask(outside.clone(), sent.clone());
    assert!(!ran.recv().unwrap());
    // Ended by its cancellation, its hook registered first.
    let late = Arc::new(Buffer::new(LEAST_READ as usize));
    let cancelled = unsafe { reader.request(vec![range(&late, LEAST_READ)], Want::undated(0.)) }.unwrap();
    let outstanding = reader.outstanding();
    cancelled.on_end(move |_| sent.send(outstanding.requests() > 0).unwrap());
    drop(cancelled);
    assert!(!ran.recv().unwrap());
    drop((read, outside));
    std::fs::remove_file(path).unwrap();
}

/// A cached piece lands through its worker's buffer even where it could
/// be read straight into its sink; an uncached one is read straight.
#[test]
fn a_cached_piece_lands_through_the_buffer() {
    let path = temporary("straight", &vec![1u8; 2 * LEAST_READ as usize]);
    let mut sink = Buffer::new(LEAST_READ as usize);
    let piece = Piece {
        start: 0,
        end: LEAST_READ,
        parts: vec![Part {
            range: 0,
            from: 0,
            length: LEAST_READ as usize,
            sink: NonNull::new(sink.bytes_mut().as_mut_ptr()).unwrap(),
        }],
    };
    let lands_straight = |policy| {
        let reader = Reader::open(&path, policy).unwrap();
        reader.file.lands_straight(&piece)
    };
    assert!(lands_straight(CachePolicy::Uncached));
    assert!(!lands_straight(CachePolicy::Cached));
    std::fs::remove_file(path).unwrap();
}

/// Whether every page of `length` bytes at `offset` of the file at
/// `path` is in the file cache.
#[cfg(unix)]
fn resident(path: &Path, offset: u64, length: usize) -> bool {
    let (cached, pages) = cached_pages(path, offset, length as u64).unwrap();
    cached == pages
}

/// A cached read keeps the pages it read in the file cache, from a file
/// written without leaving them there.
#[cfg(unix)]
#[test]
fn a_cached_read_keeps_its_pages() {
    let path = temporary("pages", &vec![7u8; 8 * LEAST_READ as usize]);
    let written = File::open(&path).unwrap();
    written.sync_all().unwrap();
    drop_pages(&written, 0, 8 * LEAST_READ).unwrap();
    let reader = Reader::open(&path, CachePolicy::Cached).unwrap();
    let (offset, length) = (LEAST_READ + 100, 3 * LEAST_READ as usize);
    // Linux drops a synced file's pages.
    #[cfg(target_os = "linux")]
    assert!(!resident(&path, offset, length));
    assert!(read(&reader, &[(offset, length)])[0].is_ok());
    assert!(resident(&path, offset, length));
    std::fs::remove_file(path).unwrap();
}

/// A file the cache holds whole is evicted, and the uncached reads after
/// leave none of it cached: they reach storage, as a probe timing them
/// needs.
#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn an_evicted_file_is_read_from_storage() {
    let length = 8 * LEAST_READ;
    let path = temporary("evicted", &vec![5u8; length as usize]);
    // Written back first: a page not yet on storage cannot be dropped,
    // and a model file's never are.
    File::open(&path).unwrap().sync_all().unwrap();
    std::fs::read(&path).unwrap();
    assert!(resident(&path, 0, length as usize), "a file just written and read is cached");
    let reader = Reader::open(&path, CachePolicy::Uncached).unwrap();
    reader.evict_cached().unwrap();
    assert_eq!(cached_pages(&path, 0, length).unwrap().0, 0);
    assert!(read(&reader, &[(LEAST_READ, 2 * LEAST_READ as usize)])[0].is_ok());
    assert_eq!(cached_pages(&path, 0, length).unwrap().0, 0, "an uncached read caches nothing");
    std::fs::remove_file(path).unwrap();
}

/// A range that cannot be read fails alone: the ranges of other reads
/// are read, whether the range lies outside the file as opened or the
/// file was cut short under the open source.
#[test]
fn a_failed_read_fails_only_the_ranges_it_serves() {
    let bytes = (0..256 * ALIGNMENT as usize)
        .map(|index| (index * 7 % 253) as u8)
        .collect::<Vec<_>>();
    let path = temporary("failed", &bytes);
    let source = Reader::open(&path, CachePolicy::Uncached).unwrap();
    // Cut the file to its first half: reads of the second half end
    // short. The ranges each side lie farther apart than reads join.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(128 * ALIGNMENT)
        .unwrap();
    let ranges = [
        (100u64, 90usize),
        (192 * ALIGNMENT, 90),
        (bytes.len() as u64 - 1, 2),
        (16 * ALIGNMENT, 90),
    ];
    let read = read(&source, &ranges);
    assert_eq!(read[0], Ok(bytes[100..190].to_vec()));
    assert!(read[1].is_err());
    assert!(read[2].is_err());
    let at = 16 * ALIGNMENT as usize;
    assert_eq!(read[3], Ok(bytes[at..at + 90].to_vec()));
    std::fs::remove_file(path).unwrap();
}

/// A request whose piece fails ends with its ranges failed; a request made
/// beside it lands every range.
#[test]
fn a_failed_piece_fails_only_its_request() {
    let bytes = (0..16 * LEAST_READ as usize)
        .map(|index| (index * 13 % 241) as u8)
        .collect::<Vec<_>>();
    let path = temporary("job", &bytes);
    let source = Reader::open(&path, CachePolicy::Uncached).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(8 * LEAST_READ)
        .unwrap();
    let short = Arc::new(Buffer::new(2 * LEAST_READ as usize));
    let whole = Arc::new(Buffer::new(3 * LEAST_READ as usize));
    let range = |offset: u64, sink: &Arc<Buffer>| Range {
        offset,
        length: sink.region().1,
        into: sink.clone(),
        at: 0,
    };
    let (failing, landing) = unsafe {
        (
            source
                .request(vec![range(10 * LEAST_READ, &short)], Want::now())
                .unwrap(),
            source
                .request(vec![range(LEAST_READ + 5, &whole)], Want::now())
                .unwrap(),
        )
    };
    assert!(matches!(failing.wait(), Ended::Read { failed } if failed.len() == 1));
    assert!(matches!(landing.wait(), Ended::Read { failed } if failed.is_empty()));
    drop((failing, landing));
    let at = LEAST_READ as usize + 5;
    // SAFETY: both requests have ended.
    let whole = unsafe { whole.bytes() };
    assert_eq!(whole, &bytes[at..at + whole.len()]);
    std::fs::remove_file(path).unwrap();
}

fn sink() -> NonNull<u8> {
    NonNull::dangling()
}

/// A coalesced piece's parts, each alone, are read over their own aligned
/// bytes within the piece: a failure the piece read can be told apart by
/// part, and one in a gap between them fails neither.
#[test]
fn a_coalesced_pieces_parts_read_alone_over_their_own_blocks() {
    let range = |offset: u64, length: usize| Span { offset, length, sink: sink() };
    let ranges = [range(0, 90), range(ALIGNMENT + 100, 90)];
    let pieces = pieces(&[(0, ranges[0]), (1, ranges[1])], 10_000_000, LEAST_READ);
    assert_eq!(pieces.len(), 1, "near ranges share one piece");
    let alone = pieces[0]
        .parts_alone()
        .map(|piece| (piece.start, piece.end, piece.parts.iter().map(|part| (part.range, part.from, part.length)).collect::<Vec<_>>()))
        .collect::<Vec<_>>();
    assert_eq!(alone, [(0, ALIGNMENT, vec![(0, 0, 90)]), (ALIGNMENT, 2 * ALIGNMENT, vec![(1, 100, 90)])]);
}

/// Near ranges share a read and far ones do not; a read is split into
/// pieces of at most `LEAST_READ`, each serving the parts of the ranges it
/// covers.
#[test]
fn reads_coalesce_then_split_into_pieces() {
    let range = |offset: u64, length: usize| Span {
        offset,
        length,
        sink: sink(),
    };
    let ranges = [
        range(0, 90),
        range(8000, 90),
        range(2_000_000, 90),
        range(1_000_000, 3 * LEAST_READ as usize),
    ];
    let pieces = pieces(&[0, 1, 3, 2].map(|index| (index, ranges[index])), 10_000_000, LEAST_READ);
    let spans = pieces
        .iter()
        .map(|piece| (piece.start, piece.end))
        .collect::<Vec<_>>();
    let start = 1_000_000 - 1_000_000 % ALIGNMENT;
    let end = (1_000_000 + 3 * LEAST_READ).div_ceil(ALIGNMENT) * ALIGNMENT;
    let far = 2_000_000 - 2_000_000 % ALIGNMENT;
    assert_eq!(
        spans,
        [
            (0, 2 * ALIGNMENT),
            (start, start + LEAST_READ),
            (start + LEAST_READ, start + 2 * LEAST_READ),
            (start + 2 * LEAST_READ, start + 3 * LEAST_READ),
            (start + 3 * LEAST_READ, end),
            (far, far + ALIGNMENT),
        ]
    );
    let parts = |piece: &Piece| {
        piece
            .parts
            .iter()
            .map(|part| (part.range, part.from, part.length))
            .collect::<Vec<_>>()
    };
    assert_eq!(parts(&pieces[0]), [(0, 0, 90), (1, 8000, 90)]);
    let skip = (1_000_000 - start) as usize;
    assert_eq!(parts(&pieces[1]), [(3, skip, LEAST_READ as usize - skip)]);
    assert_eq!(parts(&pieces[2]), [(3, 0, LEAST_READ as usize)]);
    assert_eq!(parts(&pieces[4]), [(3, 0, skip)]);
}

/// A link of its own, sharing nothing: due depth and knee `(due, knee)`,
/// its pieces cut at `read`.
fn volume(device: u64, (due, knee): (u64, u64), read: u64) -> VolumeReads {
    VolumeReads { id: Volume(device), shape: Shape { due, knee, ahead_read: read, due_read: read, cores: 0. } }
}

/// A queued request of its own, `pieces` pieces long, on a link of no
/// bound.
fn queued(pieces: usize) -> (Queued, Arc<RequestState>) {
    let state = RequestState::new(pieces, Vec::new(), Arc::default(), Vec::new());
    let path = std::env::current_exe().unwrap();
    let file = Reader::open(path, CachePolicy::Uncached)
        .unwrap()
        .file;
    let volume = volume(0, (u64::MAX, u64::MAX), LEAST_READ);
    let mut request = Queued {
        state: state.clone(),
        file,
        need: crate::scheduler::need(&volume),
        volume,
        pieces: (0..pieces as u64)
            .map(|index| Piece {
                start: index,
                end: index + 1,
                parts: Vec::new(),
            })
            .collect(),
    };
    request.need.submitted(request.pieces.make_contiguous()).unwrap();
    (request, state)
}

/// A queued request on `volume`'s link of pieces `[start, end)`.
fn queued_on(volume: &VolumeReads, spans: &[(u64, u64)]) -> Queued {
    let (mut request, _) = queued(0);
    request.volume = volume.clone();
    request.need = crate::scheduler::need(volume);
    request.pieces = spans.iter().map(|&(start, end)| Piece { start, end, parts: Vec::new() }).collect();
    request.need.submitted(request.pieces.make_contiguous()).unwrap();
    request
}

/// Pieces are taken by want (the earlier deadline, then the greater value
/// of one deadline, an undated request after every dated one), then by
/// submission, then in their request's order; a reprice moves a request
/// ahead, or behind.
#[test]
fn pieces_are_taken_most_urgent_first() {
    let mut queue = Queue::default();
    let mut states = Vec::new();
    let push = |queue: &mut Queue, states: &mut Vec<Arc<RequestState>>, want, pieces| {
        let (request, state) = queued(pieces);
        states.push(state);
        queue.push(request, want).0
    };
    let later = Instant::now() + Duration::from_secs(3600);
    let at = |seconds: u64, value: f64| Want::expected(later + Duration::from_secs(seconds), value);
    // A load's request, though submitted first, is read behind every dated one.
    push(&mut queue, &mut states, Want::undated(0.), 1);
    push(&mut queue, &mut states, at(2, 0.), 1);
    push(&mut queue, &mut states, at(1, 1.), 2);
    push(&mut queue, &mut states, at(1, 5.), 1);
    push(&mut queue, &mut states, Want::now(), 1);
    let last = push(&mut queue, &mut states, at(3, 0.), 1);
    let demoted = push(&mut queue, &mut states, at(0, 0.), 1);
    // Each take as (the request's index in submission, the piece).
    let take = |queue: &mut Queue, states: &[Arc<RequestState>]| {
        let taken = queue.take().unwrap();
        let request = states.iter().position(|known| Arc::ptr_eq(known, &taken.state));
        (request.unwrap(), taken.piece.start)
    };
    assert_eq!(take(&mut queue, &states), (4, 0), "due first");
    queue.reprice(last, Want::now());
    queue.reprice(demoted, Want::undated(0.));
    assert_eq!(take(&mut queue, &states), (5, 0), "repriced to now: due");
    assert_eq!(take(&mut queue, &states), (3, 0), "of one deadline, the greater value");
    assert_eq!(take(&mut queue, &states), (2, 0));
    assert_eq!(take(&mut queue, &states), (2, 1));
    assert_eq!(take(&mut queue, &states), (1, 0));
    assert_eq!(take(&mut queue, &states), (0, 0));
    assert_eq!(take(&mut queue, &states), (6, 0), "repriced undated: behind the earlier undated one");
    assert!(queue.take().is_none());
}

/// A prediction whose instant has passed stays read ahead ("Order and
/// depth"): a step parked on a read submitted after it reads first, and the
/// prediction is then taken as a read ahead, counted against the knee, not
/// the due depth.
#[test]
fn a_passed_expectation_stays_read_ahead() {
    let mut queue = Queue::default();
    let (expected, _) = queued(1);
    queue.push(expected, Want::expected(Instant::now(), 1.));
    let (parked, parked_state) = queued(1);
    queue.push(parked, Want::now());
    let first = queue.take().unwrap();
    assert!(Arc::ptr_eq(&first.state, &parked_state) && first.standing.due(), "the parked read first");
    assert!(!queue.take().unwrap().standing.due(), "a passed expectation parks nothing");
}

/// A cancelled request gives up the pieces no worker took, a started one
/// too, once; and a request with no piece ends at once.
#[test]
fn a_cancel_gives_up_the_pieces_no_worker_took() {
    let mut queue = Queue::default();
    let (started, _) = queue.push(queued(3).0, Want::now());
    let (untouched, _) = queued(2);
    let (untouched, _) = queue.push(untouched, Want::now());
    queue.take().unwrap();
    assert_eq!(queue.cancel(started).unwrap().pieces.len(), 2);
    assert!(queue.cancel(started).is_none());
    assert_eq!(queue.cancel(untouched).unwrap().pieces.len(), 2);
    assert!(queue.cancel(untouched).is_none());
    assert!(queue.take().is_none());
    let (empty, state) = queued(0);
    queue.push(empty, Want::now());
    assert!(matches!(
        state.progress.lock().unwrap().ended,
        Some(Ended::Read { ref failed }) if failed.is_empty()
    ));
}

/// A range past the host's file domain is refused, not wrapped.
#[cfg(unix)]
#[test]
fn cached_pages_refuses_a_range_past_the_file_domain() {
    let path = temporary("domain", &[0u8; 16]);
    for (offset, length) in [(u64::MAX - 8, 16), (u64::MAX - 8, 0), (1 << 63, 1 << 62)] {
        assert!(matches!(cached_pages(&path, offset, length), Err(Error::Invalid(_))));
    }
    std::fs::remove_file(path).unwrap();
}

/// A reader of a source reads the file the source has open, by either
/// policy, after its path names another: never the replacement's bytes.
#[test]
fn a_reader_reads_its_sources_file_after_the_path_is_replaced() {
    let original = vec![3u8; 2 * ALIGNMENT as usize];
    let path = temporary("replaced", &original);
    let source = OpenFile::open(&path).unwrap();
    let replacement = temporary("replacement", &vec![9u8; original.len()]);
    std::fs::rename(&replacement, &path).unwrap();
    for policy in [CachePolicy::Cached, CachePolicy::Uncached] {
        let reader = Reader::of(&source, policy).unwrap();
        assert_eq!(read(&reader, &[(0, original.len())]), [Ok(original.clone())], "{policy:?}");
    }
    std::fs::remove_file(path).unwrap();
}

/// A dropped handle's pieces no worker took never read: the ranges they
/// served, exactly those, end cancelled, and no worker takes them; a piece
/// in flight reads on, and its end ends the request. Driven on an engine
/// of its own with no worker, so which pieces are in flight is the test's:
/// of four pieces, one taken (as a worker takes it), three cancelled.
#[test]
fn a_dropped_handles_unstarted_pieces_never_read() {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    let engine = ENGINE.get_or_init(Engine::new);
    let (mut queued, state) = queued(4);
    let piece = |start: u64, range: usize| Piece {
        start,
        end: start + LEAST_READ,
        parts: vec![Part { range, from: 0, length: LEAST_READ as usize, sink: NonNull::dangling() }],
    };
    queued.pieces = (0..4).map(|at| piece(at * 3 * LEAST_READ, at as usize)).collect();
    let (id, _) = engine.queue().push(queued, Want::undated(0.));
    let handle = Handle { id, ending: crate::scheduler::Ending(state.clone()), engine };
    let taken = engine.queue().take().expect("a piece to take");
    drop(handle);
    assert!(engine.queue().take().is_none(), "a cancelled piece was taken");
    assert!(state.progress.lock().unwrap().ended.is_none(), "the taken piece reads on");
    // The taken piece's end, as its worker ends it.
    taken.state.pieces_ended(1, [], None);
    let Some(Ended::Read { failed }) = state.progress.lock().unwrap().ended.clone() else { panic!("a cancelled request reads on") };
    assert_eq!(failed.iter().map(|(range, _)| *range).collect::<Vec<_>>(), [1, 2, 3], "the ranges that ended cancelled");
    assert!(failed.iter().all(|(_, error)| matches!(**error, Error::Cancelled)));
}

/// A read that returns while another holds the queue's lock lands without
/// waiting for it: the piece is left, and the lock's next holder ends it at
/// the instant its read returned, before it looks at the queue.
#[test]
fn a_read_returning_under_a_held_lock_is_ended_by_its_next_holder() {
    let engine = Engine::new();
    let (mut queued, _state) = queued(1);
    queued.pieces = [Piece { start: 0, end: LEAST_READ, parts: vec![Part { range: 0, from: 0, length: LEAST_READ as usize, sink: NonNull::dangling() }] }].into();
    engine.queue().push(queued, Want::undated(0.));
    let taken = engine.queue().take().expect("a piece to take");
    let returned = taken.returned(Instant::now());
    let held = engine.queue();
    engine.read_returned(returned);
    assert_eq!(held.links[&taken.volume.id].flight.ahead, LEAST_READ, "ended while another held the lock");
    drop(held);
    let queue = engine.queue();
    let link = &queue.links[&taken.volume.id];
    assert_eq!((link.flight.ahead, link.book.requests()), (0, 1), "the next holder ended it");
}

/// A request whose handle drops before it is waited for is cancelled or
/// finishes, and lands nothing past its end: what a worker had taken
/// reads on, whole.
#[test]
fn a_dropped_request_is_cancelled_or_finishes() {
    let bytes = vec![7u8; 4 * LEAST_READ as usize];
    let path = temporary("cancel", &bytes);
    let source = Reader::open(&path, CachePolicy::Uncached).unwrap();
    let sink = Arc::new(Buffer::new(LEAST_READ as usize));
    let range = Range { offset: 0, length: LEAST_READ as usize, into: sink.clone(), at: 0 };
    let request = unsafe { source.request(vec![range], ahead()) }.unwrap();
    let (sent, ended) = std::sync::mpsc::channel();
    request.on_end(move |_| sent.send(()).unwrap());
    drop(request);
    ended.recv().unwrap();
    // Cancelled before a worker took its one piece, or read whole by
    // then.
    // SAFETY: the request has ended.
    let sink = unsafe { sink.bytes() };
    assert!(sink.iter().all(|byte| *byte == 0) || sink.iter().all(|byte| *byte == 7));
    std::fs::remove_file(path).unwrap();
}

/// A request's reader reads by the rule before's figures where only those
/// are adopted for it (whatever curve its volume holds for other requests),
/// and by the curve once it holds the request's lone times.
#[test]
fn a_reader_reads_by_the_earlier_choice_until_the_curve_holds_its_request() {
    let _volume = temporary_volume();
    let path = temporary("adopted", &[5u8; 4096]);
    let source = OpenFile::open(&path).unwrap();
    // A key its own, as a volume's other tests choose for `None`.
    let urgent = Urgent { parts: vec![4096, 8192, 12288], records: 3 };
    let earlier = EarlierChoice { window: 4 << 20, demand: 8 << 20, read: 256 << 10, demand_read: 1 << 20, cores: 1.5 };
    adopt_earlier(&source, Some(&urgent), earlier).unwrap();
    let adopted = Reader::open_urgent(&path, CachePolicy::Uncached, &urgent).unwrap().volume.shape;
    assert_eq!((adopted.due, adopted.knee, adopted.ahead_read, adopted.due_read), (8 << 20, 4 << 20, 256 << 10, 1 << 20));
    assert_eq!(read_cores([&source], &urgent).unwrap(), 1.5);
    let point = |window: u64, rate: f64| Point { read: LEAST_READ, window, rate: rate * 1e9, spread: 0.01e9, cores: 2. };
    let ranges = vec![Curve::range_point(&urgent, 1, 0.5e9, 0.01e9, 1.), Curve::range_point(&urgent, 2, 0.9e9, 0.01e9, 1.)];
    let curve = Curve { points: vec![point(LEAST_READ, 0.4), point(4 * LEAST_READ, 1.5), point(16 * LEAST_READ, 3.)], lone: vec![Lone { read: LEAST_READ, seconds: 1e-3, error: 1e-5 }], ranges };
    adopt_volume(&source, Some(&urgent), &curve).unwrap();
    let probed = Reader::open_urgent(&path, CachePolicy::Uncached, &urgent).unwrap().volume.shape;
    let request = Request { urgent: &urgent, lone: &curve.lone, ranges: &curve.ranges };
    assert_eq!(probed, Shape::of(&curve.points, Some(request)), "the curve, once it holds the request");
    assert_eq!(read_cores([&source], &urgent).unwrap(), 2.);
    std::fs::remove_file(path).unwrap();
}

/// A read ahead cancelled behind a due one ends with what it had in
/// flight, its range failed `Cancelled` unless every piece of it was taken,
/// and the due read ahead of it lands whole.
#[test]
fn a_cancelled_read_ahead_ends_with_what_it_had_in_flight_and_the_due_one_lands() {
    let _volume = temporary_volume();
    let probed = temporary("cancel-knee", &[5u8; 4096]);
    probe_volume(&OpenFile::open(&probed).unwrap(), None).unwrap();
    let knee = Reader::open(&probed, CachePolicy::Uncached).unwrap().knee();
    std::fs::remove_file(probed).unwrap();
    let bytes = vec![5u8; 4 * knee as usize];
    let path = temporary("cancel-ahead", &bytes);
    let source = Reader::open(&path, CachePolicy::Uncached).unwrap();
    let due = Arc::new(Buffer::new(bytes.len()));
    let read_ahead = Arc::new(Buffer::new(2 * LEAST_READ as usize));
    let range = |sink: &Arc<Buffer>| Range { offset: 0, length: sink.region().1, into: sink.clone(), at: 0 };
    // SAFETY: each sink is its request's alone, which ends before it is read.
    let demanded = unsafe { source.request(vec![range(&due)], Want::now()) }.unwrap();
    let ahead = unsafe { source.request(vec![range(&read_ahead)], ahead()) }.unwrap();
    let (sent, ended) = std::sync::mpsc::channel();
    ahead.on_end(move |ended| sent.send(ended.clone()).unwrap());
    drop(ahead);
    match ended.recv().unwrap() {
        // SAFETY: the request has ended.
        Ended::Read { failed } if failed.is_empty() => assert!(unsafe { read_ahead.bytes() }.iter().all(|byte| *byte == 5)),
        Ended::Read { failed } => assert!(failed.len() == 1 && matches!(*failed[0].1, Error::Cancelled)),
        Ended::Panicked(panic) => panic.resume(),
    }
    assert!(matches!(demanded.wait(), Ended::Read { failed } if failed.is_empty()));
    // SAFETY: the request has ended.
    assert!(unsafe { due.bytes() }.iter().all(|byte| *byte == 5));
    std::fs::remove_file(path).unwrap();
}

/// Due reads hold the widest window at the full rate of their line: past
/// the knee a wider one costs a read alone nothing, and serving needs it
/// (GB10's medians: 9.66 GB/s at 4 MiB, then flat through 32); a narrower
/// window short of the rate, or a wider one slower past its noise, is not
/// taken.
#[test]
fn due_reads_hold_the_widest_window_at_the_full_rate() {
    let mib = 1024 * 1024;
    let point = |window: u64, rate: f64| Point { read: LEAST_READ, window, rate: rate * 1e9, spread: 0.05e9, cores: 1. };
    let line = [point(4 * mib, 9.66), point(8 * mib, 10.09), point(16 * mib, 10.12), point(32 * mib, 10.10)];
    assert_eq!(full_rate(&line).window, 32 * mib);
    let falling = [point(8 * mib, 10.09), point(16 * mib, 10.12), point(32 * mib, 9.2)];
    assert_eq!(full_rate(&falling).window, 16 * mib);
}

/// A recorded curve's lines at each depth, GB/s, as (read, window) points
/// with a spread of 2% (a few percent past the knee, "A volume's curve").
fn recorded(lines: &[(u64, &[(u64, f64)])]) -> Vec<Point> {
    lines
        .iter()
        .flat_map(|&(read, line)| line.iter().map(move |&(depth, rate)| Point { read, window: depth * read, rate: rate * 1e9, spread: 0.02 * rate * 1e9, cores: 1. }))
        .collect()
}

/// The link's depths and read sizes from recorded curves (uncached reads at
/// random offsets of Qwen3.6 35B's file, by depth): the M4 Max's internal
/// drive reaches its best (6.73 GB/s) only 64 deep in 256 KiB reads and 128
/// deep in 128 KiB ones, 16 MiB either way, so it reads ahead 16 MiB deep in
/// the wider. The M4 Pro's flattens at 4.02 GB/s 8 deep in 256 KiB reads,
/// 128 KiB ones never reach it: 2 MiB ahead. GB10's 128 KiB line reaches
/// 10.09 GB/s at 8 MiB and is flat after: with no request it reads ahead
/// 8 MiB deep; a request's knee is its range line's, and its due depth the
/// 128 KiB line's widest, 32 MiB.
#[test]
fn a_links_depths_come_from_its_curve() {
    let mib = 1024 * 1024;
    let max = recorded(&[
        (LEAST_READ, &[(1, 0.71), (2, 1.37), (4, 2.41), (8, 3.70), (16, 5.11), (32, 5.99), (64, 6.38), (128, 6.58)]),
        (2 * LEAST_READ, &[(1, 1.12), (2, 2.01), (4, 3.30), (8, 4.70), (16, 5.86), (32, 6.41), (64, 6.68), (128, 6.73)]),
    ]);
    let shape = Shape::of(&max, None);
    assert_eq!((shape.ahead_read, shape.knee, shape.due_read), (2 * LEAST_READ, 16 * mib, 2 * LEAST_READ));
    let pro = recorded(&[
        (LEAST_READ, &[(1, 1.01), (2, 1.86), (4, 2.66), (8, 3.30), (16, 3.67), (32, 3.69), (128, 3.66)]),
        (2 * LEAST_READ, &[(1, 1.60), (2, 2.65), (4, 3.23), (8, 4.02), (16, 4.05), (32, 4.08), (128, 4.08)]),
    ]);
    let shape = Shape::of(&pro, None);
    assert_eq!((shape.ahead_read, shape.knee), (2 * LEAST_READ, 2 * mib));
    let gb10 = recorded(&[(LEAST_READ, &[(1, 0.438), (16, 6.02), (32, 9.66), (64, 10.09), (128, 10.12), (256, 10.10)])]);
    let lone = [Lone { read: LEAST_READ, seconds: 0.634e-3, error: 0.02e-3 }];
    let shape = Shape::of(&gb10, None);
    assert_eq!((shape.ahead_read, shape.knee), (LEAST_READ, 8 * mib));
    let expert = Urgent { parts: vec![589_824, 589_824, 720_896], records: 1 };
    let ranges = range_line(&expert, &[(1, 1.2), (2, 2.3), (4, 4.2), (8, 7.0), (16, 9.9), (32, 10.0)]);
    let shape = Shape::of(&gb10, Some(Request { urgent: &expert, lone: &lone, ranges: &ranges }));
    assert_eq!((shape.ahead_read, shape.knee, shape.due_read, shape.due), (8 * LEAST_READ, 16 * expert.range(), LEAST_READ, 32 * mib));
}

/// A due depth holds a step's whole request: where the due size's line
/// peaks shallower (an M4 Pro's internal drive, 1 MiB reads at 4 MiB), the
/// request's pieces are in flight at once, never waiting on the host for
/// another to end; one smaller than the line's full-rate window keeps that.
#[test]
fn a_due_depth_holds_a_steps_whole_request() {
    let mib = 1024 * 1024;
    let read = 8 * LEAST_READ;
    let internal = recorded(&[(read, &[(1, 2.1), (2, 3.6), (4, 5.05), (8, 4.6), (16, 4.5), (32, 4.5)])]);
    let lone = [Lone { read, seconds: 3.1e-3, error: 0.05e-3 }];
    let step = Urgent { parts: vec![589_824, 589_824, 720_896], records: 8 };
    let ranges = range_line(&step, &[(1, 1.9), (2, 3.2), (4, 3.9), (8, 3.95), (16, 3.9), (32, 3.9)]);
    assert_eq!(Shape::of(&internal, Some(Request { urgent: &step, lone: &lone, ranges: &ranges })).due, 16 * mib, "eight experts' 15.2 MB whole");
    let one = Urgent { records: 1, ..step };
    assert_eq!(Shape::of(&internal, Some(Request { urgent: &one, lone: &lone, ranges: &ranges })).due, 4 * mib, "the line's full-rate window holds one");
}

/// A request's knee is its range line's: an M4 Pro's internal drive reads
/// its 580–710 KiB ranges at full rate 4 deep through the engine, about
/// 2.5 MB, where its 1 MiB reads' line peaks at 4 MiB; its reads ahead are
/// cut whole, each range one piece, as the line read them.
#[test]
fn a_requests_knee_is_its_range_lines() {
    let mib = 1024 * 1024;
    let read = 8 * LEAST_READ;
    let internal = recorded(&[(read, &[(1, 2.1), (2, 3.6), (4, 5.05), (8, 4.6), (16, 4.5), (32, 4.5)])]);
    let lone = [Lone { read, seconds: 3.1e-3, error: 0.05e-3 }];
    let step = Urgent { parts: vec![589_824, 589_824, 720_896], records: 8 };
    assert_eq!(Shape::of(&internal, None).knee, 4 * mib, "the 1 MiB reads' knee");
    let ranges = range_line(&step, &[(1, 1.9), (2, 3.2), (4, 3.9), (8, 3.95), (16, 3.9), (32, 3.9)]);
    let shape = Shape::of(&internal, Some(Request { urgent: &step, lone: &lone, ranges: &ranges }));
    assert_eq!((shape.knee, shape.ahead_read), (4 * step.range(), read), "four ranges, each one piece");
    assert_eq!(step.depths(), vec![1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 27, 33, 41], "every depth to 8, a quarter more past it, to 32 MiB of 704 KiB ranges");
}

/// A request reads by the curve it adopted, whatever another request of
/// the volume adopts after it: a load's reads keep their shape when the
/// bank's request adopts its own curve of the same volume, and a second load
/// in the process reads as the first did.
#[test]
fn a_requests_shape_is_its_own_curves() {
    let step = Urgent { parts: vec![589_824, 589_824, 720_896], records: 8 };
    let loads = Curve { points: recorded(&[(LEAST_READ, &[(1, 1.2), (2, 2.3), (4, 3.6), (8, 4.4), (16, 4.5), (32, 4.5)])]), lone: Vec::new(), ranges: Vec::new() };
    let decode = Curve {
        points: recorded(&[(LEAST_READ, &[(1, 1.1), (2, 2.0), (4, 3.0), (8, 4.0), (16, 4.4), (32, 4.6), (64, 4.7)]), (8 * LEAST_READ, &[(1, 2.1), (2, 3.6), (4, 5.05), (8, 5.1)])]),
        lone: vec![Lone { read: 8 * LEAST_READ, seconds: 3.1e-3, error: 0.05e-3 }],
        ranges: range_line(&step, &[(1, 1.9), (2, 3.2), (4, 3.9), (8, 3.95), (16, 3.9)]),
    };
    let mut volume = VolumeLines::default();
    volume.adopt(None, &loads, 1 << 34);
    let alone = volume.shape(None);
    volume.adopt(Some(&step), &decode, 1 << 34);
    let bank = volume.shape(Some(&step));
    assert_eq!(volume.shape(None), alone, "the bank's curve leaves the loads' reads as they were");
    volume.adopt(None, &loads, 1 << 34);
    assert_eq!((volume.shape(None), volume.shape(Some(&step))), (alone, bank), "a second load reads as the first");
    assert_eq!(alone, Shape::of(&loads.points, None));
}

/// A request's probe reads its range line through the engine at each of
/// its depths, and its link reads ahead to that line's knee, whole ranges.
#[cfg(unix)]
#[test]
fn a_requests_probe_reads_its_range_line() {
    let _volume = temporary_volume();
    let path = temporary("range-line", &vec![7u8; 1 << 20]);
    let source = OpenFile::open(&path).unwrap();
    let urgent = Urgent { parts: vec![40_000, 24_000, 64_000], records: 2 };
    let curve = probe_volume(&source, Some(&urgent)).unwrap();
    let depths = curve.ranges.iter().map(|point| point.window / point.read).collect::<Vec<_>>();
    assert!(urgent.depths().starts_with(&depths) && !depths.is_empty(), "{depths:?}: its depths in order, to its first flat one");
    assert!(curve.ranges.iter().all(|point| point.rate > 0.), "{:?}", curve.ranges);
    let shape = link_shape(&source, Some(&urgent)).unwrap();
    assert_eq!(shape.knee % urgent.range(), 0, "a knee of whole ranges");
    assert_eq!(shape.ahead_read, LEAST_READ, "the least read holding the widest range");
    std::fs::remove_file(path).unwrap();
}

/// A request's range line at each depth, GB/s, with a spread of 2%.
fn range_line(urgent: &Urgent, line: &[(u64, f64)]) -> Vec<Point> {
    line.iter().map(|&(depth, rate)| Curve::range_point(urgent, depth, rate * 1e9, 0.02 * rate * 1e9, 1.)).collect()
}

/// A request's due pieces are cut at the size its lone record is served
/// soonest at, whatever the size read ahead; read sizes run to the least
/// holding the widest range.
#[test]
fn due_pieces_are_cut_at_the_size_a_lone_request_is_soonest() {
    let points = recorded(&[
        (LEAST_READ, &[(1, 0.5), (8, 3.), (64, 6.)]),
        (8 * LEAST_READ, &[(1, 2.), (8, 6.)]),
    ]);
    let lone = [Lone { read: LEAST_READ, seconds: 0.7e-3, error: 1e-5 }, Lone { read: 8 * LEAST_READ, seconds: 0.6e-3, error: 1e-5 }];
    let request = Urgent { parts: vec![589_824, 589_824, 720_896], records: 8 };
    let ranges = range_line(&request, &[(1, 2.), (8, 6.)]);
    let shape = Shape::of(&points, Some(Request { urgent: &request, lone: &lone, ranges: &ranges }));
    assert_eq!((shape.ahead_read, shape.due_read), (8 * LEAST_READ, 8 * LEAST_READ));
    let lone = [Lone { read: LEAST_READ, seconds: 0.5e-3, error: 1e-5 }, Lone { read: 8 * LEAST_READ, seconds: 0.6e-3, error: 1e-5 }];
    assert_eq!(Shape::of(&points, Some(Request { urgent: &request, lone: &lone, ranges: &ranges })).due_read, LEAST_READ);
    assert_eq!(request.reads(), vec![LEAST_READ, 2 * LEAST_READ, 4 * LEAST_READ, 8 * LEAST_READ]);
    assert_eq!(Urgent { parts: vec![4096], records: 1 }.reads(), vec![LEAST_READ]);
}

/// A stored range wider than the widest window is read in pieces of
/// that window: every read size probed has points of its line.
#[test]
fn a_range_past_the_widest_window_probes_only_sizes_a_window_holds() {
    let reads = Urgent { parts: vec![WIDEST_WINDOW + 1], records: 1 }.reads();
    assert_eq!(reads.last(), Some(&WIDEST_WINDOW));
    let grid = grid(reads.iter().copied());
    assert!(reads.iter().all(|read| grid.iter().any(|(probed, _)| probed == read)));
}

/// Nothing read ahead starts while a due piece is queued, however much
/// room its own depths leave: behind a due piece in flight at the due depth
/// (4 of 4), a second due piece waits, and a read ahead of 2 waits behind
/// it though the knee (8) holds it; once the first ends,
/// the second due piece goes first, then the read ahead.
#[test]
fn a_piece_read_ahead_never_starts_while_a_due_one_is_queued() {
    let link = volume(9, (4, 8), 4);
    let mut queue = Queue::default();
    queue.push(queued_on(&link, &[(0, 4), (4, 8)]), Want::now());
    queue.push(queued_on(&link, &[(8, 10)]), ahead());
    let first = queue.take().expect("a due piece");
    assert!(first.standing.due());
    assert!(queue.take().is_none(), "a read ahead started while a due piece was queued");
    queue.ended(first.returned(Instant::now()));
    let second = queue.take().expect("the second due piece");
    assert_eq!((second.piece.start, second.standing.due()), (4, true));
    let ahead = queue.take().expect("the read ahead, beside due work within the knee");
    assert_eq!((ahead.piece.start, ahead.standing.due()), (8, false));
}

/// A stream's due reads hold back nothing read ahead: a prompt group's
/// conveyor keeps its link at the due depth (4 of 4 streamed) with more
/// queued, and a fill of 2 starts beside it under the knee (8), as its
/// bytes are not counted against it; a second fill piece waits at the knee
/// only once the read ahead fills it.
#[test]
fn a_streams_due_reads_leave_the_fill_its_knee() {
    let link = volume(14, (4, 8), 4);
    let mut queue = Queue::default();
    queue.push(queued_on(&link, &[(0, 4), (4, 8)]), Want::streamed());
    queue.push(queued_on(&link, &[(8, 12), (12, 16), (16, 18)]), Want::undated(0.));
    let streamed = queue.take().expect("a streamed piece");
    assert_eq!(streamed.standing, Standing::Streamed);
    let fill = queue.take().expect("the fill beside a stream at its due depth");
    assert_eq!((fill.piece.start, fill.standing), (8, Standing::Ahead));
    let more = queue.take().expect("4 + 4 is the knee");
    assert_eq!(more.piece.start, 12);
    assert!(queue.take().is_none(), "the due depth and the knee are full");
    queue.ended(streamed.returned(Instant::now()));
    assert_eq!(queue.take().map(|taken| (taken.piece.start, taken.standing)), Some((4, Standing::Streamed)), "due first");
}

/// A parked step's due read still holds back everything read ahead beside
/// a stream: once a parked piece is queued behind the stream's at the due
/// depth, no fill piece starts, though the knee holds it.
#[test]
fn a_parked_read_holds_the_fill_back_beside_a_stream() {
    let link = volume(15, (4, 8), 4);
    let mut queue = Queue::default();
    queue.push(queued_on(&link, &[(0, 4)]), Want::streamed());
    let streamed = queue.take().expect("a streamed piece");
    queue.push(queued_on(&link, &[(4, 8)]), Want::now());
    queue.push(queued_on(&link, &[(8, 10)]), Want::undated(0.));
    assert!(queue.take().is_none(), "a fill started while a parked piece was queued");
    queue.ended(streamed.returned(Instant::now()));
    assert_eq!(queue.take().map(|taken| (taken.piece.start, taken.standing)), Some((4, Standing::Parked)));
    assert_eq!(queue.take().map(|taken| (taken.piece.start, taken.standing)), Some((8, Standing::Ahead)), "within the knee once nothing parked is queued");
}

/// A piece read ahead starts only while everything in flight, due and
/// ahead, leaves it room under the knee: beside a due piece of 4 (knee 8),
/// two reads ahead of 2 start and a third waits; once the due piece ends,
/// two more start. Another link is held by none of it.
#[test]
fn a_piece_read_ahead_starts_only_while_everything_in_flight_fits_the_knee() {
    let link = volume(10, (8, 8), 4);
    let mut queue = Queue::default();
    queue.push(queued_on(&link, &[(0, 4)]), Want::now());
    let due = queue.take().unwrap();
    queue.push(queued_on(&link, &[(4, 6), (6, 8), (8, 10), (10, 12), (12, 14)]), ahead());
    assert_eq!(queue.take().map(|taken| taken.piece.start), Some(4), "4 + 2 is within the knee");
    assert_eq!(queue.take().map(|taken| taken.piece.start), Some(6), "4 + 4 is the knee");
    assert!(queue.take().is_none(), "4 + 4 + 2 passes the knee");
    let other = volume(11, (8, 8), 4);
    queue.push(queued_on(&other, &[(0, 2)]), ahead());
    assert_eq!(queue.take().map(|taken| (taken.volume.id, taken.piece.start)), Some((Volume(11), 0)), "another link's room is its own");
    queue.ended(due.returned(Instant::now()));
    assert_eq!(queue.take().map(|taken| taken.piece.start), Some(8));
    assert_eq!(queue.take().map(|taken| taken.piece.start), Some(10), "8 of 8 read ahead");
    assert!(queue.take().is_none(), "past the knee");
}

/// Due pieces join the due bytes in flight up to the due depth, whatever
/// is read ahead: a due piece arriving at a link full of reads ahead starts
/// at once, which bounds its wait by what was in flight.
#[test]
fn a_due_piece_starts_beside_a_full_knee() {
    let link = volume(12, (4, 8), 4);
    let mut queue = Queue::default();
    queue.push(queued_on(&link, &[(0, 4), (4, 8), (8, 12)]), ahead());
    assert!(queue.take().is_some() && queue.take().is_some());
    assert!(queue.take().is_none(), "8 of 8 read ahead");
    queue.push(queued_on(&link, &[(12, 16), (16, 20)]), Want::now());
    assert_eq!(queue.take().map(|taken| (taken.piece.start, taken.standing.due())), Some((12, true)));
    assert!(queue.take().is_none(), "4 of 4 due, and a due piece queued");
}

/// A read ahead a consumer comes to wait on starts as due once repriced:
/// held back beside a due piece past the knee, it starts at the due
/// depth from its reprice, and a worker is woken for it.
#[test]
fn a_repriced_request_starts_at_its_new_depth() {
    let link = volume(13, (8, 4), 4);
    let mut queue = Queue::default();
    queue.push(queued_on(&link, &[(0, 2)]), Want::now());
    assert!(queue.take().is_some());
    let (id, woken) = queue.push(queued_on(&link, &[(2, 6)]), ahead());
    assert_eq!(woken, 0);
    assert!(queue.take().is_none(), "past the knee");
    assert_eq!(queue.reprice(id, Want::now()), 1);
    assert!(queue.take().is_some_and(|taken| taken.standing.due()), "within the due depth");
}

/// A request wakes a worker for each of its pieces its link may start:
/// none where its depth is full, one where nothing is in flight even for a
/// piece past it.
#[test]
fn a_request_wakes_a_worker_for_each_piece_its_link_may_start() {
    let link = volume(7, (4, 4), 2);
    let mut queue = Queue::default();
    assert_eq!(queue.push(queued_on(&link, &[(0, 2), (2, 4), (4, 6)]), Want::now()).1, 2);
    let mut other = Queue::default();
    assert_eq!(other.push(queued_on(&link, &[(0, 9)]), Want::now()).1, 1, "a piece past the depth is taken alone");
    queue.take().unwrap();
    queue.take().unwrap();
    assert_eq!(queue.push(queued_on(&link, &[(6, 8)]), Want::now()).1, 0, "a full depth wakes none");
}

/// Whether two counts of seconds agree, to rounding.
fn close(left: f64, right: f64) -> bool {
    (left - right).abs() < 1e-9
}

/// A volume's book counts, for due pieces and pieces read ahead, the time
/// a piece of the kind was in flight (the union of overlapping pieces'
/// spans), its pieces in flight integrated over time (their sum), and each
/// piece's bytes at its end; the device's busy time is the union over both,
/// none while idle. A look counts the time running to its instant and
/// consumes nothing: a later look counts it again, with what followed.
#[test]
fn a_volumes_book_counts_busy_time_depth_and_bytes_due_and_ahead() {
    let at = Instant::now();
    let mut book = Book::new(at);
    let ms = |ms: u64| at + Duration::from_millis(ms);
    book.began(true, ms(0));
    book.began(false, ms(2));
    book.ended(true, 1_000_000, ms(4));
    // A look while the piece read ahead is in flight: counted to 5 ms.
    let middle = book.totals_at(ms(5));
    assert!(close(middle.busy, 0.005) && close(middle.depth, 0.007), "{middle:?}");
    let due = middle.due;
    assert_eq!((due.bytes, due.requests), (1_000_000, 1));
    assert!(close(due.busy, 0.004) && close(due.depth, 0.004), "{due:?}");
    let ahead = middle.ahead;
    assert_eq!((ahead.bytes, ahead.requests), (0, 0), "a piece's bytes count at its end");
    assert!(close(ahead.busy, 0.003) && close(ahead.depth, 0.003), "{ahead:?}");
    book.ended(false, 500_000, ms(8));
    // Idle 8 to 30 ms: no busy time.
    book.began(false, ms(30));
    book.ended(false, 250_000, ms(34));
    let later = book.totals_at(ms(40));
    assert!(close(later.busy, 0.012) && close(later.depth, 0.014), "{later:?}");
    assert_eq!((later.bytes(), later.requests()), (1_750_000, 3));
    assert!(close(later.ahead.busy, 0.010) && close(later.due.busy, 0.004), "{later:?}");
    let span = later.since(&middle);
    assert!(close(span.busy, 0.007) && span.bytes() == 750_000, "{span:?}");
    assert_eq!(book.totals_at(ms(40)), later, "a look takes nothing");
}

/// A due request's wait from its submission to its last piece's end is
/// its device time (a piece of it in flight) and its time waiting, split by
/// what the device held meanwhile: behind a piece read ahead while one was
/// in flight, behind due pieces otherwise; counted as due once its last
/// piece has ended.
#[test]
fn a_due_requests_wait_splits_by_what_it_waited_behind() {
    let at = Instant::now();
    let mut book = Book::new(at);
    let ms = |ms: u64| at + Duration::from_millis(ms);
    // A piece read ahead in flight, then another request's due piece.
    book.began(false, ms(0));
    let mut wait = Wait::new(2, true, &mut book, ms(0));
    book.ended(false, 1, ms(3));
    book.began(true, ms(3));
    // Its first piece: 3 ms behind the read ahead, 2 behind the due piece.
    book.began(true, ms(5));
    wait.taken(true, &mut book, ms(5));
    book.ended(true, 1, ms(7));
    assert!(!wait.ended(&mut book, ms(7)), "a request with a piece queued has not ended");
    assert_eq!(book.totals_at(ms(7)).due.waits.requests, 0, "a request with a piece queued has not ended");
    // Its second piece, behind due pieces alone.
    book.began(true, ms(9));
    wait.taken(true, &mut book, ms(9));
    book.ended(true, 1, ms(10));
    book.ended(true, 1, ms(12));
    assert!(wait.ended(&mut book, ms(12)), "its last piece ended it");
    let waits = book.totals_at(ms(12)).due.waits;
    assert_eq!(waits.requests, 1);
    assert!(close(waits.behind_ahead, 0.003) && close(waits.behind_due, 0.004) && close(waits.device, 0.005), "{waits:?}");
}

/// A link holds at most its due depth of due bytes in flight, or one piece
/// past it: its most urgent piece that would pass it holds back the link's
/// others, even smaller ones, while another link's less urgent pieces are
/// taken; one ending says the link has a piece that may now be.
#[test]
fn a_link_holds_at_most_its_due_depth_in_flight() {
    let narrow = volume(1, (3, 3), 2);
    let mut queue = Queue::default();
    let urgent = queued_on(&narrow, &[(0, 2), (2, 4)]);
    let urgent_state = urgent.state.clone();
    queue.push(urgent, Want::now());
    queue.push(queued_on(&narrow, &[(8, 9)]), Want::now());
    let (other, _) = queued(1);
    queue.push(other, Want::undated(0.));
    // Each piece taken, with its link and start.
    let taken = |queue: &mut Queue| queue.take().map(|taken| ((taken.volume.id, taken.piece.start), taken));
    let (first, urgent_first) = taken(&mut queue).unwrap();
    assert_eq!(first, (Volume(1), 0));
    let (other, other_piece) = taken(&mut queue).unwrap();
    assert_eq!(other, (Volume(0), 0), "2 + 2 passes the due depth; 2 + 1 would not, but waits");
    assert!(taken(&mut queue).is_none());
    assert!(urgent_state.started.get().is_some(), "its first piece was taken");
    assert!(queue.ended(urgent_first.returned(Instant::now())), "a piece was held back");
    assert_eq!(taken(&mut queue).map(|taken| taken.0), Some((Volume(1), 2)));
    assert_eq!(taken(&mut queue).map(|taken| taken.0), Some((Volume(1), 8)), "2 + 1 is within the due depth");
    assert!(!queue.ended(other_piece.returned(Instant::now())));
    // A piece wider than the depth is read alone.
    queue.push(queued_on(&VolumeReads { id: Volume(2), ..narrow }, &[(0, 5)]), Want::now());
    assert_eq!(taken(&mut queue).map(|taken| taken.0), Some((Volume(2), 0)));
}
