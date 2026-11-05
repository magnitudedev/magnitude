use super::*;
use super::engine::{Aligned, Buffers, Engine};
use super::probe::{decide, grid, Point, LEAST_READ, WIDEST_WINDOW};
use super::queue::{Part, Piece, Queue, VolumeFlight};
#[cfg(unix)]
use super::platform::drop_pages;
use super::volume::{Bound, VolumeKey};
use std::collections::HashMap;
#[cfg(unix)]
use std::fs::File;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Instant;
use std::path::PathBuf;

/// A volume's bound is the process's one, shared by every reader of the
/// volume (`Bound`): an urgent reader a test opens on the temporary
/// directory's volume narrows every other test's bound there while it
/// lives. A test that opens one, or reads the bound, holds the volume.
static TEMPORARY_VOLUME: Mutex<()> = Mutex::new(());

/// Hold [`TEMPORARY_VOLUME`] for this test.
fn temporary_volume() -> std::sync::MutexGuard<'static, ()> {
    TEMPORARY_VOLUME.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn temporary(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "magnitude-reader-{name}-{}.bin",
        std::process::id()
    ));
    std::fs::write(&path, bytes).unwrap();
    path
}

/// A demand read beside bulk from other readers of its volume (two
/// loads keeping 4 MiB jobs queued and an urgent reader keeping its
/// hints queued) waits behind at most the pieces in flight, never behind
/// the bulk queued: within 0.5 s each, on any device a test runs on (a
/// 4 KiB read behind a few MiB in flight). On an RTX 3090 host: p50
/// 0.16 ms, max 0.41; beside 136 busy threads on its 34 cores, max 9.8.
#[test]
fn a_demand_read_beside_bulk_waits_only_for_the_pieces_in_flight() {
    let _volume = temporary_volume();
    let path = temporary("demand-beside-bulk", &vec![5u8; 16 << 20]);
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let urgent = Urgent { parts: vec![4096], records: 1 };
    let flood = |reader: ArtifactReader, priority: Priority, length: usize| {
        let stop = stop.clone();
        std::thread::spawn(move || {
            let mut sink = vec![0u8; length];
            let mut at = 0u64;
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                let range = Range { offset: at, length, sink: NonNull::new(sink.as_mut_ptr()).unwrap() };
                // SAFETY: the sink outlives the job, waited for here.
                let ended = unsafe { reader.submit(vec![range], priority) }.unwrap().wait();
                assert!(matches!(ended, Ended::Read { ref failed } if failed.is_empty()), "{ended:?}");
                at = (at + length as u64) % (reader.size() - length as u64);
            }
        })
    };
    let floods = [
        flood(ArtifactReader::open(&path, CachePolicy::Uncached).unwrap(), Priority::Load, 4 << 20),
        flood(ArtifactReader::open(&path, CachePolicy::Uncached).unwrap(), Priority::Load, 4 << 20),
        flood(ArtifactReader::open_urgent(&path, CachePolicy::Uncached, &urgent).unwrap(), Priority::Prefetch, 1 << 20),
    ];
    let reader = ArtifactReader::open_urgent(&path, CachePolicy::Uncached, &urgent).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));
    let mut sink = vec![0u8; 4096];
    let longest = (0..100u64)
        .map(|sample| {
            let range = Range { offset: (sample * 7919 * 4096) % (reader.size() - 4096), length: 4096, sink: NonNull::new(sink.as_mut_ptr()).unwrap() };
            let began = Instant::now();
            // SAFETY: the sink outlives the job, waited for here.
            let ended = unsafe { reader.submit(vec![range], Priority::Demand) }.unwrap().wait();
            assert!(matches!(ended, Ended::Read { ref failed } if failed.is_empty()), "{ended:?}");
            began.elapsed().as_secs_f64() * 1e3
        })
        .fold(0., f64::max);
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for thread in floods {
        thread.join().unwrap();
    }
    std::fs::remove_file(path).unwrap();
    assert!(longest < 500., "a demand read waited {longest:.1} ms behind bulk");
}

/// Each range of `ranges`, `(offset, length)`, read as one demand job
/// into a buffer of its own: its bytes, or the error that kept them
/// from landing.
fn read(source: &ArtifactReader, ranges: &[(u64, usize)]) -> Vec<Result<Vec<u8>, String>> {
    let mut buffers = ranges
        .iter()
        .map(|(_, length)| vec![0u8; *length])
        .collect::<Vec<_>>();
    let landing = ranges
        .iter()
        .zip(&mut buffers)
        .map(|((offset, length), buffer)| Range {
            offset: *offset,
            length: *length,
            sink: NonNull::new(buffer.as_mut_ptr()).unwrap(),
        })
        .collect();
    let ended = unsafe { source.submit(landing, Priority::Demand) }.unwrap().wait();
    let Ended::Read { failed } = ended else {
        panic!("the job ended {ended:?}")
    };
    let failed = failed.into_iter().collect::<HashMap<_, _>>();
    buffers
        .into_iter()
        .enumerate()
        .map(|(index, buffer)| match failed.get(&index) {
            Some(error) => Err(error.to_string()),
            None => Ok(buffer),
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
        let source = ArtifactReader::open(&path, policy).unwrap();
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

/// A job's hook runs once, with how it ended: registered before it
/// ends, on the thread that ends it; registered after, at once. A job of
/// no piece ends at its submission. Each hook is waited for through what
/// it sends, never through time.
#[test]
fn a_jobs_hook_runs_once_with_how_it_ended() {
    let path = temporary("hook", &vec![3u8; 4 * LEAST_READ as usize]);
    let reader = ArtifactReader::open(&path, CachePolicy::Uncached).unwrap();
    let mut sink = vec![0u8; 3 * LEAST_READ as usize];
    let range = Range { offset: 0, length: sink.len(), sink: NonNull::new(sink.as_mut_ptr()).unwrap() };
    let (sent, ran) = std::sync::mpsc::channel::<Ended>();
    let job = unsafe { reader.submit(vec![range], Priority::Demand) }.unwrap();
    let send = sent.clone();
    job.on_end(move |ended| send.send(ended.clone()).unwrap());
    assert!(matches!(job.wait(), Ended::Read { .. }));
    assert!(matches!(ran.recv().unwrap(), Ended::Read { failed } if failed.is_empty()));
    let send = sent.clone();
    job.on_end(move |ended| send.send(ended.clone()).unwrap());
    assert!(ran.try_recv().is_ok(), "a hook after the end runs at once");
    let outside = Range { offset: 8 * LEAST_READ, length: 1, sink: NonNull::new(sink.as_mut_ptr()).unwrap() };
    let failed = unsafe { reader.submit(vec![outside], Priority::Demand) }.unwrap();
    failed.on_end(move |ended| sent.send(ended.clone()).unwrap());
    assert!(matches!(ran.try_recv().unwrap(), Ended::Read { failed } if failed.len() == 1), "a job of no piece ended at its submission");
    assert!(ran.try_recv().is_err(), "each hook ran once");
    drop((job, failed));
    std::fs::remove_file(path).unwrap();
}

/// The job's end is set before its hook is announced: whatever a hook
/// wakes finds the job ended. The interleaving that hung the suite
/// (residency woken by a conveyor read's hook, finding the read still
/// in flight and staying asked, with no event left to wake it) is driven
/// by construction: the hook itself asks whether the job runs, the
/// question a woken `Conveyor::quiet` asks through `Ticket::running`.
/// It finds it ended, for a job a worker ends, for one withdrawn, and
/// for one of no piece.
#[test]
fn a_hook_finds_its_job_ended() {
    let path = temporary("ended-first", &vec![4u8; 4 * LEAST_READ as usize]);
    let reader = ArtifactReader::open(&path, CachePolicy::Uncached).unwrap();
    let mut sink = vec![0u8; LEAST_READ as usize];
    let (sent, ran) = std::sync::mpsc::channel::<bool>();
    // Whether the job runs, asked from inside its hook.
    let ask = |job: Arc<Job>, sent: std::sync::mpsc::Sender<bool>| {
        let asked = job.clone();
        job.on_end(move |_| sent.send(asked.wait_until(Instant::now()).is_none()).unwrap());
    };
    let range = |sink: &mut Vec<u8>, offset| Range { offset, length: sink.len(), sink: NonNull::new(sink.as_mut_ptr()).unwrap() };
    // Ended by the worker that reads its piece.
    let read = Arc::new(unsafe { reader.submit(vec![range(&mut sink, 0)], Priority::Demand) }.unwrap());
    ask(read.clone(), sent.clone());
    assert!(!ran.recv().unwrap(), "a hook saw its read still running");
    // Ended at its submission (no piece in the file).
    let outside = Arc::new(unsafe { reader.submit(vec![range(&mut sink, 8 * LEAST_READ)], Priority::Demand) }.unwrap());
    ask(outside.clone(), sent.clone());
    assert!(!ran.recv().unwrap());
    // Ended by its withdrawal, its hook registered first.
    let mut late = vec![0u8; LEAST_READ as usize];
    let withdrawn = Arc::new(unsafe { reader.submit(vec![range(&mut late, LEAST_READ)], Priority::Load) }.unwrap());
    ask(withdrawn.clone(), sent);
    withdrawn.withdraw();
    assert!(!ran.recv().unwrap());
    drop((read, outside, withdrawn));
    std::fs::remove_file(path).unwrap();
}

/// A cached piece lands through its worker's buffer even where it could
/// be read straight into its sink; an uncached one is read straight.
#[test]
fn a_cached_piece_lands_through_the_buffer() {
    let path = temporary("straight", &vec![1u8; 2 * LEAST_READ as usize]);
    let mut sink = Aligned::new(LEAST_READ as usize);
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
        let reader = ArtifactReader::open(&path, policy).unwrap();
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
    let (cached, pages) = super::cached_pages(path, offset, length as u64).unwrap();
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
    let reader = ArtifactReader::open(&path, CachePolicy::Cached).unwrap();
    let (offset, length) = (LEAST_READ + 100, 3 * LEAST_READ as usize);
    // Linux drops a synced file's pages.
    #[cfg(target_os = "linux")]
    assert!(!resident(&path, offset, length));
    assert!(read(&reader, &[(offset, length)])[0].is_ok());
    assert!(resident(&path, offset, length));
    std::fs::remove_file(path).unwrap();
}

/// A source's range asks the cache for exactly itself: an empty one for
/// nothing (advice of length zero would mean the rest of the file), a
/// range read whole for its pages and none beside them.
#[cfg(target_os = "linux")]
#[test]
fn a_source_range_caches_only_itself() {
    use std::io::Read as _;
    let path = temporary("source-range", &vec![7u8; 8 * LEAST_READ as usize]);
    let written = File::open(&path).unwrap();
    written.sync_all().unwrap();
    drop_pages(&written, 0, 8 * LEAST_READ).unwrap();
    let source = crate::FileSource::open(&path).unwrap();
    let mut empty = source.range(LEAST_READ, 0).unwrap();
    assert_eq!(empty.read(&mut [0u8; 1]).unwrap(), 0);
    assert_eq!(super::cached_pages(&path, 0, 8 * LEAST_READ).unwrap().0, 0);
    let mut bytes = Vec::new();
    source.range(2 * LEAST_READ, LEAST_READ).unwrap().read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, vec![7u8; LEAST_READ as usize]);
    assert!(resident(&path, 2 * LEAST_READ, LEAST_READ as usize));
    assert_eq!(super::cached_pages(&path, 0, 2 * LEAST_READ).unwrap().0, 0);
    assert_eq!(super::cached_pages(&path, 3 * LEAST_READ, 5 * LEAST_READ).unwrap().0, 0);
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
    let reader = ArtifactReader::open(&path, CachePolicy::Uncached).unwrap();
    reader.evict_cached().unwrap();
    assert_eq!(super::cached_pages(&path, 0, length).unwrap().0, 0);
    assert!(read(&reader, &[(LEAST_READ, 2 * LEAST_READ as usize)])[0].is_ok());
    assert_eq!(super::cached_pages(&path, 0, length).unwrap().0, 0, "an uncached read caches nothing");
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
    let source = ArtifactReader::open(&path, CachePolicy::Uncached).unwrap();
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

/// A job whose piece fails ends with its ranges failed; a job submitted
/// beside it lands every range.
#[test]
fn a_failed_piece_fails_only_its_job() {
    let bytes = (0..16 * LEAST_READ as usize)
        .map(|index| (index * 13 % 241) as u8)
        .collect::<Vec<_>>();
    let path = temporary("job", &bytes);
    let source = ArtifactReader::open(&path, CachePolicy::Uncached).unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(8 * LEAST_READ)
        .unwrap();
    let mut short = vec![0u8; 2 * LEAST_READ as usize];
    let mut whole = vec![0u8; 3 * LEAST_READ as usize];
    let range = |offset: u64, sink: &mut Vec<u8>| Range {
        offset,
        length: sink.len(),
        sink: NonNull::new(sink.as_mut_ptr()).unwrap(),
    };
    let (failing, landing) = unsafe {
        (
            source
                .submit(vec![range(10 * LEAST_READ, &mut short)], Priority::Demand)
                .unwrap(),
            source
                .submit(vec![range(LEAST_READ + 5, &mut whole)], Priority::Demand)
                .unwrap(),
        )
    };
    assert!(matches!(failing.wait(), Ended::Read { failed } if failed.len() == 1));
    assert!(matches!(landing.wait(), Ended::Read { failed } if failed.is_empty()));
    drop((failing, landing));
    let at = LEAST_READ as usize + 5;
    assert_eq!(whole, bytes[at..at + whole.len()]);
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
    let range = |offset: u64, length: usize| Range { offset, length, sink: sink() };
    let ranges = [range(0, 90), range(ALIGNMENT + 100, 90)];
    let pieces = pieces(&ranges, &[0, 1], 10_000_000, LEAST_READ);
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
    let range = |offset: u64, length: usize| Range {
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
    let pieces = pieces(&ranges, &[0, 1, 3, 2], 10_000_000, LEAST_READ);
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

/// A volume `key` of its own, read as named, sharing nothing.
fn volume(key: u64, window: u64, demand: u64, read: u64) -> Volume {
    let choice = VolumeChoice { window, demand, read, demand_read: read, cores: 0. };
    Volume { key: VolumeKey(key), choice, background: Arc::default(), delivery: Arc::default() }
}

/// A queued job of its own, `pieces` pieces long.
fn queued(pieces: usize) -> (Queued, Arc<JobState>) {
    let state = JobState::new(pieces, Vec::new(), Arc::default());
    let path = std::env::current_exe().unwrap();
    let file = ArtifactReader::open(path, CachePolicy::Uncached)
        .unwrap()
        .file;
    let job = Queued {
        state: state.clone(),
        file,
        volume: volume(0, u64::MAX, u64::MAX, LEAST_READ),
        pieces: (0..pieces as u64)
            .map(|index| Piece {
                start: index,
                end: index + 1,
                parts: Vec::new(),
            })
            .collect(),
    };
    (job, state)
}

/// Pieces are taken by priority, then by submission, then in their
/// job's order; a promotion moves a job ahead and never behind.
#[test]
fn pieces_are_taken_most_urgent_first() {
    let mut queue = Queue::default();
    let mut states = Vec::new();
    let push = |queue: &mut Queue, states: &mut Vec<Arc<JobState>>, priority, pieces| {
        let (job, state) = queued(pieces);
        states.push(state);
        queue.push(job, priority).0
    };
    // A load's job, though submitted first, is read behind every step's.
    push(&mut queue, &mut states, Priority::Load, 1);
    push(&mut queue, &mut states, Priority::Prefetch, 1);
    push(&mut queue, &mut states, Priority::Prefetch, 2);
    push(&mut queue, &mut states, Priority::Demand, 1);
    let later = push(&mut queue, &mut states, Priority::Prefetch, 1);
    // Each take as (the job's index in submission, the piece).
    let take = |queue: &mut Queue, states: &[Arc<JobState>]| {
        let (state, _, _, piece) = queue.take().unwrap();
        let job = states.iter().position(|known| Arc::ptr_eq(known, &state));
        (job.unwrap(), piece.start)
    };
    assert_eq!(take(&mut queue, &states), (3, 0));
    queue.promote(later, Priority::Demand);
    assert_eq!(take(&mut queue, &states), (4, 0));
    // A demand job is not demoted: it stays ahead of every prefetch.
    let demand = push(&mut queue, &mut states, Priority::Demand, 1);
    queue.promote(demand, Priority::Prefetch);
    assert_eq!(take(&mut queue, &states), (5, 0));
    assert_eq!(take(&mut queue, &states), (1, 0));
    assert_eq!(take(&mut queue, &states), (2, 0));
    assert_eq!(take(&mut queue, &states), (2, 1));
    assert_eq!(take(&mut queue, &states), (0, 0));
    assert!(queue.take().is_none());
}

/// A withdrawn job gives up the pieces no worker took, a started one
/// too, once; and a job with no piece ends at once.
#[test]
fn a_job_withdraws_the_pieces_no_worker_took() {
    let mut queue = Queue::default();
    let (started, _) = queue.push(queued(3).0, Priority::Demand);
    let (untouched, _) = queued(2);
    let (untouched, _) = queue.push(untouched, Priority::Demand);
    queue.take().unwrap();
    assert_eq!(queue.withdraw(started).unwrap().pieces.len(), 2);
    assert!(queue.withdraw(started).is_none());
    assert_eq!(queue.withdraw(untouched).unwrap().pieces.len(), 2);
    assert!(queue.withdraw(untouched).is_none());
    assert!(queue.take().is_none());
    let (empty, state) = queued(0);
    queue.push(empty, Priority::Demand);
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
    let source = crate::FileSource::open(&path).unwrap();
    let replacement = temporary("replacement", &vec![9u8; original.len()]);
    std::fs::rename(&replacement, &path).unwrap();
    for policy in [CachePolicy::Cached, CachePolicy::Uncached] {
        let reader = ArtifactReader::of(&source, policy).unwrap();
        assert_eq!(read(&reader, &[(0, original.len())]), [Ok(original.clone())], "{policy:?}");
    }
    std::fs::remove_file(path).unwrap();
}

/// A withdrawal returns the bytes of the ranges its untaken pieces served,
/// exactly those that end withdrawn; a piece in flight reads on and is
/// not returned. Driven on an engine of its own with no worker, so which
/// pieces are in flight is the test's: of four pieces, one taken (as a
/// worker takes it), three withdrawn; the taken one's end ends the job.
#[test]
fn a_withdrawal_returns_the_bytes_it_left_unread() {
    static ENGINE: OnceLock<Engine> = OnceLock::new();
    let engine = ENGINE.get_or_init(|| Engine {
        queue: Mutex::new(Queue::default()),
        work: Condvar::new(),
        workers: Mutex::new(0),
        flights: Mutex::new(Vec::new()),
        buffers: Buffers::default(),
    });
    let (mut queued, state) = queued(4);
    let piece = |start: u64, range: usize| Piece {
        start,
        end: start + LEAST_READ,
        parts: vec![Part { range, from: 0, length: LEAST_READ as usize, sink: NonNull::dangling() }],
    };
    queued.pieces = (0..4).map(|at| piece(at * 3 * LEAST_READ, at as usize)).collect();
    let (id, _) = engine.queue.lock().unwrap().push(queued, Priority::Load);
    let job = Job { id, state: state.clone(), engine };
    let taken = engine.queue.lock().unwrap().take().expect("a piece to take");
    job.withdraw();
    assert!(job.wait_until(Instant::now()).is_none(), "the taken piece reads on");
    // The taken piece's end, as its worker ends it.
    taken.0.pieces_ended(1, [], None);
    let Ended::Read { failed } = job.wait() else { panic!("a withdrawn job reads on") };
    assert_eq!(failed.iter().map(|(range, _)| *range).collect::<Vec<_>>(), [1, 2, 3], "the ranges that ended withdrawn");
    assert!(failed.iter().all(|(_, error)| matches!(**error, Error::Withdrawn)));
    job.withdraw();
    drop(job);
}

/// A job dropped before it is waited for withdraws or finishes first:
/// a withdrawn job lands nothing.
#[test]
fn a_dropped_job_withdraws_or_finishes() {
    let bytes = vec![7u8; 4 * LEAST_READ as usize];
    let path = temporary("cancel", &bytes);
    let source = ArtifactReader::open(&path, CachePolicy::Uncached).unwrap();
    let mut sink = vec![0u8; LEAST_READ as usize];
    let job = unsafe {
        source.submit(
            vec![Range {
                offset: 0,
                length: sink.len(),
                sink: NonNull::new(sink.as_mut_ptr()).unwrap(),
            }],
            Priority::Prefetch,
        )
    }
    .unwrap();
    drop(job);
    // Withdrawn before a worker took its one piece, or read whole by
    // then.
    assert!(sink.iter().all(|byte| *byte == 0) || sink.iter().all(|byte| *byte == 7));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn a_reader_reads_at_the_bounds_until_its_volume_adopts_a_choice() {
    let path = temporary("adopted", &[5u8; 4096]);
    let source = crate::FileSource::open(&path).unwrap();
    // A key its own, as a volume's other tests choose for `None`.
    let urgent = Urgent { parts: vec![4096, 8192, 12288], records: 3 };
    let pending = ArtifactReader::open_urgent(&path, CachePolicy::Uncached, &urgent).unwrap();
    assert_eq!((pending.window(), pending.volume.choice.read), (WIDEST_WINDOW, LEAST_READ), "nothing was measured");
    let choice = VolumeChoice { window: 4 << 20, demand: 8 << 20, read: 256 << 10, demand_read: 1 << 20, cores: 1.5 };
    adopt_volume(&source, Some(&urgent), choice).unwrap();
    let adopted = ArtifactReader::open_urgent(&path, CachePolicy::Uncached, &urgent).unwrap();
    assert_eq!((adopted.window(), adopted.volume.choice.demand, adopted.volume.choice.read), (4 << 20, 8 << 20, 256 << 10));
    assert_eq!(read_cores([&source], &urgent).unwrap(), 1.5);
    std::fs::remove_file(path).unwrap();
}

/// A prefetch withdrawn behind demand ends with what it had in flight,
/// its range failed `Withdrawn` unless every piece of it was taken, and
/// the demand ahead of it lands whole.
#[test]
fn a_withdrawn_prefetch_ends_with_what_it_had_in_flight_and_demand_lands() {
    // Probed on a file of its own, so the file read is four windows long.
    let probed = temporary("withdraw-window", &[5u8; 4096]);
    probe_volume(&crate::FileSource::open(&probed).unwrap(), None).unwrap();
    let window = ArtifactReader::open(&probed, CachePolicy::Uncached).unwrap().window();
    std::fs::remove_file(probed).unwrap();
    let bytes = vec![5u8; 4 * window as usize];
    let path = temporary("withdraw", &bytes);
    let source = ArtifactReader::open(&path, CachePolicy::Uncached).unwrap();
    let mut demand = vec![0u8; bytes.len()];
    let mut ahead = vec![0u8; 2 * LEAST_READ as usize];
    let range = |sink: &mut Vec<u8>| Range { offset: 0, length: sink.len(), sink: NonNull::new(sink.as_mut_ptr()).unwrap() };
    // SAFETY: each sink outlives its job, which ends before it is read.
    let demanded = unsafe { source.submit(vec![range(&mut demand)], Priority::Demand) }.unwrap();
    let prefetch = unsafe { source.submit(vec![range(&mut ahead)], Priority::Prefetch) }.unwrap();
    prefetch.withdraw();
    let ended = prefetch.wait();
    match &ended {
        Ended::Read { failed } if failed.is_empty() => assert!(ahead.iter().all(|byte| *byte == 5)),
        Ended::Read { failed } => assert!(failed.len() == 1 && matches!(*failed[0].1, Error::Withdrawn)),
        Ended::Panicked(panic) => panic.resume(),
    }
    prefetch.withdraw();
    assert!(matches!(demanded.wait(), Ended::Read { failed } if failed.is_empty()));
    assert!(demand.iter().all(|byte| *byte == 5));
    std::fs::remove_file(path).unwrap();
}

/// A request is served at the point of least `T`, priced from the
/// curve: GB10's medians and its lone
/// experts by read size put a decode request of
/// 8 records at 4 MiB in 128 KiB reads, where 16 MiB in 2 MiB reads
/// reads 10% faster but waits 0.8 ms more; a near tie takes the narrower
/// window, then the least read; a reader's widest range bounds its read
/// sizes.
#[test]
fn a_request_is_served_at_the_point_of_least_time() {
    let mib = 1024 * 1024;
    let point = |read: u64, window: u64, rate: f64| Point { read, window, rate: rate * 1e9, spread: 0.01e9, cores: 1. };
    let points = [
        point(LEAST_READ, 2 * mib, 6.02),
        point(LEAST_READ, 4 * mib, 9.66),
        point(LEAST_READ, 8 * mib, 10.09),
        point(16 * LEAST_READ, 8 * mib, 10.09),
        point(16 * LEAST_READ, 16 * mib, 10.62),
    ];
    let lone = HashMap::from([(LEAST_READ, (0.634e-3, 0.02e-3)), (16 * LEAST_READ, (0.699e-3, 0.02e-3))]);
    let urgent = Urgent { parts: vec![589_824, 589_824, 720_896], records: 8 };
    let chosen = decide(&points, &lone, &urgent);
    assert_eq!((chosen.read, chosen.window), (LEAST_READ, 4 * mib));
    // Two points whose times are within their noise: the narrower window.
    let tied = [point(LEAST_READ, 4 * mib, 9.66), point(LEAST_READ, 8 * mib, 18.0)];
    let lone = HashMap::from([(LEAST_READ, (0.5e-3, 1e-3))]);
    assert_eq!(decide(&tied, &lone, &urgent).window, 4 * mib);
    // Read sizes run to the least holding the widest range.
    assert_eq!(urgent.reads(), vec![LEAST_READ, 2 * LEAST_READ, 4 * LEAST_READ, 8 * LEAST_READ]);
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

/// A demand piece joins the bytes in flight up to the full rate's window,
/// any other piece only up to the urgent request's: behind two bytes in
/// flight of a window of 4 (and a demand window of 8), a demand piece of
/// 4 is taken where a hint's is held back.
#[test]
fn a_demand_piece_joins_up_to_the_full_rate_window_and_others_to_the_urgent_one() {
    let volume = volume(9, 4, 8, 4);
    let piece = |start: u64, end: u64| Piece { start, end, parts: Vec::new() };
    for (priority, taken) in [(Priority::Prefetch, false), (Priority::Load, false), (Priority::Demand, true)] {
        let mut queue = Queue::default();
        queue.in_flight.insert(volume.key, VolumeFlight { bytes: 2, held_back: false, delivery: Arc::default() });
        let (mut job, _) = queued(0);
        job.volume = volume.clone();
        job.pieces = [piece(0, 4)].into();
        queue.push(job, priority);
        assert_eq!(queue.take().is_some(), taken, "{priority:?}");
    }
}

/// A hint a step comes to wait on joins as demand once promoted: held
/// back at the urgent window behind a demand piece, it is taken at the
/// full rate's window from its promotion, and a worker is woken for it.
#[test]
fn a_promoted_job_joins_at_its_new_prioritys_window() {
    let volume = volume(13, 4, 8, 4);
    let piece = |start: u64, end: u64| Piece { start, end, parts: Vec::new() };
    let mut queue = Queue::default();
    let (mut held, _) = queued(0);
    held.volume = volume.clone();
    held.pieces = [piece(0, 2)].into();
    queue.push(held, Priority::Demand);
    assert!(queue.take().is_some());
    let (mut hint, _) = queued(0);
    hint.volume = volume;
    hint.pieces = [piece(2, 6)].into();
    let (id, woken) = queue.push(hint, Priority::Prefetch);
    assert_eq!(woken, 0);
    assert!(queue.take().is_none(), "past the urgent window");
    assert_eq!(queue.promote(id, Priority::Demand), 1);
    assert!(queue.take().is_some(), "within the full rate's");
}

/// A load's import beside a serving reader puts no more ahead of its
/// decode request than the request's window: a bulk reader of the
/// volume (window 8) takes a load piece only within the volume's bound
/// (4, the urgent window chosen on it), so the urgent request's wait,
/// and its `T`, is what it was alone.
#[test]
fn a_load_beside_a_serving_reader_does_not_raise_the_urgent_requests_time() {
    let bound = Arc::<Bound>::default();
    let serving = Registered::new(bound.clone(), 4);
    let bulk = Volume { background: bound.clone(), ..volume(11, 8, 8, 4) };
    let piece = |start: u64, end: u64| Piece { start, end, parts: Vec::new() };
    let mut queue = Queue::default();
    queue.in_flight.insert(bulk.key, VolumeFlight { bytes: 2, held_back: false, delivery: Arc::default() });
    let (mut job, _) = queued(0);
    job.volume = bulk.clone();
    job.pieces = [piece(0, 4)].into();
    queue.push(job, Priority::Load);
    assert!(queue.take().is_none(), "a load piece joined past the urgent window");
    // Its demand pieces still read at its full-rate window.
    let (mut job, _) = queued(0);
    job.volume = bulk.clone();
    job.pieces = [piece(0, 4)].into();
    queue.push(job, Priority::Demand);
    assert!(queue.take().is_some());
    drop(serving);
    assert_eq!(bulk.window_for(Priority::Load), 8);
}

/// Through the readers themselves: an urgent reader opened on a volume
/// bounds a bulk reader's less urgent pieces by its window while it
/// lives (and its clones do), and once it drops, a bulk piece joins at
/// its full window again.
#[test]
fn an_urgent_reader_dropped_frees_its_volumes_bulk_pieces() {
    let _volume = temporary_volume();
    let path = temporary("bound", &vec![3u8; 1 << 20]);
    let bulk = ArtifactReader::open(&path, CachePolicy::Uncached).unwrap();
    let urgent = ArtifactReader::open_urgent(&path, CachePolicy::Uncached, &Urgent { parts: vec![4096], records: 1 }).unwrap();
    let clone = urgent.clone();
    assert_eq!(bulk.volume.window_for(Priority::Load), bulk.volume.choice.window.min(urgent.window()));
    drop(urgent);
    assert_eq!(bulk.volume.window_for(Priority::Load), bulk.volume.choice.window.min(clone.window()), "a clone holds its window");
    drop(clone);
    assert_eq!(bulk.volume.window_for(Priority::Load), bulk.volume.choice.window);
    std::fs::remove_file(path).unwrap();
}

/// The bound is the least window of the volume's live urgent readers:
/// two readers' windows hold it at the narrower; one dropping leaves the
/// other's; the last dropping frees a bulk piece to its full window.
#[test]
fn an_urgent_readers_window_bounds_its_volume_only_while_it_lives() {
    let bound = Arc::<Bound>::default();
    let bulk = Volume { background: bound.clone(), ..volume(12, 8, 8, 4) };
    assert_eq!(bulk.window_for(Priority::Load), 8, "no urgent reader: the full window");
    let narrow = Registered::new(bound.clone(), 2);
    let wide = Registered::new(bound.clone(), 4);
    let again = Registered::new(bound.clone(), 2);
    assert_eq!(bulk.window_for(Priority::Prefetch), 2);
    drop(narrow);
    assert_eq!(bulk.window_for(Priority::Prefetch), 2, "a window held twice stays until both drop");
    drop(again);
    assert_eq!(bulk.window_for(Priority::Prefetch), 4);
    drop(wide);
    assert_eq!(bulk.window_for(Priority::Load), 8, "a bulk piece joins at its full window again");
    assert_eq!(bulk.window_for(Priority::Demand), 8);
}

/// A job wakes a worker for each of its pieces its volume's window has
/// room for: none where it is full, one where nothing is in flight even
/// for a piece past the window.
#[test]
fn a_job_wakes_a_worker_for_each_piece_its_window_has_room_for() {
    let volume = volume(7, 4, 4, 2);
    let piece = |start: u64, end: u64| Piece { start, end, parts: Vec::new() };
    let mut queue = Queue::default();
    let (mut job, _) = queued(0);
    job.volume = volume.clone();
    job.pieces = [piece(0, 2), piece(2, 4), piece(4, 6)].into();
    assert_eq!(queue.room(&job, Priority::Demand), 2);
    job.pieces = [piece(0, 9)].into();
    assert_eq!(queue.room(&job, Priority::Demand), 1, "a piece past the window is taken alone");
    queue.in_flight.insert(volume.key, VolumeFlight { bytes: 3, held_back: false, delivery: Arc::default() });
    job.pieces = [piece(0, 2)].into();
    assert_eq!(queue.room(&job, Priority::Demand), 0, "a full window wakes none");
}

/// A volume's delivery counts every piece that ends and the time it
/// held one in flight: from a piece taken idle to the end that leaves it
/// idle, none while idle, and a busy span running at a take counted to
/// the take and on from there.
#[test]
fn a_volume_delivers_its_bytes_over_its_busy_time() {
    let delivery = Delivery::default();
    let at = Instant::now();
    let ms = |ms: u64| at + std::time::Duration::from_millis(ms);
    delivery.began(ms(0));
    delivery.ended(1_000_000, false, ms(4));
    delivery.ended(1_000_000, true, ms(8));
    // Idle 8 to 30 ms: no busy time.
    delivery.began(ms(30));
    delivery.ended(1_000_000, false, ms(34));
    let (bytes, seconds) = delivery.take_at(ms(36));
    assert_eq!(bytes, 3_000_000);
    assert!((seconds - 0.014).abs() < 1e-9, "{seconds}");
    delivery.ended(1_000_000, true, ms(40));
    let (bytes, seconds) = delivery.take_at(ms(50));
    assert_eq!(bytes, 1_000_000);
    assert!((seconds - 0.004).abs() < 1e-9, "{seconds}");
    assert_eq!(delivery.take_at(ms(60)), (0, 0.));
}

/// A volume holds at most its window's bytes in flight, or one piece
/// past it: its most urgent piece that would pass the window holds back
/// the volume's others, even smaller ones, while another volume's less
/// urgent pieces are taken; one ending says a held-back piece may be.
#[test]
fn a_volume_holds_at_most_its_window_in_flight() {
    let narrow = volume(1, 3, 3, 2);
    let piece = |start: u64, end: u64| Piece { start, end, parts: Vec::new() };
    let mut queue = Queue::default();
    let (mut urgent, _) = queued(0);
    urgent.volume = narrow.clone();
    urgent.pieces = [piece(0, 2), piece(2, 4)].into();
    let (mut small, _) = queued(0);
    small.volume = narrow.clone();
    small.pieces = [piece(8, 9)].into();
    let (other, _) = queued(1);
    let urgent_state = urgent.state.clone();
    queue.push(urgent, Priority::Demand);
    queue.push(small, Priority::Prefetch);
    queue.push(other, Priority::Load);
    let taken = |queue: &mut Queue| queue.take().map(|taken| (taken.2.key, taken.3.start));
    assert_eq!(taken(&mut queue), Some((VolumeKey(1), 0)));
    assert_eq!(taken(&mut queue), Some((VolumeKey(0), 0)), "2 + 2 passes the window; 2 + 1 would not, but waits");
    assert_eq!(taken(&mut queue), None);
    assert!(urgent_state.started.get().is_some(), "its first piece was taken");
    assert!(queue.ended(VolumeKey(1), 2, Instant::now()), "a piece was held back");
    assert_eq!(taken(&mut queue), Some((VolumeKey(1), 2)));
    assert_eq!(taken(&mut queue), Some((VolumeKey(1), 8)), "2 + 1 is within the window");
    assert!(!queue.ended(VolumeKey(0), 1, Instant::now()));
    // A piece wider than the window is read alone.
    let (mut wide, _) = queued(0);
    wide.volume = Volume { key: VolumeKey(2), delivery: Arc::default(), ..narrow };
    wide.pieces = [piece(0, 5)].into();
    queue.push(wide, Priority::Demand);
    assert_eq!(taken(&mut queue), Some((VolumeKey(2), 0)));
}
