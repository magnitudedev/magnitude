//! What the read engine delivers of a volume against the drive read raw, at
//! a streamed-expert decode's pattern: reads of 1 MiB (a range, about a
//! third of a record) and of one whole 3.09 MB record, at random offsets of
//! a model's files, past the file cache, at each depth in flight. Each depth
//! is held four ways, every cell reading distinct bytes:
//!
//! - `raw`: as many threads as the depth, each reading positionally at
//!   16 KiB-aligned offsets into a 16 KiB-aligned buffer (the drive's
//!   ceiling run's pattern);
//! - `raw4k`: the same, each offset 4 KiB past a 16 KiB boundary, as the
//!   engine's 4 KiB-aligned reads of a model's ranges mostly are;
//! - `engine`: one submitter keeping more requests queued than the link
//!   admits, the link's due depth that many reads, at `raw`'s offsets: what
//!   the engine itself costs a read that lands straight;
//! - `grain`: the same at offsets of a model file's 32-byte grain, each
//!   piece read into a worker's buffer and copied out, as a decode's ranges
//!   are.
//!
//! And one decode park repeated (`burst`): 16 ranges of a third of a record
//! at the grain submitted at once, due, on the link a decode measured
//! (due 32 MiB in 2 MiB reads, knee 2600 KiB), waited for, then the next.
//!
//! Each byte is read at most once (offsets drawn without replacement from
//! disjoint slots), every file's cached pages are dropped before each pass,
//! and a sample of the slots about to be read, and of those read, is checked
//! uncached (`mincore`). Rates are requested bytes over wall time, the median
//! of two passes in opposite order, with their half range.
//!
//! Run alone on the device, `READ_PATH_DIR` the directory of the model's
//! files (its `*-of-*.gguf` shards), `READ_PATH_GB` each cell's bytes (0.5):
//! `cargo test -p seismic-io --release --test read_path -- --ignored --nocapture`.
#![cfg(target_os = "macos")]
#![allow(
    clippy::disallowed_methods,
    reason = "a test starts threads of its own"
)]

use seismic_io::{adopt_earlier, cached_pages, Buffer, CachePolicy, EarlierChoice, Ended, Handle, OpenFile, Range, Reader, Want, ALIGNMENT};
use std::collections::VecDeque;
use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// One whole expert record of Qwen3.8 Flash Next (755 × 4 KiB), and 1 MiB.
const RECORD: u64 = 3_092_480;
const MIB: u64 = 1 << 20;
const SIZES: [u64; 2] = [MIB, RECORD];
const DEPTHS: [usize; 5] = [1, 2, 3, 4, 8];
/// The host's page: the raw run's offset and buffer alignment.
const PAGE: u64 = 16 * 1024;
/// A model file stores tensors at this grain.
const GRAIN: u64 = 32;
/// A slot holds any read of a cell with room to misalign it.
const SLOT: u64 = RECORD.next_multiple_of(PAGE) + 2 * PAGE;
/// A decode park: its ranges, each a third of a record at the grain.
const BURST: usize = 16;
const BURST_RANGE: u64 = RECORD / 3 / GRAIN * GRAIN;
const PASSES: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Way {
    Raw,
    Raw4k,
    Engine,
    Grain,
}

impl Way {
    fn name(self) -> &'static str {
        match self {
            Way::Raw => "raw",
            Way::Raw4k => "raw4k",
            Way::Engine => "engine",
            Way::Grain => "grain",
        }
    }

    /// Where a read of `size` lies in the slot at `base`: a random page of
    /// it, moved 4 KiB past the page or to a random grain of it.
    fn offset(self, base: u64, size: u64, random: &mut Random) -> u64 {
        let page = base + random.below((SLOT - size - PAGE) / PAGE + 1) * PAGE;
        match self {
            Way::Raw | Way::Engine => page,
            Way::Raw4k => page + 4096,
            Way::Grain => page + 4096 + random.below((PAGE - 4096) / GRAIN) * GRAIN,
        }
    }
}

struct Random(u64);

impl Random {
    fn below(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % bound
    }
}

/// A page-aligned heap buffer for the raw readers.
struct Aligned(*mut u8, std::alloc::Layout);

unsafe impl Send for Aligned {}

impl Aligned {
    fn new(bytes: u64) -> Self {
        let layout = std::alloc::Layout::from_size_align(bytes as usize, PAGE as usize).unwrap();
        // SAFETY: a nonzero layout.
        Self(unsafe { std::alloc::alloc_zeroed(layout) }, layout)
    }

    fn bytes(&mut self, length: u64) -> &mut [u8] {
        // SAFETY: within its allocation, held alone.
        unsafe { std::slice::from_raw_parts_mut(self.0, length as usize) }
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout.
        unsafe { std::alloc::dealloc(self.0, self.1) }
    }
}

struct Files {
    paths: Vec<PathBuf>,
    opened: Vec<OpenFile>,
    /// Each file opened past the cache (`F_NOCACHE`), for the raw readers.
    raw: Vec<File>,
    /// Every slot, shuffled, and those taken.
    slots: Vec<(usize, u64)>,
    taken: usize,
}

impl Files {
    fn open(directory: &str) -> Self {
        let mut paths = std::fs::read_dir(directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.contains("-of-") && name.ends_with(".gguf")))
            .collect::<Vec<_>>();
        paths.sort();
        assert!(!paths.is_empty(), "no *-of-*.gguf shards in {directory}");
        let opened = paths.iter().map(|path| OpenFile::open(path).unwrap()).collect::<Vec<OpenFile>>();
        let raw = paths
            .iter()
            .map(|path| {
                let file = File::open(path).unwrap();
                // SAFETY: a flag on a descriptor this test owns.
                assert_ne!(unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) }, -1);
                file
            })
            .collect();
        let mut slots = opened
            .iter()
            .enumerate()
            .flat_map(|(index, file)| (0..file.size() / SLOT).map(move |slot| (index, slot * SLOT)))
            .collect::<Vec<_>>();
        let mut random = Random(0x5eed_5eed_5eed_5eed);
        for at in (1..slots.len()).rev() {
            slots.swap(at, random.below(at as u64 + 1) as usize);
        }
        Self { paths, opened, raw, slots, taken: 0 }
    }

    fn take(&mut self, count: usize) -> Vec<(usize, u64)> {
        assert!(self.taken + count <= self.slots.len(), "{} slots left, {count} wanted: lower READ_PATH_GB", self.slots.len() - self.taken);
        self.taken += count;
        self.slots[self.taken - count..self.taken].to_vec()
    }

    /// Drop every file's cached pages.
    fn evict(&self) {
        for file in &self.opened {
            Reader::of(file, CachePolicy::Uncached).unwrap().evict_cached().unwrap();
        }
    }

    /// The share of the pages of `reads` the file cache holds, of a sample.
    fn cached(&self, reads: &[(usize, u64, u64)]) -> f64 {
        let step = (reads.len() / 256).max(1);
        let (mut cached, mut pages) = (0, 0);
        for &(file, offset, length) in reads.iter().step_by(step) {
            let (held, all) = cached_pages(&self.paths[file], offset, length).unwrap();
            cached += held;
            pages += all;
        }
        cached as f64 / pages.max(1) as f64
    }

    /// Readers of every file on a link of `choice`.
    fn readers(&self, choice: EarlierChoice) -> Vec<Reader> {
        for file in &self.opened {
            adopt_earlier(file, None, choice).unwrap();
        }
        self.opened.iter().map(|file| Reader::of(file, CachePolicy::Uncached).unwrap()).collect()
    }
}

/// `reads` of `size` read raw by `depth` threads: bytes a second.
fn raw(files: &Files, reads: &[(usize, u64)], size: u64, depth: usize) -> f64 {
    let next = AtomicUsize::new(0);
    let began = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..depth {
            scope.spawn(|| {
                let mut buffer = Aligned::new(size.next_multiple_of(PAGE));
                while let Some(&(file, offset)) = reads.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let into = buffer.bytes(size);
                    let mut done = 0;
                    while done < into.len() {
                        done += files.raw[file].read_at(&mut into[done..], offset + done as u64).unwrap();
                    }
                }
            });
        }
    });
    (reads.len() as u64 * size) as f64 / began.elapsed().as_secs_f64()
}

fn ended(handle: Handle) {
    match handle.wait() {
        Ended::Read { failed } => assert!(failed.is_empty(), "a read failed: {failed:?}"),
        Ended::Panicked(panic) => panic.resume(),
    }
}

/// `reads` of `size` through the engine, its link's due depth `depth`
/// reads, one submitter keeping twice that and two more queued: bytes a
/// second.
fn engine(files: &Files, reads: &[(usize, u64)], size: u64, depth: usize) -> f64 {
    // The widest piece a read of `size` at any offset is cut in: one piece
    // a read, `depth` of them in flight.
    let piece = size.next_multiple_of(ALIGNMENT) + ALIGNMENT;
    let due = depth as u64 * piece;
    let readers = files.readers(EarlierChoice { window: due, demand: due, read: piece, demand_read: piece, cores: 0. });
    let queued = 2 * depth + 2;
    let mut spare = (0..queued).map(|_| Arc::new(Buffer::new(size as usize))).collect::<Vec<_>>();
    let mut flight = VecDeque::new();
    let began = Instant::now();
    for &(file, offset) in reads {
        if flight.len() == queued {
            let (handle, buffer) = flight.pop_front().unwrap();
            ended(handle);
            spare.push(buffer);
        }
        let buffer = spare.pop().unwrap();
        let range = Range { offset, length: size as usize, into: buffer.clone(), at: 0 };
        // SAFETY: each buffer is one request's alone until it has ended.
        flight.push_back((unsafe { readers[file].request(vec![range], Want::now()) }.unwrap(), buffer));
    }
    for (handle, _) in flight {
        ended(handle);
    }
    (reads.len() as u64 * size) as f64 / began.elapsed().as_secs_f64()
}

/// `reads` of `BURST_RANGE` through a decode's link, `BURST` at once, each
/// burst waited for before the next: bytes a second.
fn burst(files: &Files, reads: &[(usize, u64)]) -> f64 {
    let readers = files.readers(EarlierChoice { window: 2600 * 1024, demand: 32 * MIB, read: 2 * MIB, demand_read: 2 * MIB, cores: 0. });
    let buffers = (0..BURST).map(|_| Arc::new(Buffer::new(BURST_RANGE as usize))).collect::<Vec<_>>();
    let began = Instant::now();
    for burst in reads.chunks(BURST) {
        let handles = burst
            .iter()
            .zip(&buffers)
            .map(|(&(file, offset), buffer)| {
                let range = Range { offset, length: BURST_RANGE as usize, into: buffer.clone(), at: 0 };
                // SAFETY: each buffer is one request's alone until its burst ends.
                unsafe { readers[file].request(vec![range], Want::now()) }.unwrap()
            })
            .collect::<Vec<_>>();
        handles.into_iter().for_each(ended);
    }
    (reads.len() as u64 * BURST_RANGE) as f64 / began.elapsed().as_secs_f64()
}

#[test]
#[ignore = "reads a model's files past the cache (`READ_PATH_DIR`)"]
fn the_engine_reads_at_the_drives_rate() {
    let directory = std::env::var("READ_PATH_DIR").expect("READ_PATH_DIR: the directory of a model's files");
    let per_cell = (std::env::var("READ_PATH_GB").map_or(0.5, |gb| gb.parse::<f64>().unwrap()) * 1e9) as u64;
    let mut files = Files::open(&directory);
    let ways = [Way::Raw, Way::Raw4k, Way::Engine, Way::Grain];
    let mut grid = Vec::new();
    for size in SIZES {
        for depth in DEPTHS {
            for way in ways {
                grid.push(Some((way, size, depth)));
            }
        }
    }
    grid.push(None);
    let total_slots = files.slots.len();
    println!("{} files, {:.1} GB, {total_slots} slots; ALIGNMENT {ALIGNMENT}", files.paths.len(), files.opened.iter().map(OpenFile::size).sum::<u64>() as f64 / 1e9);
    let mut rates = vec![Vec::new(); grid.len()];
    let mut read = Vec::new();
    let mut random = Random(0x2545_f491_4f6c_dd1d);
    for pass in 0..PASSES {
        files.evict();
        let ahead = files.slots[files.taken..(files.taken + 2048).min(total_slots)].iter().map(|&(file, base)| (file, base, SLOT)).collect::<Vec<_>>();
        println!("pass {}: cached share of slots about to be read {:.2}%", pass + 1, 100. * files.cached(&ahead));
        let order = (0..grid.len()).collect::<Vec<_>>();
        let order = if pass % 2 == 0 { order } else { order.into_iter().rev().collect() };
        for at in order {
            let (rate, label) = match grid[at] {
                Some((way, size, depth)) => {
                    let reads = files.take(per_cell.div_ceil(size) as usize).into_iter().map(|(file, base)| (file, way.offset(base, size, &mut random))).collect::<Vec<_>>();
                    let rate = match way {
                        Way::Raw | Way::Raw4k => raw(&files, &reads, size, depth),
                        Way::Engine | Way::Grain => engine(&files, &reads, size, depth),
                    };
                    read.extend(reads.iter().map(|&(file, offset)| (file, offset, size)));
                    (rate, format!("{:<6} {:.2} MB d{depth}", way.name(), size as f64 / 1e6))
                }
                None => {
                    let count = (per_cell.div_ceil(BURST_RANGE) as usize).next_multiple_of(BURST);
                    let reads = files.take(count).into_iter().map(|(file, base)| (file, Way::Grain.offset(base, BURST_RANGE, &mut random))).collect::<Vec<_>>();
                    let rate = burst(&files, &reads);
                    read.extend(reads.iter().map(|&(file, offset)| (file, offset, BURST_RANGE)));
                    (rate, format!("burst  {BURST} x {:.2} MB", BURST_RANGE as f64 / 1e6))
                }
            };
            rates[at].push(rate);
            println!("  pass {}  {label}  {:.2} GB/s", pass + 1, rate / 1e9);
        }
    }
    println!("cached share of pages read, at end: {:.2}% (expect ~0: F_NOCACHE, no byte read twice)", 100. * files.cached(&read));
    println!();
    let cell = |at: usize| {
        let mut sorted = rates[at].clone();
        sorted.sort_by(f64::total_cmp);
        let (least, most) = (sorted[0], sorted[sorted.len() - 1]);
        format!("{:5.2}+-{:.2}", (least + most) / 2e9, (most - least) / 2e9)
    };
    for size in SIZES {
        println!("{:.2} MB reads (GB/s, mean of passes +- half range)", size as f64 / 1e6);
        println!("depth  {}", ways.iter().map(|way| format!("{:>11}", way.name())).collect::<String>());
        for depth in DEPTHS {
            let row = ways
                .iter()
                .map(|&way| format!("{:>11}", cell(grid.iter().position(|cell| *cell == Some((way, size, depth))).unwrap())))
                .collect::<String>();
            println!("d{depth:<5} {row}");
        }
    }
    println!("burst  {BURST} x {:.2} MB at once, each waited: {}", BURST_RANGE as f64 / 1e6, cell(grid.len() - 1));
}
