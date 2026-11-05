//! The read engine against the device it reads, uncached: a decode step's
//! records read by the engine against raw `pread` threads at each read size
//! and queue depth, and one record's read alone by how it is issued (whole,
//! in pieces from spinning or woken threads, the engine). On a real model
//! file, the target of the acquisition object `MAGNITUDE_ARTIFACTS` names,
//! the measurements are timed, alone on the device (`-p magnitude-artifacts
//! --release --test read_depth -- --ignored --nocapture`); in the normal suite each
//! runs once on a small synthetic file of the same layout, its results
//! checked for shape, so an API change breaks it where it is made.

#![cfg(unix)]

use magnitude_artifacts::{ArtifactReader, CachePolicy, Priority, Range, Urgent, VolumeChoice};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::os::unix::fs::FileExt;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const ALIGN: u64 = 4096;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// `count` aligned sinks of `bytes` each.
struct Sinks {
    base: NonNull<u8>,
    layout: Layout,
    stride: usize,
}
impl Sinks {
    fn new(count: usize, bytes: usize) -> Self {
        let stride = bytes.next_multiple_of(ALIGN as usize);
        let layout = Layout::from_size_align(count * stride, ALIGN as usize).unwrap();
        Self { base: NonNull::new(unsafe { alloc_zeroed(layout) }).unwrap(), layout, stride }
    }
    fn sink(&self, index: usize) -> NonNull<u8> {
        unsafe { self.base.add(index * self.stride) }
    }
}
impl Drop for Sinks {
    fn drop(&mut self) {
        unsafe { dealloc(self.base.as_ptr(), self.layout) }
    }
}

/// `path` opened for reads the page cache does not serve: `O_DIRECT` on
/// Linux (through the cache where its filesystem refuses direct I/O, a
/// tmpfs temporary directory's), `F_NOCACHE` on macOS.
fn uncached(path: &str) -> std::fs::File {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "linux")]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut options, libc::O_DIRECT);
    let file = options.open(path).unwrap_or_else(|_| std::fs::File::open(path).unwrap());
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: an open descriptor; F_NOCACHE takes an int.
        assert_ne!(unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) }, -1);
    }
    file
}

/// A model file and its experts' stored layout: each routed layer's stacks
/// (each `_exps` tensor's offset and one expert's bytes) and the experts a
/// stack holds.
struct Model {
    path: String,
    size: u64,
    layers: Vec<Vec<(u64, u64)>>,
    experts: u64,
    /// Removed when the model drops: a synthetic file's.
    temporary: bool,
}

impl Model {
    /// The model file `MAGNITUDE_ARTIFACTS` names, its stacks from its
    /// GGUF tensor table (`blk.N.` layers' `_exps` tensors), only those
    /// that lie in this file (a sharded model's others address later
    /// shards).
    fn acquired() -> Self {
        let artifacts = std::env::var_os("MAGNITUDE_ARTIFACTS").expect("set MAGNITUDE_ARTIFACTS");
        let acquisition: serde_json::Value = serde_json::from_slice(&std::fs::read(&artifacts).unwrap()).unwrap();
        let path = acquisition["target"].as_str().unwrap().to_owned();
        let artifact = magnitude_artifacts::gguf::GgufArtifact::open(&path).unwrap();
        let mut layers = std::collections::BTreeMap::<u64, std::collections::BTreeMap<String, (u64, u64)>>::new();
        let mut experts = 0;
        for name in artifact.directory().tensors.iter().map(|tensor| tensor.name.clone()).collect::<Vec<_>>() {
            let Some((layer, tensor)) = name.strip_prefix("blk.").and_then(|rest| rest.split_once('.')) else { continue };
            if !tensor.ends_with("_exps.weight") {
                continue;
            }
            let slice = artifact.tensor(&name).unwrap();
            experts = *slice.shape.last().unwrap();
            layers.entry(layer.parse().unwrap()).or_default().insert(tensor.to_owned(), (slice.offset, slice.nbytes / experts));
        }
        let size = std::fs::metadata(&path).unwrap().len();
        let layers = layers
            .into_values()
            .map(|stacks| stacks.into_values().collect::<Vec<_>>())
            .filter(|stacks| stacks.iter().all(|&(offset, per)| offset + per * experts <= size))
            .collect::<Vec<_>>();
        assert!(!layers.is_empty(), "{path} has no expert stacks");
        Self { path, size, layers, experts, temporary: false }
    }

    /// A small file of two layers, each three stacks of four experts of
    /// unequal unaligned bytes, as a GGUF lays them: what the suite runs
    /// the measurements on once.
    fn synthetic(name: &str) -> Self {
        let (experts, parts) = (4u64, [36_864u64 + 32, 36_864 + 64, 45_056 + 96]);
        let mut layers = Vec::new();
        let mut at = 4096 + 32;
        for _ in 0..2 {
            let mut stacks = Vec::new();
            for per in parts {
                stacks.push((at, per));
                at += per * experts;
            }
            layers.push(stacks);
        }
        let size = (at + (4 << 20)).next_multiple_of(ALIGN);
        let path = std::env::temp_dir().join(format!("magnitude-read-depth-{name}-{}", std::process::id()));
        std::fs::write(&path, vec![7u8; size as usize]).unwrap();
        Self { path: path.to_str().unwrap().to_owned(), size, layers, experts, temporary: true }
    }

    /// One record's bytes: the first layer's stacks' experts together.
    fn record(&self) -> usize {
        self.layers[0].iter().map(|&(_, per)| per as usize).sum()
    }

    /// The decode request the engine is read for: one row's experts, each
    /// a record of the first layer's stored ranges.
    fn urgent(&self) -> Urgent {
        Urgent { parts: self.layers[0].iter().map(|&(_, per)| per).collect(), records: 1 }
    }

    /// `experts` random experts' stored ranges of one random layer.
    fn request(&self, rng: &mut Rng, experts: usize) -> Vec<(u64, usize)> {
        let layer = &self.layers[(rng.next() % self.layers.len() as u64) as usize];
        (0..experts)
            .flat_map(|_| {
                let expert = rng.next() % self.experts;
                layer.iter().map(move |&(offset, per)| (offset + expert * per, per as usize))
            })
            .collect()
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if self.temporary {
            std::fs::remove_file(&self.path).unwrap();
        }
    }
}

/// Raw: `depth` threads each reading `bytes` at random aligned offsets,
/// uncached, for `seconds`: bytes a second.
fn raw(model: &Model, bytes: u64, depth: usize, seconds: f64) -> f64 {
    let stop = Arc::new(AtomicBool::new(false));
    let moved = Arc::new(AtomicU64::new(0));
    let slots = model.size / ALIGN - bytes / ALIGN;
    let threads = (0..depth as u64)
        .map(|thread| {
            let (stop, moved, path) = (stop.clone(), moved.clone(), model.path.clone());
            std::thread::spawn(move || {
                let file = uncached(&path);
                let mut buffer = vec![0u8; (bytes + ALIGN) as usize];
                let at = buffer.as_ptr().align_offset(ALIGN as usize);
                let buffer = &mut buffer[at..at + bytes as usize];
                let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ (thread + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9));
                while !stop.load(Ordering::Relaxed) {
                    file.read_exact_at(buffer, rng.next() % slots * ALIGN).unwrap();
                    moved.fetch_add(bytes, Ordering::Relaxed);
                }
            })
        })
        .collect::<Vec<_>>();
    let began = Instant::now();
    std::thread::sleep(Duration::from_secs_f64(seconds));
    stop.store(true, Ordering::Relaxed);
    for thread in threads {
        thread.join().unwrap();
    }
    moved.load(Ordering::Relaxed) as f64 / began.elapsed().as_secs_f64()
}

/// The engine: `steps` decode-like steps, each `records` records of
/// `record` bytes at random offsets in jobs of `per_job`, all submitted at
/// once, then waited for: the mean step's seconds.
fn engine(reader: &ArtifactReader, sinks: &Sinks, record: usize, rng: &mut Rng, records: usize, per_job: usize, steps: usize) -> f64 {
    let mut total = 0.0;
    for _ in 0..steps {
        let ranges = (0..records)
            .map(|index| Range { offset: rng.next() % (reader.size() - record as u64), length: record, sink: sinks.sink(index) })
            .collect::<Vec<_>>();
        let began = Instant::now();
        let jobs = ranges
            .chunks(per_job)
            .map(|chunk| unsafe { reader.submit(chunk.to_vec(), Priority::Demand) }.unwrap())
            .collect::<Vec<_>>();
        for job in jobs {
            assert!(matches!(job.wait(), magnitude_artifacts::Ended::Read { failed } if failed.is_empty()));
        }
        total += began.elapsed().as_secs_f64();
    }
    total / steps as f64
}

/// What a rate measurement reads: steps of each (records, per job) the
/// engine reads, and the raw read sizes, depths and seconds.
struct Rates {
    alone_steps: usize,
    batches: &'static [(usize, usize)],
    batch_steps: usize,
    raw_bytes: &'static [u64],
    raw_depths: &'static [usize],
    raw_seconds: f64,
}

/// The engine's reads of `model` against raw threads' (`rates`): each line,
/// labelled, with its value (seconds for the engine's, bytes a second for
/// raw).
fn rates(model: &Model, rates: &Rates) -> Vec<(String, f64)> {
    let urgent = model.urgent();
    let source = magnitude_artifacts::FileSource::open(&model.path).unwrap();
    let opened = Instant::now();
    let choice = magnitude_artifacts::probe_volume(&source, Some(&urgent)).unwrap();
    let probed = opened.elapsed();
    let reader = ArtifactReader::open_urgent(&model.path, CachePolicy::Uncached, &urgent).unwrap();
    let cores = magnitude_artifacts::read_cores([&source], &urgent).unwrap();
    eprintln!(
        "probe: {:.0} ms, window {} KiB (demand {} KiB), read {} KiB, {cores:.2} cores",
        probed.as_secs_f64() * 1e3,
        choice.window >> 10,
        choice.demand >> 10,
        choice.read >> 10,
    );
    let record = model.record();
    let widest = rates.batches.iter().map(|&(records, _)| records).max().unwrap_or(1);
    let sinks = Sinks::new(widest, record);
    let mut rng = Rng(0x5eed);
    // Every line reads the device, not the page cache: uncached reads alone
    // are served from pages already cached (macOS's `F_NOCACHE`), so each
    // line starts with the file's pages evicted (`ArtifactReader::evict_cached`).
    let evicted = || reader.evict_cached().unwrap();
    evicted();
    let mut lines = vec![("one record alone".to_owned(), engine(&reader, &sinks, record, &mut rng, 1, 1, rates.alone_steps))];
    for &(records, per_job) in rates.batches {
        evicted();
        lines.push((format!("{records} records in jobs of {per_job}"), engine(&reader, &sinks, record, &mut rng, records, per_job, rates.batch_steps)));
    }
    for &bytes in rates.raw_bytes {
        for &depth in rates.raw_depths {
            evicted();
            lines.push((format!("raw {} KiB at depth {depth}", bytes >> 10), raw(model, bytes, depth, rates.raw_seconds)));
        }
    }
    for (label, value) in &lines {
        eprintln!("{label}: {value:.6}");
    }
    lines
}

#[test]
#[ignore = "times the read engine against the device on a real model file (`MAGNITUDE_ARTIFACTS`)"]
fn the_engine_reads_at_the_devices_rate() {
    let model = Model::acquired();
    rates(
        &model,
        &Rates {
            alone_steps: 200,
            batches: &[(16, 16), (175, 5), (175, 175)],
            batch_steps: 20,
            raw_bytes: &[128 << 10, 256 << 10, 512 << 10, 1 << 20, 2 << 20],
            raw_depths: &[4, 8, 16, 32, 64, 128],
            raw_seconds: 0.75,
        },
    );
}

/// A pool of threads reading a lone request's reads, each read taken by
/// the next free thread, which either spins for the next request or sleeps
/// on a condition variable until it comes.
struct Pool {
    shared: Arc<Shared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

struct Shared {
    file: std::fs::File,
    sleeping: bool,
    generation: AtomicU64,
    reads: std::sync::Mutex<Arc<Vec<(u64, usize)>>>,
    woken: std::sync::Condvar,
    next: std::sync::atomic::AtomicUsize,
    done: std::sync::atomic::AtomicUsize,
    stop: AtomicBool,
}

impl Pool {
    /// `workers` threads, each with a buffer of `widest` bytes.
    fn new(path: &str, workers: usize, widest: usize, sleeping: bool) -> Self {
        let shared = Arc::new(Shared {
            file: uncached(path),
            sleeping,
            generation: AtomicU64::new(0),
            reads: std::sync::Mutex::new(Arc::new(Vec::new())),
            woken: std::sync::Condvar::new(),
            next: Default::default(),
            done: Default::default(),
            stop: AtomicBool::new(false),
        });
        let threads = (0..workers)
            .map(|_| {
                let shared = shared.clone();
                std::thread::spawn(move || {
                    let mut buffer = vec![0u8; widest + ALIGN as usize];
                    let aligned = buffer.as_ptr().align_offset(ALIGN as usize);
                    let mut seen = 0;
                    loop {
                        if shared.sleeping {
                            let guard = shared.reads.lock().unwrap();
                            drop(shared.woken.wait_while(guard, |_| {
                                shared.generation.load(Ordering::Acquire) == seen && !shared.stop.load(Ordering::Relaxed)
                            }));
                        } else {
                            while shared.generation.load(Ordering::Acquire) == seen && !shared.stop.load(Ordering::Relaxed) {
                                std::hint::spin_loop();
                            }
                        }
                        if shared.stop.load(Ordering::Relaxed) {
                            return;
                        }
                        seen = shared.generation.load(Ordering::Acquire);
                        let reads = shared.reads.lock().unwrap().clone();
                        loop {
                            let index = shared.next.fetch_add(1, Ordering::AcqRel);
                            let Some(&(at, length)) = reads.get(index) else { break };
                            shared.file.read_exact_at(&mut buffer[aligned..aligned + length], at).unwrap();
                            shared.done.fetch_add(1, Ordering::AcqRel);
                        }
                    }
                })
            })
            .collect();
        Self { shared, threads }
    }

    /// Read `reads` once, all issued at once: the seconds until the last
    /// ends.
    fn read(&self, reads: Vec<(u64, usize)>) -> f64 {
        let count = reads.len();
        let began = Instant::now();
        {
            let mut held = self.shared.reads.lock().unwrap();
            *held = Arc::new(reads);
            self.shared.next.store(0, Ordering::Release);
            self.shared.done.store(0, Ordering::Release);
            self.shared.generation.fetch_add(1, Ordering::AcqRel);
        }
        if self.shared.sleeping {
            for _ in 0..count.min(self.threads.len()) {
                self.shared.woken.notify_one();
            }
        }
        while self.shared.done.load(Ordering::Acquire) < count {
            std::hint::spin_loop();
        }
        began.elapsed().as_secs_f64()
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        {
            let _held = self.shared.reads.lock().unwrap();
            self.shared.woken.notify_all();
        }
        for thread in self.threads.drain(..) {
            thread.join().unwrap();
        }
    }
}

/// `ranges` as aligned reads split at `split` bytes (`None`: each range
/// one read).
fn split(ranges: &[(u64, usize)], split: Option<u64>) -> Vec<(u64, usize)> {
    let mut reads = Vec::new();
    for &(offset, length) in ranges {
        let start = offset / ALIGN * ALIGN;
        let end = (offset + length as u64).div_ceil(ALIGN) * ALIGN;
        let step = split.unwrap_or(end - start);
        let mut at = start;
        while at < end {
            let stop = (at + step).min(end);
            reads.push((at, (stop - at) as usize));
            at = stop;
        }
    }
    reads
}

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// The splits a lone request's reads are timed at.
const SPLITS: [Option<u64>; 6] = [Some(128 << 10), Some(256 << 10), Some(512 << 10), Some(1 << 20), Some(2 << 20), None];

/// A lone request's read of `model`, by how it is split and issued: one
/// record contiguous, one expert's stored ranges, and four experts' (a
/// decode layer's misses), each at each split, from `workers` threads
/// spinning for it or woken for it; and the engine. The median seconds of
/// `samples`, labelled.
fn lone(model: &Model, samples: usize, workers: usize) -> Vec<(String, f64)> {
    let reader = ArtifactReader::open_urgent(&model.path, CachePolicy::Uncached, &model.urgent()).unwrap();
    let (size, record) = (model.size, model.record());
    let mut rng = Rng(0xa10e);
    let kinds: [(&str, Box<dyn Fn(&mut Rng) -> Vec<(u64, usize)>>); 3] = [
        ("record", Box::new(move |rng: &mut Rng| vec![(rng.next() % ((size - record as u64) / ALIGN) * ALIGN, record)])),
        ("expert", Box::new(|rng: &mut Rng| model.request(rng, 1))),
        ("4 experts", Box::new(|rng: &mut Rng| model.request(rng, 4))),
    ];
    let widest = 4 * model.layers.iter().flatten().map(|&(_, per)| per as usize).max().unwrap() + 2 * ALIGN as usize;
    let mut lines = Vec::new();
    // Each line reads the device, its pages evicted first (as `rates`).
    let evicted = || reader.evict_cached().unwrap();
    for sleeping in [false, true] {
        let pool = Pool::new(&model.path, workers, widest.max(record + 2 * ALIGN as usize), sleeping);
        for (kind, request) in &kinds {
            for at in SPLITS {
                evicted();
                let time = median((0..samples).map(|_| pool.read(split(&request(&mut rng), at))).collect());
                let split = at.map_or("whole".into(), |at| format!("{}K", at >> 10));
                lines.push((format!("lone {kind}, threads {}, split {split}", if sleeping { "woken" } else { "spinning" }), time));
            }
        }
    }
    let sinks = Sinks::new(12, widest);
    for (kind, request) in &kinds {
        evicted();
        let time = median(
            (0..samples)
                .map(|_| {
                    let ranges = request(&mut rng)
                        .into_iter()
                        .enumerate()
                        .map(|(index, (offset, length))| Range { offset, length, sink: sinks.sink(index) })
                        .collect();
                    let began = Instant::now();
                    let ended = unsafe { reader.submit(ranges, Priority::Demand) }.unwrap().wait();
                    assert!(matches!(ended, magnitude_artifacts::Ended::Read { failed } if failed.is_empty()));
                    began.elapsed().as_secs_f64()
                })
                .collect(),
        );
        lines.push((format!("lone {kind}, the engine"), time));
    }
    // The engine's demand read size, each at a probe's windows: the sizes the
    // threads above were split at, through the engine's own queue and buffers.
    let source = magnitude_artifacts::FileSource::open(&model.path).unwrap();
    for demand_read in SPLITS.into_iter().flatten() {
        let choice = VolumeChoice { window: 4 << 20, demand: 8 << 20, read: 128 << 10, demand_read, cores: 1. };
        magnitude_artifacts::adopt_volume(&source, Some(&model.urgent()), choice).unwrap();
        let sized = ArtifactReader::of_urgent(&source, CachePolicy::Uncached, &model.urgent()).unwrap();
        for (kind, request) in &kinds {
            evicted();
            let time = median(
                (0..samples)
                    .map(|_| {
                        let ranges = request(&mut rng)
                            .into_iter()
                            .enumerate()
                            .map(|(index, (offset, length))| Range { offset, length, sink: sinks.sink(index) })
                            .collect();
                        let began = Instant::now();
                        let ended = unsafe { sized.submit(ranges, Priority::Demand) }.unwrap().wait();
                        assert!(matches!(ended, magnitude_artifacts::Ended::Read { failed } if failed.is_empty()));
                        began.elapsed().as_secs_f64()
                    })
                    .collect(),
            );
            lines.push((format!("lone {kind}, the engine, demand read {} KiB", demand_read >> 10), time));
        }
    }
    for (label, seconds) in &lines {
        eprintln!("{label}: {:.3} ms", seconds * 1e3);
    }
    lines
}

#[test]
#[ignore = "times lone requests on a real model file (`MAGNITUDE_ARTIFACTS`)"]
fn a_lone_request() {
    lone(&Model::acquired(), 200, 64);
}

/// Each measurement once on a small synthetic file of a model's layout:
/// every line it reports, each a finite positive time or rate. Its values
/// are not judged; its shape is, so a change to what it calls breaks it in
/// the suite.
#[test]
fn the_measurements_run_once_on_a_small_file() {
    let model = Model::synthetic("smoke");
    let lines = rates(
        &model,
        &Rates { alone_steps: 1, batches: &[(4, 2)], batch_steps: 1, raw_bytes: &[128 << 10], raw_depths: &[2], raw_seconds: 0.01 },
    );
    assert_eq!(lines.len(), 3, "the record alone, one batch, one raw line");
    let lone = lone(&model, 1, 4);
    assert_eq!(lone.len(), 2 * 3 * SPLITS.len() + 3 + 3 * (SPLITS.len() - 1), "each issue, kind and split, the engine's each kind, and at each split's demand read size");
    for (label, value) in lines.iter().chain(&lone) {
        assert!(value.is_finite() && *value > 0., "{label}: {value}");
    }
}
