//! `machine`: the host CPU's own line, which a streamed `time` run prices a
//! CPU-computed miss by ([`crate::bar`]): its int8 peak as the measurement
//! job stored it for the host's CPU executor (`MachineLine`, one owner), or
//! unmeasured with why; and the storage device's own lines
//! ([`StorageMachine`]).

use magnitude_engine::worker::measure::InProcessJobs;
use magnitude_executor::assessment::{MachineLine, MachineLineError, Operand, Peak, Rate};
use magnitude_executor::KernelCache;
use serde::Serialize;
use std::path::Path;
use std::sync::{Arc, Barrier};
use std::time::Instant;

/// The host CPU's line over the cores the process may use.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CpuMachine {
    /// The CPUs the process may run on (`available_parallelism`, which
    /// follows the affinity mask and quota).
    pub threads: usize,
    /// int8 operations a second (a multiply and an add each) the CPU
    /// executor's stored line reaches on its quantized dots, scaled with
    /// its work.
    pub int8_ops_per_second: Result<f64, String>,
}

/// The seconds a rate is measured over: long past a thread's start, and
/// short beside a load.
const SECONDS: f64 = 0.25;

impl CpuMachine {
    /// The host's line from the records `cache` holds of its CPU executor,
    /// which the measurement job measures where they are missing: the job
    /// of the executor, in this process, as any load's job is, so the
    /// figure is never left to a manual step.
    pub fn measured(cache: Option<&Path>, storage: &Path) -> Result<Self, String> {
        let threads = std::thread::available_parallelism().map_err(|error| error.to_string())?.get();
        let int8_ops_per_second = match cache {
            Some(cache) => Self::int8(cache, storage),
            None => Err("no kernel cache holds the CPU's line (--cache-dir)".to_owned()),
        };
        Ok(Self { threads, int8_ops_per_second })
    }

    fn int8(cache: &Path, storage: &Path) -> Result<f64, String> {
        let catalog = seismic::Devices::discover(&[]).map_err(|error| error.to_string())?;
        let topology = catalog.topology();
        let executor = topology
            .executors()
            .iter()
            .find(|executor| executor.backend() == seismic::BackendName::Cpu)
            .ok_or("the host has no CPU executor")?
            .clone();
        let records = KernelCache::measurement_records(cache);
        let mut jobs = None;
        let line = loop {
            match MachineLine::stored(&topology, &executor, &records) {
                Err(MachineLineError::Unrecorded { .. } | MachineLineError::UnrecordedSweep { .. }) => {
                    // Each run measures something or fails with the jobs'
                    // stall, so the loop ends.
                    jobs
                        .get_or_insert_with(|| {
                            InProcessJobs::of_line(
                                &catalog,
                                cache.to_path_buf(),
                                storage.to_path_buf(),
                                executor.clone(),
                            )
                        })
                        .run()
                        .map_err(|error| format!("measuring the CPU's line: {error}"))?;
                }
                other => break other.map_err(|error| format!("the CPU's line: {error}"))?,
            }
        };
        match line.peaks.iter().find(|(operand, _)| *operand == Operand::Int8) {
            Some((_, Peak::Rate(Rate::Reached(peak)))) => Ok(peak.per_second.value),
            Some((_, Peak::Rate(Rate::Open(peak)))) => Err(format!(
                "the CPU's int8 peak was not reached (at least {:.3e} operations a second)",
                peak.per_second.value
            )),
            Some((_, Peak::Unscaled { .. })) => Err("the CPU's int8 work was not shown to scale".into()),
            Some((_, Peak::Defect(reason))) => Err(format!("the CPU's int8 probe measured nothing: {reason}")),
            Some((_, Peak::Unobtainable(reason))) => Err(format!("the CPU's int8 peak is unobtainable: {reason}")),
            None => Err("the CPU executor has no int8 peak".into()),
        }
    }
}

/// The storage device's own lines at a record's size, measured on the
/// model's file with the page cache bypassed (`O_DIRECT` on Linux,
/// `F_NOCACHE` on macOS) and nothing of our read path: reads of `bytes` at
/// random aligned offsets, at each queue depth (threads each reading one at
/// a time) for [`SECONDS`] each, then a stream at the peak's depth that
/// drains any burst the device holds ([`StorageMachine::measure`]).
///
/// A device that bursts is a token bucket:
/// it reads at most `β + S·t` bytes in any `t`
/// seconds, at its peak `P` until its bucket `β` (the bytes beyond `S`) has
/// drained, then at `S`, refilling at `S` less what it reads.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct StorageMachine {
    /// The bytes each read reads: a record's, aligned up to [`ALIGN`].
    pub bytes: u64,
    /// One read's seconds alone (queue depth 1): latency and transfer.
    pub first_seconds: f64,
    /// The highest rate over the queue depths (`P`, the burst rate), and the
    /// least depth that reaches it within its windows' resolution.
    pub peak_bytes_per_second: f64,
    pub peak_depth: usize,
    /// The rate it holds at the peak's depth once any burst has drained
    /// (`S`): a volume that bursts (a 3.17 GB/s peak holding 2.51) is read
    /// at this.
    pub sustained_bytes_per_second: f64,
    /// The bytes beyond `S` the device read from a full bucket to its drain
    /// (`β`): 0 where it held one rate through the stream (no burst, or one
    /// outlasting [`BURST_WINDOWS`]).
    pub burst_bytes: f64,
    /// Each queue depth's rate.
    pub depths: Vec<(usize, f64)>,
}

/// The alignment of uncached reads' offsets, lengths and buffers.
const ALIGN: u64 = 4096;

/// The queue depths probed.
const DEPTHS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

/// The stream's windows, of [`SECONDS`] each, a burst may last and still be
/// measured: 8 s at the peak, which drains a bucket of `(P − S)·8 s` beyond
/// what the depths drained (an M4 Pro volume's: 0.66 GB/s × 8 s = 5.3 GB,
/// about twice its 2.8 GB). A longer burst reads as the sustained rate.
const BURST_WINDOWS: usize = 32;

/// What the storage probe reads through: a device's reads of a record's
/// bytes, each read counted when it completes.
trait Reads {
    /// `depth` readers from idle for [`SECONDS`], each finishing the read
    /// it has under way: the reads, and the seconds to the last one's end.
    fn probe(&mut self, depth: usize) -> Result<(u64, f64), String>;
    /// The next [`SECONDS`] of one stream of `depth` readers, which runs
    /// from the first call to the probe's end: the reads it completed, and
    /// its seconds.
    fn stream(&mut self, depth: usize) -> Result<(u64, f64), String>;
}

/// A window of reads at a queue depth.
#[derive(Clone, Copy)]
struct Window {
    bytes: f64,
    seconds: f64,
    depth: usize,
    /// The record's bytes.
    read: f64,
}

impl Window {
    fn rate(&self) -> f64 {
        self.bytes / self.seconds
    }

    /// The window's resolution, its own noise: the reads under way at each
    /// of its edges are counted in the window they complete in, so its bytes
    /// are known to within its depth's reads.
    fn noise(&self) -> f64 {
        self.depth as f64 * self.read / self.seconds
    }

    /// Whether the two windows read at one rate, within their resolutions.
    fn agrees(&self, other: &Self) -> bool {
        (self.rate() - other.rate()).abs() <= self.noise() + other.noise()
    }

    /// Whether this window reads below `other` by more than their resolutions.
    fn below(&self, other: &Self) -> bool {
        self.rate() < other.rate() - (self.noise() + other.noise())
    }
}

impl StorageMachine {
    /// Measure the device holding `file` at reads of `bytes`.
    pub fn measure(file: &std::path::Path, bytes: u64) -> Result<Self, String> {
        let bytes = bytes.div_ceil(ALIGN) * ALIGN;
        let length = std::fs::metadata(file).map_err(|error| format!("{}: {error}", file.display()))?.len();
        let slots = length / ALIGN - bytes / ALIGN;
        if slots == 0 {
            return Err(format!("{} holds no read of {bytes} B", file.display()));
        }
        let mut reads = FileReads { file, bytes, slots, stream: None };
        let lines = Self::search(&mut reads, bytes);
        let stopped = reads.stream.map_or(Ok(0.), |(stream, ..)| stream.stop());
        let lines = lines?;
        stopped?;
        Ok(lines)
    }

    /// The lines `reads` gives at reads of `bytes`: each depth's window, then
    /// one stream at the peak's depth in windows of [`SECONDS`].
    ///
    /// The stream stops at the first of:
    /// - a window below the peak beyond their resolutions (the drop) followed
    ///   by two windows that agree within theirs: `S` is their rate;
    /// - the second window after the drop: the token-bucket model puts the
    ///   drain within the drop's window, so the next reads at `S` and the one
    ///   after confirms it; a device still slowing there is one the model
    ///   does not explain, and `S` is the rate that window reached;
    /// - [`BURST_WINDOWS`] windows with no drop: the device holds its rate
    ///   (`S`, the stream's) with no burst it can measure.
    ///
    /// So the stream reads at most `BURST_WINDOWS + 2` windows (a drop in the
    /// last allowed window and two after it), and the probe at most
    /// `7 + 34 = 41` windows of 0.25 s: 10.25 s, plus the read each reader
    /// has under way at each window's end, and at most that time at the
    /// device's rate (an M4 Pro: 10.25 s × 3.17 GB/s ≤ 32.5 GB).
    fn search(reads: &mut impl Reads, bytes: u64) -> Result<Self, String> {
        let read = bytes as f64;
        let window = |(count, seconds): (u64, f64), depth| Window { bytes: count as f64 * read, seconds, depth, read };
        // Every window, in the order read: the bucket's account runs over all.
        let mut windows = Vec::new();
        for depth in DEPTHS {
            windows.push(window(reads.probe(depth)?, depth));
        }
        let first_seconds = windows[0].seconds * read / windows[0].bytes;
        let highest = *windows.iter().max_by(|a, b| a.rate().total_cmp(&b.rate())).expect("a depth");
        let peak = *windows.iter().find(|window| !window.below(&highest)).expect("the highest is not below itself");
        let sweep = windows.len();
        let mut drop = None;
        let sustained = loop {
            windows.push(window(reads.stream(peak.depth)?, peak.depth));
            let at = windows.len() - 1;
            let (last, before) = (windows[at], windows[at - 1]);
            match drop {
                None if last.below(&peak) => drop = Some(at),
                None if at - sweep + 1 == BURST_WINDOWS => {
                    let stream = &windows[sweep..];
                    break stream.iter().map(|window| window.bytes).sum::<f64>() / stream.iter().map(|window| window.seconds).sum::<f64>();
                }
                None => {}
                Some(_) if last.agrees(&before) => break (last.bytes + before.bytes) / (last.seconds + before.seconds),
                Some(dropped) if at == dropped + 2 => break last.rate(),
                Some(_) => {}
            }
        };
        // The bucket's deficit from full, refilling at `S` less use, up to
        // the drop's window: its deepest is the bytes beyond `S` it held.
        let burst = drop.map_or(0., |dropped| {
            let mut deficit = 0f64;
            let mut deepest = 0f64;
            for window in &windows[..=dropped] {
                deficit = (deficit + window.bytes - sustained * window.seconds).max(0.);
                deepest = deepest.max(deficit);
            }
            deepest
        });
        Ok(Self {
            bytes,
            first_seconds,
            peak_bytes_per_second: highest.rate(),
            peak_depth: peak.depth,
            sustained_bytes_per_second: sustained,
            burst_bytes: burst,
            depths: windows[..sweep].iter().map(|window| (window.depth, window.rate())).collect(),
        })
    }
}

/// [`Reads`] of a file's device, uncached.
struct FileReads<'a> {
    file: &'a std::path::Path,
    bytes: u64,
    slots: u64,
    /// The stream, its reads when last sampled, and when.
    stream: Option<(ReadStream, u64, Instant)>,
}

impl Reads for FileReads<'_> {
    fn probe(&mut self, depth: usize) -> Result<(u64, f64), String> {
        probe(self.file, self.bytes, self.slots, depth)
    }

    fn stream(&mut self, depth: usize) -> Result<(u64, f64), String> {
        if self.stream.is_none() {
            let stream = ReadStream::start(self.file, self.bytes, depth)?;
            let began = stream.began;
            self.stream = Some((stream, 0, began));
        }
        let (stream, reads, at) = self.stream.as_mut().expect("started above");
        std::thread::sleep((*at + std::time::Duration::from_secs_f64(SECONDS)).saturating_duration_since(Instant::now()));
        let (now, when) = (stream.reads(), Instant::now());
        let window = (now - *reads, (when - *at).as_secs_f64());
        (*reads, *at) = (now, when);
        Ok(window)
    }
}

/// `depth` threads each reading `bytes` at random offsets among `slots`
/// aligned ones of `file`, uncached, for [`SECONDS`]: the reads and the
/// seconds from their start to the last one's end. Every thread's file is
/// opened before any starts, so one that cannot open fails the probe
/// rather than leaving the others at the start.
fn probe(file: &std::path::Path, bytes: u64, slots: u64, depth: usize) -> Result<(u64, f64), String> {
    let opened = (0..depth).map(|_| uncached(file)).collect::<Result<Vec<_>, _>>()?;
    let start = Arc::new(Barrier::new(depth + 1));
    let handles = (0..depth as u64)
        .zip(opened)
        .map(|(thread, handle)| {
            let (start, file) = (start.clone(), file.to_owned());
            std::thread::spawn(move || -> Result<(u64, Instant), String> {
                let mut buffer = vec![0u8; (bytes + ALIGN) as usize];
                let offset = buffer.as_ptr().align_offset(ALIGN as usize);
                let buffer = &mut buffer[offset..offset + bytes as usize];
                let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ (thread + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                start.wait();
                let began = Instant::now();
                let mut reads = 0;
                while began.elapsed().as_secs_f64() < SECONDS {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let at = state % slots * ALIGN;
                    std::os::unix::fs::FileExt::read_exact_at(&handle, buffer, at).map_err(|error| format!("{}: {error}", file.display()))?;
                    reads += 1;
                }
                Ok((reads, Instant::now()))
            })
        })
        .collect::<Vec<_>>();
    start.wait();
    let began = Instant::now();
    let mut total = 0;
    let mut ended = began;
    for handle in handles {
        let (reads, end) = handle.join().map_err(|_| "a probing thread panicked".to_owned())??;
        total += reads;
        ended = ended.max(end);
    }
    Ok((total, (ended - began).as_secs_f64()))
}

/// `file` opened for reads the page cache does not serve.
fn uncached(file: &std::path::Path) -> Result<std::fs::File, String> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(target_os = "linux")]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut options, libc::O_DIRECT);
    let handle = options.open(file).map_err(|error| format!("{}: {error}", file.display()))?;
    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: an open descriptor; F_NOCACHE takes an int.
        if unsafe { libc::fcntl(handle.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
            return Err(format!("{}: F_NOCACHE: {}", file.display(), std::io::Error::last_os_error()));
        }
    }
    Ok(handle)
}

/// The read stream that reaches a storage device's peak on a file: the read
/// size and queue depth of the highest rate, each searched by doubling from
/// the least (one aligned block, one read in flight) until the rate stops
/// rising, the depth within the probed depths ([`DEPTHS`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PeakStream {
    pub bytes: u64,
    pub depth: usize,
    pub bytes_per_second: f64,
}

impl PeakStream {
    /// Search `file` for its device's peak stream.
    pub fn search(file: &std::path::Path) -> Result<Self, String> {
        let length = std::fs::metadata(file).map_err(|error| format!("{}: {error}", file.display()))?.len();
        let rate = |bytes: u64, depth: usize| -> Result<Option<f64>, String> {
            let slots = (length / ALIGN).saturating_sub(bytes / ALIGN);
            if slots == 0 {
                return Ok(None);
            }
            let (reads, seconds) = probe(file, bytes, slots, depth)?;
            Ok(Some((reads * bytes) as f64 / seconds))
        };
        let mut best: Option<Self> = None;
        let mut bytes = ALIGN;
        loop {
            // The depth of this size's highest rate, of the probed depths.
            let mut at: Option<Self> = None;
            for depth in DEPTHS {
                let Some(bytes_per_second) = rate(bytes, depth)? else {
                    break;
                };
                if at.is_some_and(|at| bytes_per_second <= at.bytes_per_second) {
                    break;
                }
                at = Some(Self { bytes, depth, bytes_per_second });
            }
            match (at, best) {
                (Some(at), Some(best)) if at.bytes_per_second <= best.bytes_per_second => return Ok(best),
                (Some(at), _) => best = Some(at),
                (None, Some(best)) => return Ok(best),
                (None, None) => return Err(format!("{} holds no read of {bytes} B", file.display())),
            }
            bytes *= 2;
        }
    }
}

/// A read stream running on a file until it is stopped
/// ([`ReadStream::stop`]): `depth` threads each reading `bytes` at random
/// aligned offsets, uncached.
pub struct ReadStream {
    stop: Arc<std::sync::atomic::AtomicBool>,
    /// The reads completed so far.
    reads: Arc<std::sync::atomic::AtomicU64>,
    began: Instant,
    bytes: u64,
    handles: Vec<std::thread::JoinHandle<Result<(), String>>>,
}

impl ReadStream {
    /// Start `depth` threads reading `bytes` at a time on `file`.
    pub fn start(file: &std::path::Path, bytes: u64, depth: usize) -> Result<Self, String> {
        let length = std::fs::metadata(file).map_err(|error| format!("{}: {error}", file.display()))?.len();
        let slots = (length / ALIGN).saturating_sub(bytes / ALIGN).max(1);
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reads = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let handles = (0..depth as u64)
            .map(|thread| {
                let (stop, reads, file) = (stop.clone(), reads.clone(), file.to_owned());
                std::thread::spawn(move || -> Result<(), String> {
                    let handle = uncached(&file)?;
                    let mut buffer = vec![0u8; (bytes + ALIGN) as usize];
                    let offset = buffer.as_ptr().align_offset(ALIGN as usize);
                    let buffer = &mut buffer[offset..offset + bytes as usize];
                    let mut state = 0x2545_f491_4f6c_dd1du64 ^ (thread + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        std::os::unix::fs::FileExt::read_exact_at(&handle, buffer, state % slots * ALIGN)
                            .map_err(|error| format!("{}: {error}", file.display()))?;
                        reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    Ok(())
                })
            })
            .collect();
        Ok(Self { stop, reads, began: Instant::now(), bytes, handles })
    }

    /// The reads completed so far.
    fn reads(&self) -> u64 {
        self.reads.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Stop the stream: the rate it read at.
    pub fn stop(self) -> Result<f64, String> {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for handle in self.handles {
            handle.join().map_err(|_| "a streaming thread panicked".to_owned())??;
        }
        Ok((self.reads.load(std::sync::atomic::Ordering::Relaxed) * self.bytes) as f64 / self.began.elapsed().as_secs_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A record's bytes, as an M4 Pro streams them.
    const RECORD: u64 = 1_900_544;
    /// The simulation's step.
    const STEP: f64 = 1e-4;

    /// A simulated device: each reader reads a record in `alone` seconds, the
    /// readers together at most the rate `cap` gives for the next step (from
    /// the clock and the rate they would read at), which may keep its own
    /// state. Reads complete whole, so windows carry the probe's resolution.
    struct Simulated<F> {
        alone: f64,
        cap: F,
        /// The seconds every window has read for.
        clock: f64,
        /// The stream's bytes read but not yet completed as a whole read.
        partial: f64,
    }

    impl<F: FnMut(f64, f64) -> f64> Simulated<F> {
        fn new(alone: f64, cap: F) -> Self {
            Self { alone, cap, clock: 0., partial: 0. }
        }

        fn window(&mut self, depth: usize, mut partial: f64) -> (u64, f64, f64) {
            let wanted = depth as f64 * RECORD as f64 / self.alone;
            for _ in 0..(SECONDS / STEP).round() as usize {
                partial += (self.cap)(self.clock, wanted) * STEP;
                self.clock += STEP;
            }
            let reads = (partial / RECORD as f64).floor();
            (reads as u64, SECONDS, partial - reads * RECORD as f64)
        }
    }

    impl<F: FnMut(f64, f64) -> f64> Reads for Simulated<F> {
        fn probe(&mut self, depth: usize) -> Result<(u64, f64), String> {
            let (reads, seconds, _) = self.window(depth, 0.);
            Ok((reads, seconds))
        }

        fn stream(&mut self, depth: usize) -> Result<(u64, f64), String> {
            let (reads, seconds, partial) = self.window(depth, self.partial);
            self.partial = partial;
            Ok((reads, seconds))
        }
    }

    /// The most seconds a probe reads for ([`StorageMachine::search`]).
    const BOUND: f64 = (DEPTHS.len() + BURST_WINDOWS + 2) as f64 * SECONDS;

    fn near(measured: f64, expected: f64, tolerance: f64) -> bool {
        (measured / expected - 1.).abs() <= tolerance
    }

    #[test]
    fn a_token_bucket_reads_as_its_burst_rate_burst_and_sustained_rate() {
        // An M4 Pro volume: 3.17 GB/s, a 2.8 GB bucket beyond
        // 2.51 GB/s, one record alone in 0.971 ms.
        let (peak, burst, sustained) = (3.17e9, 2.8e9, 2.51e9);
        let mut level = burst;
        let mut device = Simulated::new(0.971e-3, |_, wanted: f64| {
            let rate = wanted.min(if level > 0. { peak } else { sustained });
            level = (level - (rate - sustained) * STEP).clamp(0., burst);
            rate
        });
        let lines = StorageMachine::search(&mut device, RECORD).unwrap();
        assert!(near(lines.peak_bytes_per_second, peak, 0.01), "{lines:?}");
        assert!(near(lines.sustained_bytes_per_second, sustained, 0.01), "{lines:?}");
        assert!(near(lines.burst_bytes, burst, 0.05), "{lines:?}");
        assert!(device.clock <= BOUND, "{} s", device.clock);
    }

    #[test]
    fn a_device_slowing_forever_stops_within_the_bound_at_the_rate_it_reached() {
        // Halving its rate every 2 s from 3.17 GB/s.
        let mut reached = 0.;
        let mut device = Simulated::new(0.971e-3, |clock: f64, wanted: f64| {
            reached = wanted.min(3.17e9 * 0.5f64.powf(clock / 2.));
            reached
        });
        let lines = StorageMachine::search(&mut device, RECORD).unwrap();
        assert!(device.clock <= BOUND, "{} s", device.clock);
        // The last window's rate, of a rate that fell through it.
        assert!(near(lines.sustained_bytes_per_second, reached, 0.1), "{lines:?} reached {reached}");
    }

    #[test]
    fn a_flat_device_stops_after_the_minimum_windows() {
        let mut device = Simulated::new(0.416e-3, |_, wanted: f64| wanted.min(10.15e9));
        let lines = StorageMachine::search(&mut device, RECORD).unwrap();
        assert!(near(device.clock, (DEPTHS.len() + BURST_WINDOWS) as f64 * SECONDS, 1e-9), "{} s", device.clock);
        assert!(near(lines.sustained_bytes_per_second, 10.15e9, 0.01), "{lines:?}");
        assert_eq!(lines.burst_bytes, 0., "{lines:?}");
    }
}
