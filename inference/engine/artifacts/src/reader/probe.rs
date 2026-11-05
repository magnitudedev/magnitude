//! # A volume's window
//!
//! How fast a device reads depends on the bytes it holds in flight and on
//! the size of the reads that hold them, and how soon it serves a lone
//! request on how the request is split; no two devices measured agree
//! (`tests/read_depth.rs`, uncached reads at random offsets of Qwen3.6
//! 35B's file):
//!
//! - GB10's NVMe: the rate is the
//!   bytes in flight's alone, 8 MiB reaching 10.1–10.2 GB/s in reads of
//!   128 KiB to 2 MiB and 2 MiB only 4.5–5.2; a 1.9 MB record alone reads in
//!   0.41 ms as one read, 0.65 ms as fifteen 128 KiB reads at once.
//! - an RTX 3090 host (Xeon E5-2696 v3): 2 MiB in flight reads
//!   2.84 GB/s in 128 KiB reads and 3.47 in 512 KiB ones, and 128 KiB reads
//!   near the device's 3.6 only at 16 MiB; yet a record alone reads in
//!   1.9 ms as fifteen 128 KiB reads, 3.8 ms as one.
//!
//! So only a volume can say how it is read best, and it is probed
//! ([`VolumeLines`]): a file of the volume is read uncached, its cached
//! pages dropped first, at each read size its request needs and every
//! window doubling from [`LEAST_WINDOW`] to
//! [`WIDEST_WINDOW`], from as many threads as hold the window, for
//! [`PROBE_SECONDS`], in [`PASSES`] passes in alternating order; each point
//! is the median of its passes, with their half range as its spread. One
//! pass's points differ by up to 30% at the knee and a few percent past it,
//! so a decision from one pass flipped between processes (GB10: 8 MiB in
//! 1 MiB reads, then 16 MiB in 2 MiB). Each volume is probed under its own
//! lock, so one volume's probe holds no other's reader.
//!
//! How a reader reads its volume is chosen for its urgent request
//! ([`Urgent`]): the most one step's request reads, `records` records of
//! the model file each stored as separate ranges (`parts`). An urgent read
//! queues behind every byte in flight, so a window `W` read at rate `R`
//! serves it in
//!
//! ```text
//! T(r, W) = (W + q) / R(r, W) + λ(r)
//! ```
//!
//! where `q` is the request's bytes and `λ(r)` its latency beyond its
//! bytes, measured: one record alone in its stored ranges, submitted to
//! the read engine as a demand job in pieces of `r`, as a step's own reads
//! are, less its bytes at the best rate. The read sizes probed run from
//! [`LEAST_READ`] doubling to the least that holds a record's widest range,
//! at most [`WIDEST_WINDOW`], past which no window holds a read.
//! `T` prices both what a wider window costs an urgent read and what a
//! slower rate costs every byte it waits for. The volume reads at the read
//! size and window of least `T`; where several are within the noise of the
//! least (their rates' spreads and the lone medians' standard errors), at the
//! narrowest window, then the least read size: a tie gains nothing from
//! wider pieces, and narrower ones are finer grains of priority. Recorded
//! probes, by this rule (`T` at a request of 8 records of 1.9 MB): GB10
//! 4 MiB in 128 KiB reads, 9.66 GB/s, `T` 2.52 ms,
//! with 256 and 512 KiB reads at 4 MiB within its noise, against 2.84 ms
//! at 8 MiB in 1 MiB reads and 3.53 at 16 MiB in 2 MiB; the RTX 3090
//! host 2 MiB in 512 KiB reads, 3.48 GB/s, `T` 5.10 ms,
//! its 128 KiB reads reaching that rate only at 16 MiB.
//!
//! One volume serves both kinds of reads at once: an urgent request, and
//! bulk (a prompt's fill, a tier's fill, hints) that wants the full rate.
//! A piece the device holds cannot be taken back, so an urgent read waits
//! behind every byte already in flight, however its priority places it
//! ahead of the pieces queued: the window that bounds its wait is the bytes
//! in flight, not the queue. So each volume has two windows. A demand piece
//! joins the bytes in flight up to the full rate's window: the read size's
//! widest within the noise of its line's best rate (GB10: 32 MiB in
//! 128 KiB reads). A prompt's fill and a step's own reads are demand, and
//! read at the full rate. Past its line's knee a wider window costs a read
//! alone nothing (GB10, a prompt's experts alone: 9.75 GB/s at 8 MiB, 9.79
//! at 32), while serving lengthens each read's time in flight, so the knee
//! measured alone falls short of the window a served prompt needs (GB10
//! CUDA's prompt fill, beside the device's rounds: 7.07 GB/s at 8 MiB, 7.50
//! at 32; capped 4k prefill 2160 to 2021 ms). A step's demand job waits
//! behind that window only where it follows a prompt's fill; within a step
//! its jobs run one at a time, each well inside it. Every other piece (hints, a tier's
//! fill) joins only up to the urgent request's window, its point of least
//! `T` (GB10: 4 MiB), whichever reader submits it (the volume's own bound,
//! the narrowest window of its live urgent readers, held while each lives),
//! so the bulk ahead of a decode request is no deeper than its `T` allows. That bound costs the less
//! urgent reads their rate past it: GB10's hints read at 9.4–9.6 GB/s
//! against 10.1 (−5%), but a decode step is not bound by its reads'
//! bandwidth there (its replay: all wrong hints cost 3 ms of 62); and
//! the RTX 3090 host's tier fill at 3.48 against 3.6 GB/s (−3%). Widening it while
//! no urgent read is in flight would put the wider window's bytes ahead of
//! the next step's first demand read ((W_full − W)/R a step: 16 − 2 MiB at
//! 3.6 GB/s is 4 ms on that host), which the read engine cannot see coming;
//! so it does not widen.
//!
//! A stored table's rows are an urgent request too (one head's rows of a
//! step), and theirs is a small request, so its window is narrow: while a
//! stored table's reader lives, it narrows the bound for the volume's hints
//! and tier fill. That cost is unmeasured: no cell measured has a stored
//! table (an n-gram model).
//!
//! A demand job reads in reads of its own size, the one a lone request
//! is served soonest at through the engine: what the job pays, its pieces'
//! cut, wakes and copies included, which threads reading the same sizes do
//! not show (GB10, 200 lone experts of three ranges each: 0.63–0.73 ms by
//! threads at any size, 0.70–0.96 through the engine, 128 KiB the slowest).
//! Sizes within each other's noise cost it the same, so no tie rule picks
//! among them: one that took the least such size by threads' timing chose
//! 128 KiB demand reads on Mac Metal, slower than 256 KiB through the
//! engine (capped decode +3.4%). A lone request gains from a wider piece
//! what a bulk read's priority grain loses, and only it pays neither's
//! price.
//!
//! A reader no step waits on (a load's import: no [`Urgent`]) reads at the
//! least read size, the pool's grain, at the full rate's window for every
//! piece.
//!
//! A reader measures nothing: the probe runs where nothing waits on it
//! ([`probe_volume`], the measurement job), a load adopts its choice
//! ([`adopt_volume`]), and until one exists a volume reads at the search's
//! bounds ([`VolumeChoice`]).
//!
//! The probe also measures the cores the reads occupy at the chosen size
//! and window, each read copied out of a buffer as a piece that does not
//! land straight is, the median of its passes ([`read_cores`]): what the
//! host keeps free for the pool's workers.

use super::engine::Aligned;
use super::file::SourceFile;
use super::platform::thread_cpu_seconds;
use super::volume::{Bound, Delivery, Volume, VolumeChoice, VolumeKey};
#[cfg(doc)]
use super::volume::{adopt_volume, probe_volume, read_cores};
use super::{queue, Ended, Priority, Range, ALIGNMENT};
use crate::Error;
use std::collections::{BTreeMap, HashMap};
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// The least read size a volume's probe reads in, doubling up to the least
/// that holds a record's widest range, at most the widest window ("A
/// volume's window"): the grain the
/// pool's workers are counted in, and below which no device measured read
/// faster or sooner (`tests/read_depth.rs`).
pub(super) const LEAST_READ: u64 = 128 * 1024;

/// The least and the widest windows a volume's probe holds in flight,
/// doubling: below 1 MiB no device measured nears its rate, and 32 MiB is
/// twice the widest window any measured device needed to reach it
/// (an RTX 3090 host's 128 KiB reads, 16 MiB).
const LEAST_WINDOW: u64 = 1024 * 1024;
pub(super) const WIDEST_WINDOW: u64 = 32 * 1024 * 1024;

/// The seconds a volume's probe reads at each read size and window, and
/// the passes it takes each point's median of.
const PROBE_SECONDS: f64 = 0.03;
const PASSES: usize = 3;

/// The seconds a probe times lone records in, in rounds of one record at
/// each read size, so drift is shared within a round: on GB10 a record
/// alone takes 0.4–0.9 ms, so about 50 rounds of five sizes.
const LONE_SECONDS: f64 = 0.2;

/// The urgent request a reader's steps wait on: `records` records of its
/// file, each stored as ranges of `parts` bytes (an expert's gate, up and
/// down, separate tensors), what one row's request at a step reads ("A
/// volume's window"). A step of several rows asks up to that many times
/// as much at once; the window is priced at one row's, the request every
/// step has.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Urgent {
    pub parts: Vec<u64>,
    pub records: u64,
}

impl Urgent {
    /// A request's bytes.
    fn bytes(&self) -> u64 {
        self.parts.iter().sum::<u64>() * self.records
    }

    /// The read sizes it is probed at: from [`LEAST_READ`] doubling to the
    /// least that holds its widest range, past which a read reads nothing
    /// more of a record, or to [`WIDEST_WINDOW`], past which no window of
    /// the probe holds one.
    pub(super) fn reads(&self) -> Vec<u64> {
        let widest = self.parts.iter().copied().max().expect("a request has a range").max(LEAST_READ).next_power_of_two();
        std::iter::successors(Some(LEAST_READ), |read| Some(read * 2)).take_while(|read| *read <= widest.min(WIDEST_WINDOW)).collect()
    }
}

/// A volume's probed lines, and how each urgent request reads it ("A
/// volume's window"): each read size's line, its windows' rates and cores
/// as the medians of their passes, each probed when a probe first needs
/// it, with the bytes of the file it was probed through; and each urgent
/// request's choice (none: a reader no step waits on).
#[derive(Default)]
pub(super) struct VolumeLines {
    lines: BTreeMap<u64, (u64, Vec<Point>)>,
    /// The volume's bound on its less urgent bytes in flight, which its
    /// urgent readers hold their windows in while they live.
    background: Arc<Bound>,
    /// What the volume reads and how long it is busy ([`Delivery`]).
    delivery: Arc<Delivery>,
    pub(super) chosen: HashMap<Option<Urgent>, Volume>,
}

/// One read size and window of a volume: the median rate and cores of its
/// passes, and the rates' half range, its spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Point {
    pub(super) read: u64,
    pub(super) window: u64,
    pub(super) rate: f64,
    pub(super) spread: f64,
    pub(super) cores: f64,
}

impl Point {
    fn of(read: u64, window: u64, passes: &[(f64, f64)]) -> Self {
        let rates = passes.iter().map(|&(rate, _)| rate).collect::<Vec<_>>();
        let (least, most) = rates.iter().fold((f64::INFINITY, 0f64), |(least, most), &rate| (least.min(rate), most.max(rate)));
        let cores = median(&passes.iter().map(|&(_, cores)| cores).collect::<Vec<_>>());
        Self { read, window, rate: median(&rates), spread: (most - least) / 2., cores }
    }
}

impl VolumeLines {
    /// The volume `key` names, read as `choice`.
    pub(super) fn volume(&self, key: VolumeKey, choice: VolumeChoice) -> Volume {
        Volume { key, choice, background: self.background.clone(), delivery: self.delivery.clone() }
    }

    /// How its volume is read for `urgent`, probing through `file` the
    /// lines it needs that are not yet probed.
    pub(super) fn choose(&mut self, key: VolumeKey, file: &Arc<SourceFile>, urgent: Option<&Urgent>) -> Result<VolumeChoice, Error> {
        let reads = urgent.map_or(vec![LEAST_READ], Urgent::reads);
        self.probe(file, &reads)?;
        let points = reads.iter().flat_map(|read| self.lines[read].1.iter().copied()).collect::<Vec<_>>();
        let Some(urgent) = urgent else {
            // No step waits on the reads: every piece at the full rate.
            let chosen = full_rate(&points);
            return Ok(VolumeChoice { window: chosen.window, demand: chosen.window, read: chosen.read, demand_read: chosen.read, cores: chosen.cores });
        };
        let seconds = lone(self, key, file, &urgent.parts, &reads)?;
        let lone = reads.iter().zip(&seconds).map(|(&read, seconds)| (read, (median(seconds), median_error(seconds)))).collect::<HashMap<_, _>>();
        let chosen = decide(&points, &lone, urgent);
        let demand_read = soonest(&lone);
        // Demand reads fill the chosen size's full-rate window: a demand
        // read queued behind them is itself bulk (a prompt's fill), or waits
        // on reads its own step needs first.
        let full = full_rate(&self.lines[&chosen.read].1);
        Ok(VolumeChoice { window: chosen.window, demand: full.window.max(chosen.window), read: chosen.read, demand_read, cores: full.cores })
    }

    /// Probe `reads`' lines not yet probed through a file as large as
    /// `file`: every window doubling from [`LEAST_WINDOW`] (or one read) to
    /// [`WIDEST_WINDOW`], in [`PASSES`] passes over the points in
    /// alternating order. A line a smaller file was probed through (a
    /// draft model's, probed first) is probed again: the device's own
    /// caches may hold more of a small file than its reads would of the
    /// model's.
    fn probe(&mut self, file: &SourceFile, reads: &[u64]) -> Result<(), Error> {
        let grid = grid(reads.iter().copied().filter(|read| self.lines.get(read).is_none_or(|(size, _)| *size < file.size)));
        let mut passes = vec![Vec::with_capacity(PASSES); grid.len()];
        for pass in 0..PASSES {
            for at in 0..grid.len() {
                let at = if pass % 2 == 0 { at } else { grid.len() - 1 - at };
                let (read, window) = grid[at];
                passes[at].push(rate(file, read, (window / read) as usize)?);
            }
        }
        // A line probed again changes every choice made from it: those are
        // forgotten, and read at the search's bounds until chosen again.
        if grid.iter().any(|(read, _)| self.lines.contains_key(read)) {
            self.chosen.clear();
        }
        for read in grid.iter().map(|&(read, _)| read) {
            self.lines.insert(read, (file.size, Vec::new()));
        }
        for (&(read, window), passes) in grid.iter().zip(&passes) {
            self.lines.get_mut(&read).expect("inserted above").1.push(Point::of(read, window, passes));
        }
        Ok(())
    }
}

/// Of `points`, the full rate's: the widest window within the noise of the
/// best rate, then the least read ("A volume's window": serving lengthens a
/// read's time in flight, which only a wider window holds the rate through).
pub(super) fn full_rate(points: &[Point]) -> Point {
    let top = points.iter().max_by(|a, b| a.rate.total_cmp(&b.rate)).expect("a probed point");
    *points
        .iter()
        .filter(|point| point.rate + point.spread + top.spread >= top.rate)
        .max_by_key(|point| (point.window, std::cmp::Reverse(point.read)))
        .expect("the best is within its own noise")
}

/// Of `points`, the one `urgent` is served soonest at, by `lone` (each read
/// size's lone record: its median seconds and that median's error) ("A
/// volume's window"): least `T`, and of those within the noise of the least,
/// the narrowest window, then the least read.
pub(super) fn decide(points: &[Point], lone: &HashMap<u64, (f64, f64)>, urgent: &Urgent) -> Point {
    let best = points.iter().map(|point| point.rate).fold(0., f64::max);
    let record = urgent.parts.iter().sum::<u64>() as f64;
    let timed = points
        .iter()
        .map(|point| {
            let (alone, alone_error) = lone[&point.read];
            let waited = (point.window + urgent.bytes()) as f64 / point.rate;
            (*point, waited + alone - record / best, waited * point.spread / point.rate + alone_error)
        })
        .collect::<Vec<_>>();
    let &(_, least, noise) = timed.iter().min_by(|a, b| a.1.total_cmp(&b.1)).expect("a probed point");
    let (chosen, ..) = *timed
        .iter()
        .filter(|(_, time, spread)| *time <= least + noise + spread)
        .min_by_key(|(point, ..)| (point.window, point.read))
        .expect("the least is within its own noise");
    chosen
}

/// The read size a lone request is served soonest at through the engine:
/// its least median ("A volume's window"). Sizes within each other's noise
/// cost a request the same, so which of them it takes gains or loses
/// nothing; a tie broken toward one size by rule instead chose one the
/// engine pays more for (Mac Metal, 128 KiB against 256 KiB demand reads:
/// capped decode +3.4%).
fn soonest(lone: &HashMap<u64, (f64, f64)>) -> u64 {
    let (&read, _) = lone.iter().min_by(|a, b| a.1 .0.total_cmp(&b.1 .0)).expect("a probed read size");
    read
}

/// The points a probe of read sizes `reads` reads: each size at every
/// window doubling from [`LEAST_WINDOW`] (or one read) to [`WIDEST_WINDOW`].
pub(super) fn grid(reads: impl IntoIterator<Item = u64>) -> Vec<(u64, u64)> {
    reads
        .into_iter()
        .flat_map(|read| {
            std::iter::successors(Some(read.max(LEAST_WINDOW)), |window| Some(window * 2))
                .take_while(|window| *window <= WIDEST_WINDOW)
                .map(move |window| (read, window))
        })
        .collect()
}

/// The median of `values`.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

/// The standard error of `values`' median, from their interquartile range
/// (robust to a lone read's long tail, which a range is not): the spread a
/// median within it is not told apart by.
fn median_error(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let quartile = |at: usize| sorted[(at * (sorted.len() - 1)) / 4];
    // A normal's IQR is 1.349 σ, and its median's error 1.253 σ/√n.
    1.253 * (quartile(3) - quartile(1)) / 1.349 / (sorted.len() as f64).sqrt()
}

/// A random aligned offset of `file` that a read of `length` bytes from it
/// lies within the file's blocks.
fn random_offset(file: &SourceFile, state: &mut u64, length: u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let span = file.size.div_ceil(ALIGNMENT) * ALIGNMENT;
    *state % (span.saturating_sub(length) / ALIGNMENT + 1) * ALIGNMENT
}

/// The bytes a second `file` reads at, and the cores its reads occupy, with
/// `depth` reads of `read` bytes in flight: `depth` threads, each reading
/// at random aligned offsets for [`PROBE_SECONDS`] from when every one has
/// started, each read into a buffer and copied out as a piece that does not
/// land straight is (the most a piece costs the host).
fn rate(file: &SourceFile, read: u64, depth: usize) -> Result<(f64, f64), Error> {
    let span = file.size.div_ceil(ALIGNMENT) * ALIGNMENT;
    let length = read.min(span) as usize;
    // Where every thread reads until, once all have started; none where one
    // could not, which ends the others at once.
    let start = (Mutex::new(None::<Option<Instant>>), Condvar::new());
    let start = &start;
    std::thread::scope(|scope| {
        let reader = |thread: u64| {
            move || -> Result<(u64, f64, Instant), Error> {
                let (mut buffer, mut sink) = (Aligned::new(length), vec![0u8; length]);
                let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ (thread + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                let until = *start.1.wait_while(start.0.lock().expect("probe start"), |until| until.is_none()).expect("probe start");
                let until = until.expect("the probe started");
                let began = thread_cpu_seconds()?;
                let mut bytes = 0;
                while until.is_some_and(|until| Instant::now() < until) {
                    let at = random_offset(file, &mut state, length as u64);
                    file.read_block(at, buffer.bytes_mut())?;
                    sink.copy_from_slice(buffer.bytes_mut());
                    bytes += (at + length as u64).min(file.size) - at;
                }
                Ok((bytes, thread_cpu_seconds()? - began, Instant::now()))
            }
        };
        let spawned = (0..depth as u64)
            .map(|thread| std::thread::Builder::new().spawn_scoped(scope, reader(thread)))
            .collect::<Result<Vec<_>, _>>();
        let began = Instant::now();
        let until = spawned.as_ref().ok().map(|_| began + std::time::Duration::from_secs_f64(PROBE_SECONDS));
        *start.0.lock().expect("probe start") = Some(until);
        start.1.notify_all();
        let threads = spawned.map_err(Error::ThreadStart)?;
        let (mut bytes, mut cpu, mut ended) = (0, 0., began);
        for thread in threads {
            let (read, seconds, end) = thread.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))?;
            bytes += read;
            cpu += seconds;
            ended = ended.max(end);
        }
        let wall = (ended - began).as_secs_f64();
        Ok((bytes as f64 / wall, cpu / wall))
    })
}

/// Lone requests of `file`, each one record (its `parts`, each at a random
/// offset of its own, unaligned at the 32-byte grain a model file stores
/// tensors at, landing back to back in one record's sink as a bank's
/// records do), each submitted alone to the read engine as a demand job of
/// volume `key` read in pieces of each size of `reads`, as a step's own
/// reads are; in rounds of one request at each size, for [`LONE_SECONDS`]:
/// their seconds, by size and round. What a request pays is the engine's
/// (its cut into pieces, the workers it wakes, the copy of a piece that
/// does not land straight), not a read's alone.
fn lone(lines: &VolumeLines, key: VolumeKey, file: &Arc<SourceFile>, parts: &[u64], reads: &[u64]) -> Result<Vec<Vec<f64>>, Error> {
    const GRAIN: u64 = 32;
    let record = parts.iter().sum::<u64>();
    assert!(record + ALIGNMENT * parts.len() as u64 <= file.size, "a record of {record} bytes beyond a file of {}", file.size);
    let mut sink = Aligned::new(record as usize);
    // Every piece of a request at once: its window holds a record whole.
    let volumes = reads
        .iter()
        .map(|&read| lines.volume(key, VolumeChoice { window: WIDEST_WINDOW, demand: WIDEST_WINDOW, read, demand_read: read, cores: 0. }))
        .collect::<Vec<_>>();
    super::engine::engine().deepen(volumes.iter().map(Volume::workers).max().expect("a read size"))?;
    let outstanding = Arc::default();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut seconds = vec![Vec::new(); reads.len()];
    let began = Instant::now();
    while began.elapsed().as_secs_f64() < LONE_SECONDS || seconds[0].is_empty() {
        for (volume, times) in volumes.iter().zip(&mut seconds) {
            let base = sink.bytes_mut().as_mut_ptr();
            let mut at = 0;
            let ranges = parts
                .iter()
                .map(|&part| {
                    let offset = random_offset(file, &mut state, part + ALIGNMENT) + state % (ALIGNMENT / GRAIN) * GRAIN;
                    // SAFETY: within the record's sink, `part` bytes from where the last part ended.
                    let sink = NonNull::new(unsafe { base.add(at) }).expect("a sink");
                    at += part as usize;
                    Range { offset, length: part as usize, sink }
                })
                .collect();
            let started = Instant::now();
            // SAFETY: each range's sink is its own bytes of `sink`, which
            // nothing else touches until the job, waited on here, ends.
            let job = unsafe { queue(file, volume, &outstanding, ranges, Priority::Demand) }?;
            match job.wait() {
                Ended::Read { failed } => {
                    if let Some((_, error)) = failed.first() {
                        return Err(Error::Invalid(format!("a lone request's read: {error}")));
                    }
                }
                Ended::Panicked(panic) => panic.resume(),
            }
            times.push(started.elapsed().as_secs_f64());
        }
    }
    Ok(seconds)
}
