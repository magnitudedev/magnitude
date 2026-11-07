//! # A volume's curve
//!
//! How fast a device reads depends on the bytes it holds in flight and on
//! the size of the reads that hold them, and how soon it serves a lone
//! request on how the request is split; no two devices measured agree
//! (`magnitude-artifacts`' `tests/read_depth`, uncached reads at random
//! offsets of Qwen3.6 35B's file):
//!
//! - GB10's NVMe: the rate is the bytes in flight's alone, 8 MiB reaching
//!   10.1–10.2 GB/s in reads of 128 KiB to 2 MiB and 2 MiB only 4.5–5.2; a
//!   1.9 MB record alone reads in 0.41 ms as one read, 0.65 ms as fifteen
//!   128 KiB reads at once.
//! - an RTX 3090 host (Xeon E5-2696 v3): 2 MiB in flight reads 2.84 GB/s in
//!   128 KiB reads and 3.47 in 512 KiB ones, and 128 KiB reads near the
//!   device's 3.6 only at 16 MiB; yet a record alone reads in 1.9 ms as
//!   fifteen 128 KiB reads, 3.8 ms as one.
//! - Mac internal drives: the rate tracks the reads in flight more than
//!   their bytes. An M4 Max reads stacked ranges at 6.48 GB/s 16 deep and
//!   6.73 32 deep, 256 KiB reads at 5.86 16 deep; an M4 Pro's stacked ranges
//!   rise from 4.21 GB/s 2 deep to 5.13 4 deep and are flat after.
//!
//! So only a volume can say how it is read best, and it is probed
//! ([`VolumeLines`]): a file of the volume is read uncached, its cached
//! pages dropped first, at each read size its request needs and every
//! window doubling from one read to [`WIDEST_WINDOW`], from as many threads
//! as hold the window, for [`PROBE_SECONDS`], in [`PASSES`] passes in
//! alternating order; each point is the median of its passes, with their
//! half range as its spread. One pass's points differ by up to 30% at the
//! knee and a few percent past it, so a decision from one pass flipped
//! between processes (GB10: 8 MiB in 1 MiB reads, then 16 MiB in 2 MiB).
//! Each volume is probed under its own lock, so one volume's probe holds no
//! other's reader.
//!
//! A step's request ([`Urgent`]) is the most one step's request reads,
//! `records` records of the model file each stored as separate ranges
//! (`parts`). Its read sizes run from [`LEAST_READ`] doubling to the least
//! that holds a record's widest range, at most [`WIDEST_WINDOW`], past
//! which no window holds a read; and its lone record is timed at each:
//! submitted to the read engine due in pieces of that size, as a step's own
//! reads are, so what it pays is the engine's (its cut, wakes and copies),
//! which threads reading the same sizes do not show (GB10, 200 lone experts
//! of three ranges each: 0.63–0.73 ms by threads at any size, 0.70–0.96
//! through the engine, 128 KiB the slowest). A reader no step waits on (a
//! load's import: no [`Urgent`]) probes [`LEAST_READ`] alone.
//!
//! The probe keeps the whole curve ([`Curve`]): every point it measured, with
//! its noise, and each lone time. Nothing is collapsed where it is measured;
//! the link's depths and read sizes ([`Shape`]) are derived from it where
//! they are used, by these rules (every "within the noise" is of the two
//! points' spreads, or a median's standard error):
//!
//! - **The knee** of a line: the narrowest window whose rate is within the
//!   noise of its best rate. M4 Max stacked ranges: 16 to 32 deep, by the
//!   noise; M4 Pro: 4 deep; GB10 above any decode depth.
//! - **The knee depth**: the most in flight, due and ahead, beside which a
//!   read ahead may start. Below it depth still adds rate, so a read ahead
//!   runs on throughput a due read would leave unused; past it every byte
//!   more in flight delays a due byte (an M4 Pro's internal drive: about
//!   0.12 ms a due read for each 640 KB range more). A request's is the knee
//!   of its **range line**: its own ranges, each one piece, read through the
//!   engine at every depth to 8 ranges and a quarter more each past it, to
//!   the first depth within the noise of the one before, as its reads ahead
//!   are read. A knee of other reads misplaces it: an M4 Pro's 1 MiB reads put
//!   it at 4 MiB where its 580–710 KiB ranges reach their rate 4 deep, about
//!   2.6 MB; and threads reading those ranges reach 5.1 GB/s where the
//!   engine reads a prompt's experts at 3.86, so only the engine's own
//!   reads say what its reads ahead are served at. Its reads ahead
//!   are cut at the least read size holding its widest range, so each is
//!   one piece, as the line read them.
//! - **The size read ahead** with no request: of the sizes that reach the
//!   volume's best rate, the one whose knee holds the fewest bytes, the
//!   wider of two that tie (fewer pieces, the same depth); its knee is the
//!   knee depth.
//! - **The size read due:** the one a lone request, the step's whole
//!   request ([`Urgent`]), is served soonest at through the engine, its
//!   least median: a park waits for every piece of its request, not for its
//!   first. Sizes within each other's noise
//!   cost it the same, so no tie rule picks among them: one that took the
//!   least such size by threads' timing chose 128 KiB demand reads on Mac
//!   Metal, slower than 256 KiB through the engine (capped decode +3.4%). A
//!   reader with no request reads due pieces at the size read ahead.
//! - **The due depth:** the least window holding a step's whole request,
//!   the depth its lone time was measured at, so none of its pieces waits
//!   on the host for another to end (Qwen3.6 on an M4 Pro's internal
//!   drive: its 1 MiB line peaks by 4 MiB, and a 4 MiB due depth decoded
//!   1.1% slower than 32 MiB; the read model gives +0.46 ms of the measured
//!   +0.47); at least the due size's widest window within the noise of its
//!   line's best rate, and the knee depth. Due reads only
//!   share the link with due reads, so a wider window costs nothing they
//!   would not wait for anyway, while serving lengthens each read's time in
//!   flight, so the knee measured alone falls short of the window a served
//!   prompt needs (GB10 CUDA's prompt fill beside the device's rounds:
//!   7.07 GB/s at 8 MiB, 7.50 at 32; capped 4k prefill 2160 to 2021 ms).
//!
//! A volume probed by the rule before this one keeps five figures instead
//! of its curve ([`EarlierChoice`]), which a load reads as they were while
//! the job probes the curve.
//!
//! The probe also measures the cores the reads occupy at every point, each
//! read copied out of a buffer as a piece that does not land straight is,
//! the median of its passes ([`read_cores`]): what the host keeps free for
//! the pool's workers, at the deepest the link reads at once.
//!
//! A reader measures nothing: the probe runs where nothing waits on it
//! ([`probe_volume`], the measurement job), a load adopts its curve
//! ([`adopt_volume`]), and until one exists a volume reads at the search's
//! bounds ([`Shape::PENDING`]).

use super::file::SourceFile;
use super::platform::{process_cpu_seconds, thread_cpu_seconds};
use super::volume::{EarlierChoice, Shape, VolumeReads};
#[cfg(doc)]
use super::volume::{adopt_volume, probe_volume, read_cores};
use super::ALIGNMENT;
use crate::scheduler::{self, request, Ended, Range, Want};
use crate::{Buffer, Volume, Error};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// The least read size a volume's probe reads in, doubling up to the least
/// that holds a record's widest range, at most the widest window ("A
/// volume's curve"): the grain the pool's workers are counted in, and below
/// which no device measured read faster or sooner (`magnitude-artifacts`'
/// `tests/read_depth`).
pub(crate) const LEAST_READ: u64 = 128 * 1024;

/// The widest window a volume's probe holds in flight: twice the widest
/// any measured device needed to reach its rate (an RTX 3090 host's 128 KiB
/// reads, 16 MiB).
pub(crate) const WIDEST_WINDOW: u64 = 32 * 1024 * 1024;

/// The depth, in ranges, to which a range line is probed at every depth
/// ([`Urgent::depths`]): past every knee measured on a drive whose rate
/// tracks its reads in flight (an M4 Pro's internal drive, 3 to 4).
const EVERY_DEPTH_TO: u64 = 8;

/// The seconds a volume's probe reads at each read size and window, and
/// the passes it takes each point's median of.
const PROBE_SECONDS: f64 = 0.03;
const PASSES: usize = 3;

/// The seconds a probe times lone records in, in rounds of one record at
/// each read size, so drift is shared within a round: on GB10 a record
/// alone takes 0.4–0.9 ms, so about 50 rounds of five sizes.
const LONE_SECONDS: f64 = 0.2;

/// The request a reader's steps wait on: `records` records of its file,
/// each stored as ranges of `parts` bytes (an expert's gate, up and down,
/// separate tensors), what one row's request at a step reads ("A volume's
/// curve").
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Urgent {
    pub parts: Vec<u64>,
    pub records: u64,
}

impl Urgent {
    /// The bytes it reads: every record's ranges.
    pub(crate) fn bytes(&self) -> u64 {
        self.records * self.parts.iter().sum::<u64>()
    }

    /// A range's bytes, the mean of its ranges: what one range of its range
    /// line holds in flight.
    pub(crate) fn range(&self) -> u64 {
        self.parts.iter().sum::<u64>() / self.parts.len() as u64
    }

    /// The ranges in flight its range line is probed at: every depth to
    /// [`EVERY_DEPTH_TO`], then a quarter more each, while its widest
    /// ranges, counted at least [`LEAST_READ`] (the grain the read sizes'
    /// lines are deepest in), fit the widest window. A knee is admitted in
    /// whole ranges, so a coarser grid places it past the depth that reaches
    /// the rate: doubling put an M4 Pro's at 4 ranges, where 3 read as fast
    /// (4.07 against 4.17 GB/s through the engine) and each 580 KiB range
    /// took 576 µs against 437.
    pub fn depths(&self) -> Vec<u64> {
        let widest = self.parts.iter().copied().max().expect("a request has a range").max(LEAST_READ);
        std::iter::successors(Some(1), |&depth| Some(if depth < EVERY_DEPTH_TO { depth + 1 } else { depth + depth / 4 }))
            .take_while(|depth| depth * widest <= WIDEST_WINDOW)
            .collect()
    }

    /// The least read size holding its widest range whole: its reads ahead
    /// are each one piece of it, as its range line read them.
    fn whole_range(&self) -> u64 {
        *self.reads().last().expect("a request reads at least one size")
    }

    /// The least window of a probe's grid that holds it whole, at most
    /// [`WIDEST_WINDOW`]: the due depth at which every piece it is cut
    /// into is in flight at once.
    fn whole(&self) -> u64 {
        self.bytes().max(LEAST_READ).next_power_of_two().min(WIDEST_WINDOW)
    }

    /// The read sizes it is probed at: from [`LEAST_READ`] doubling to the
    /// least that holds its widest range, past which a read reads nothing
    /// more of a record, or to [`WIDEST_WINDOW`], past which no window of
    /// the probe holds one.
    pub(crate) fn reads(&self) -> Vec<u64> {
        let widest = self.parts.iter().copied().max().expect("a request has a range").max(LEAST_READ).next_power_of_two();
        std::iter::successors(Some(LEAST_READ), |read| Some(read * 2)).take_while(|read| *read <= widest.min(WIDEST_WINDOW)).collect()
    }
}

/// The read sizes a probe for `urgent` reads ([`Urgent::reads`]; a reader
/// no step waits on, [`LEAST_READ`] alone).
fn reads_of(urgent: Option<&Urgent>) -> Vec<u64> {
    urgent.map_or(vec![LEAST_READ], Urgent::reads)
}

/// What a volume's probe for one request measured ("A volume's curve"):
/// each point of its read sizes' lines, and its lone record's time at each
/// read size and its range line (none for a reader no step waits on).
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    pub points: Vec<Point>,
    pub lone: Vec<Lone>,
    pub ranges: Vec<Point>,
}

impl Curve {
    /// The points, as (read size, window), the read sizes of lone times,
    /// and the depths the range line a probe for `urgent` may reach, in
    /// order (it ends at its first flat depth): what a stored curve holds.
    pub fn grid(urgent: Option<&Urgent>) -> (Vec<(u64, u64)>, Vec<u64>, Vec<u64>) {
        let reads = reads_of(urgent);
        let (lone, depths) = urgent.map_or((Vec::new(), Vec::new()), |urgent| (reads.clone(), urgent.depths()));
        (grid(reads.iter().copied()), lone, depths)
    }

    /// The point of `urgent`'s range line `depth` ranges deep: its window
    /// the bytes of that many ranges.
    pub fn range_point(urgent: &Urgent, depth: u64, rate: f64, spread: f64, cores: f64) -> Point {
        let range = urgent.range();
        Point { read: range, window: depth * range, rate, spread, cores }
    }
}

/// One read size and window of a volume: the median rate (bytes a second)
/// and cores of its passes, and the rates' half range, its spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub read: u64,
    pub window: u64,
    pub rate: f64,
    pub spread: f64,
    pub cores: f64,
}

impl Point {
    fn of(read: u64, window: u64, passes: &[(f64, f64)]) -> Self {
        let rates = passes.iter().map(|&(rate, _)| rate).collect::<Vec<_>>();
        let (least, most) = rates.iter().fold((f64::INFINITY, 0f64), |(least, most), &rate| (least.min(rate), most.max(rate)));
        let cores = median(&passes.iter().map(|&(_, cores)| cores).collect::<Vec<_>>());
        Self { read, window, rate: median(&rates), spread: (most - least) / 2., cores }
    }

}

/// A lone request's time through the engine in pieces of `read` bytes, the
/// whole of what a step's request reads ([`Urgent`]): the median seconds of
/// its rounds and that median's standard error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lone {
    pub read: u64,
    pub seconds: f64,
    pub error: f64,
}

/// A volume's curve as probed or adopted, and how each request reads it
/// ("A volume's curve"): each read size's line, with the bytes of the file
/// it was probed through; and what each request last adopted.
#[derive(Default)]
pub(crate) struct VolumeLines {
    lines: BTreeMap<u64, (u64, Vec<Point>)>,
    requests: HashMap<Option<Urgent>, Adopted>,
}

/// What a request last adopted: its curve (its read sizes' points as
/// probed for it, its lone times and range line), which alone sets how it
/// reads, whatever other requests of the volume adopt after it; or the rule
/// before's five figures, where only those are stored.
enum Adopted {
    Curve(Curve),
    Earlier(EarlierChoice),
}

impl VolumeLines {
    /// How `urgent` reads the volume: by the curve it last adopted; else by
    /// the lines other requests hold, its due pieces at the size read ahead;
    /// else at the search's bounds.
    pub(crate) fn shape(&self, urgent: Option<&Urgent>) -> Shape {
        match self.requests.get(&urgent.cloned()) {
            Some(Adopted::Earlier(earlier)) => earlier.shape(),
            Some(Adopted::Curve(curve)) => {
                Shape::of(&curve.points, urgent.map(|urgent| Request { urgent, lone: &curve.lone, ranges: &curve.ranges }))
            }
            None => {
                let points = self.lines.values().flat_map(|(_, points)| points.iter().copied()).collect::<Vec<_>>();
                if points.is_empty() { Shape::PENDING } else { Shape::of(&points, None) }
            }
        }
    }

    /// Hold `curve`, probed for `urgent` through a file of `size` bytes:
    /// each of its lines replaces the volume's of that read size, and the
    /// request reads by the curve.
    pub(crate) fn adopt(&mut self, urgent: Option<&Urgent>, curve: &Curve, size: u64) {
        let mut lines = BTreeMap::<u64, Vec<Point>>::new();
        for point in &curve.points {
            lines.entry(point.read).or_default().push(*point);
        }
        for (read, points) in lines {
            self.lines.insert(read, (size, points));
        }
        self.requests.insert(urgent.cloned(), Adopted::Curve(curve.clone()));
    }

    /// Read `urgent` by the rule before's `choice`.
    pub(crate) fn adopt_earlier(&mut self, urgent: Option<&Urgent>, choice: EarlierChoice) {
        self.requests.insert(urgent.cloned(), Adopted::Earlier(choice));
    }

    /// Probe `urgent`'s curve through `file`: the lines it needs not yet
    /// probed through a file as large, its lone times and its range line.
    pub(crate) fn probe(&mut self, key: Volume, file: &Arc<SourceFile>, urgent: Option<&Urgent>) -> Result<Curve, Error> {
        self.probe_lines(file, &reads_of(urgent))?;
        let (lone, ranges) = match urgent {
            Some(urgent) => {
                let reads = urgent.reads();
                let seconds = lone(key, file, urgent, &reads)?;
                let lone = reads.iter().zip(&seconds).map(|(&read, seconds)| Lone { read, seconds: median(seconds), error: median_error(seconds) }).collect();
                (lone, range_line(key, file, urgent)?)
            }
            None => (Vec::new(), Vec::new()),
        };
        let points = reads_of(urgent).iter().flat_map(|read| self.lines.get(read).into_iter().flat_map(|(_, points)| points.iter().copied())).collect();
        let curve = Curve { points, lone, ranges };
        self.requests.insert(urgent.cloned(), Adopted::Curve(curve.clone()));
        Ok(curve)
    }

    /// Probe `reads`' lines not yet probed through a file as large as
    /// `file`: every window doubling from one read to [`WIDEST_WINDOW`], in
    /// [`PASSES`] passes over the points in alternating order. A line a
    /// smaller file was probed through (a draft model's, probed first) is
    /// probed again: the device's own caches may hold more of a small file
    /// than its reads would of the model's.
    fn probe_lines(&mut self, file: &SourceFile, reads: &[u64]) -> Result<(), Error> {
        let grid = grid(reads.iter().copied().filter(|read| self.lines.get(read).is_none_or(|(size, _)| *size < file.size)));
        let mut passes = vec![Vec::with_capacity(PASSES); grid.len()];
        for pass in 0..PASSES {
            for at in 0..grid.len() {
                let at = if pass % 2 == 0 { at } else { grid.len() - 1 - at };
                let (read, window) = grid[at];
                passes[at].push(rate(file, read, (window / read) as usize)?);
            }
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

/// A request as its probe measured it: its lone times and its range line.
pub(crate) struct Request<'a> {
    pub(crate) urgent: &'a Urgent,
    pub(crate) lone: &'a [Lone],
    pub(crate) ranges: &'a [Point],
}

impl Shape {
    /// How a volume of `points` is read ("A volume's curve"): for a
    /// request, its due pieces at the size it is read whole soonest at, at a
    /// depth holding it whole, and its reads ahead in whole ranges to its
    /// range line's knee; with none, at the size read ahead to its knee.
    pub(crate) fn of(points: &[Point], request: Option<Request>) -> Self {
        let top = *points.iter().max_by(|a, b| a.rate.total_cmp(&b.rate)).expect("a probed point");
        let at_best = |point: &Point| point.rate + point.spread + top.spread >= top.rate;
        let line = |read: u64| {
            let mut line = points.iter().copied().filter(|point| point.read == read).collect::<Vec<_>>();
            line.sort_by_key(|point| point.window);
            line
        };
        let mut reads = points.iter().map(|point| point.read).collect::<Vec<_>>();
        reads.sort_unstable();
        reads.dedup();
        let cores_at = |line: &[Point], window: u64| line.iter().filter(|point| point.window <= window).map(|point| point.cores).fold(0., f64::max);
        let (ahead_read, knee, ahead_cores) = match &request {
            // The range line's knee, its reads ahead whole ranges.
            Some(request) => {
                let knee = knee(request.ranges);
                (request.urgent.whole_range(), knee.window, cores_at(request.ranges, knee.window))
            }
            // Each size reaching the best rate, at its knee.
            None => {
                let (read, knee) = reads
                    .iter()
                    .filter_map(|&read| line(read).iter().find(|point| at_best(point)).map(|knee| (read, knee.window)))
                    .min_by_key(|&(read, knee)| (knee, std::cmp::Reverse(read)))
                    .expect("the best point reaches its own rate");
                (read, knee, cores_at(&line(read), knee))
            }
        };
        let due_read = request.as_ref().map_or(ahead_read, |request| request.lone.iter().min_by(|a, b| a.seconds.total_cmp(&b.seconds)).expect("a request's lone times").read);
        let due_line = line(due_read);
        let full = full_rate(&due_line);
        Self {
            due: full.window.max(knee).max(request.as_ref().map_or(0, |request| request.urgent.whole())),
            knee,
            ahead_read,
            due_read,
            cores: cores_at(&due_line, full.window).max(ahead_cores),
        }
    }
}

/// Of a line's points, its knee's: the narrowest window within the noise of
/// the line's best rate ("A volume's curve": the knee depth).
fn knee(line: &[Point]) -> Point {
    let top = line.iter().max_by(|a, b| a.rate.total_cmp(&b.rate)).expect("a probed range line");
    *line
        .iter()
        .filter(|point| point.rate + point.spread + top.spread >= top.rate)
        .min_by_key(|point| point.window)
        .expect("the best is within its own noise")
}

/// Of a line's points, the full rate's: the widest window within the noise
/// of the line's best rate ("A volume's curve": the due depth).
pub(crate) fn full_rate(line: &[Point]) -> Point {
    let top = line.iter().max_by(|a, b| a.rate.total_cmp(&b.rate)).expect("a probed point");
    *line
        .iter()
        .filter(|point| point.rate + point.spread + top.spread >= top.rate)
        .max_by_key(|point| point.window)
        .expect("the best is within its own noise")
}

/// The points a probe of read sizes `reads` reads: each size at every
/// window doubling from one read to [`WIDEST_WINDOW`].
pub(crate) fn grid(reads: impl IntoIterator<Item = u64>) -> Vec<(u64, u64)> {
    reads
        .into_iter()
        .flat_map(|read| std::iter::successors(Some(read), |window| Some(window * 2)).take_while(|window| *window <= WIDEST_WINDOW).map(move |window| (read, window)))
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
                let (mut buffer, mut sink) = (Buffer::new(length), vec![0u8; length]);
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

/// Lone requests of `file`, each `urgent`'s whole request (its `records`
/// records, each its `parts`, each part at a random offset of its own,
/// unaligned at the 32-byte grain a model file stores tensors at, landing
/// back to back in one sink as a bank's records do): a park waits for every
/// piece of its request, so the size it is read whole soonest at is its
/// due size. Each is submitted alone to the read engine as a due request of
/// volume `key` read in pieces of each size of `reads`, as a step's own
/// reads are; in rounds of one request at each size, for [`LONE_SECONDS`]:
/// their seconds, by size and round. What a request pays is the engine's
/// (its cut into pieces, the workers it wakes, the copy of a piece that
/// does not land straight), not a read's alone.
fn lone(key: Volume, file: &Arc<SourceFile>, urgent: &Urgent, reads: &[u64]) -> Result<Vec<Vec<f64>>, Error> {
    const GRAIN: u64 = 32;
    let parts = &urgent.parts;
    let record = parts.iter().sum::<u64>();
    assert!(record + ALIGNMENT * parts.len() as u64 <= file.size, "a record of {record} bytes beyond a file of {}", file.size);
    let sink: Arc<dyn crate::Destination> = Arc::new(Buffer::new(urgent.bytes() as usize));
    // Every piece of a request at once: its due depth holds it whole.
    let volumes = reads
        .iter()
        .map(|&read| VolumeReads { id: key, shape: Shape { ahead_read: read, due_read: read, ..Shape::PENDING } })
        .collect::<Vec<_>>();
    scheduler::deepen(volumes.iter().map(|volume| volume.shape.workers()).max().expect("a read size"))?;
    let outstanding = Arc::default();
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut seconds = vec![Vec::new(); reads.len()];
    let began = Instant::now();
    while began.elapsed().as_secs_f64() < LONE_SECONDS || seconds[0].is_empty() {
        for (volume, times) in volumes.iter().zip(&mut seconds) {
            let mut at = 0;
            let ranges = (0..urgent.records)
                .flat_map(|_| parts.iter())
                .map(|&part| {
                    let offset = random_offset(file, &mut state, part + ALIGNMENT) + state % (ALIGNMENT / GRAIN) * GRAIN;
                    let range = Range { offset, length: part as usize, into: sink.clone(), at };
                    at += part as usize;
                    range
                })
                .collect();
            let started = Instant::now();
            // SAFETY: each range's bytes are its own of `sink`, which
            // nothing else touches until the request, waited on here, ends.
            let request = unsafe { request(file, volume, &outstanding, ranges, Want::now()) }?;
            match request.wait() {
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

/// `urgent`'s range line through `file` ("A volume's curve": the knee
/// depth): at each of its depths in turn, the rate and cores of that many
/// of its ranges kept in flight through the read engine, the median of
/// [`PASSES`] passes, to the first depth within the noise of the one before:
/// the knee is the narrowest window within the noise of the line's best, so
/// no deeper point moves it, and a drive whose knee is 3 ranges is probed at
/// 4 depths.
fn range_line(key: Volume, file: &Arc<SourceFile>, urgent: &Urgent) -> Result<Vec<Point>, Error> {
    let mut line = Vec::<Point>::new();
    for depth in urgent.depths() {
        let passes = (0..PASSES).map(|_| ranges_rate(key, file, urgent, depth)).collect::<Result<Vec<_>, _>>()?;
        let point = Point::of(urgent.range(), depth * urgent.range(), &passes);
        let flat = line.last().is_some_and(|before| point.rate <= before.rate + before.spread + point.spread);
        line.push(point);
        if flat {
            break;
        }
    }
    Ok(line)
}

/// The bytes a second `file` reads at, and the cores the process runs,
/// with `depth` of `urgent`'s ranges in flight through the read engine:
/// `depth` threads, each submitting one range at a time as a due request of
/// volume `key`, its whole range one piece, at a random offset unaligned at
/// the 32-byte grain a model file stores tensors at, for [`PROBE_SECONDS`]
/// from when every one has started. What a range pays is the engine's (its wakes and the copy of a piece
/// that does not land straight), as a read ahead's is.
fn ranges_rate(key: Volume, file: &Arc<SourceFile>, urgent: &Urgent, depth: u64) -> Result<(f64, f64), Error> {
    const GRAIN: u64 = 32;
    let parts = &urgent.parts;
    let widest = *parts.iter().max().expect("a request has a range");
    let volume = VolumeReads { id: key, shape: Shape { ahead_read: urgent.whole_range(), due_read: urgent.whole_range(), ..Shape::PENDING } };
    scheduler::deepen(volume.shape.workers())?;
    let outstanding = Arc::default();
    let (volume, outstanding) = (&volume, &outstanding);
    // Where every thread reads until, once all have started; none where one
    // could not, which ends the others at once.
    let start = (Mutex::new(None::<Option<Instant>>), Condvar::new());
    let start = &start;
    let cpu = process_cpu_seconds()?;
    let bytes = std::thread::scope(|scope| -> Result<(u64, Instant), Error> {
        let reader = |thread: u64| {
            move || -> Result<u64, Error> {
                let sink: Arc<dyn crate::Destination> = Arc::new(Buffer::new(widest as usize));
                let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ (thread + 1).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                let (mut bytes, mut part) = (0, thread as usize % parts.len());
                let until = *start.1.wait_while(start.0.lock().expect("probe start"), |until| until.is_none()).expect("probe start");
                let until = until.expect("the probe started");
                while until.is_some_and(|until| Instant::now() < until) {
                    let length = parts[part];
                    part = (part + 1) % parts.len();
                    let offset = random_offset(file, &mut state, length + ALIGNMENT) + state % (ALIGNMENT / GRAIN) * GRAIN;
                    let ranges = vec![Range { offset, length: length as usize, into: sink.clone(), at: 0 }];
                    // SAFETY: the range's bytes are this thread's sink's,
                    // which nothing else touches until the request, waited
                    // on here, ends.
                    match unsafe { request(file, volume, outstanding, ranges, Want::now()) }?.wait() {
                        Ended::Read { failed } => {
                            if let Some((_, error)) = failed.first() {
                                return Err(Error::Invalid(format!("a range line's read: {error}")));
                            }
                        }
                        Ended::Panicked(panic) => panic.resume(),
                    }
                    bytes += length;
                }
                Ok(bytes)
            }
        };
        let spawned = (0..depth).map(|thread| std::thread::Builder::new().spawn_scoped(scope, reader(thread))).collect::<Result<Vec<_>, _>>();
        let began = Instant::now();
        *start.0.lock().expect("probe start") = Some(spawned.as_ref().ok().map(|_| began + std::time::Duration::from_secs_f64(PROBE_SECONDS)));
        start.1.notify_all();
        let threads = spawned.map_err(Error::ThreadStart)?;
        let bytes = threads.into_iter().map(|thread| thread.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))).sum::<Result<u64, Error>>()?;
        Ok((bytes, began))
    })?;
    let (bytes, began) = bytes;
    let wall = began.elapsed().as_secs_f64();
    Ok((bytes as f64 / wall, (process_cpu_seconds()? - cpu) / wall))
}
