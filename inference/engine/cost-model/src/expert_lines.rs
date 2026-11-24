//! The one cost function of where a missed expert is computed: what a
//! streamed layer's misses add to its step, given the
//! GPU's hits, from lines this machine measured. The route's assignment
//! (through the policy words the host writes between flights,
//! [`ExpertLines::policy`], which the route computes this function from:
//! streaming_words.seismic's assignment), the bank's sizing
//! and assessment's streamed estimate (both
//! [`crate::StreamedLayers::step_seconds`]), a prompt group's stages, which its
//! size, merge sets and places are formed by (the executor's `streaming`), and the
//! harness's bar all price a layer by [`ExpertLines::serve`]; nothing else
//! prices one.
//!
//! An expert of `r` rows a layer's route chose costs, by where it is
//! computed:
//! - **the CPU**, from host memory: `t_cpu(r)` ([`CpuLines`], measured by
//!   the CPU pool's own kernel), the CPU's first of a step adding the
//!   handshake;
//! - **the GPU in place**, its record read where the host tier holds it
//!   (over the link on a discrete GPU): `t_link(r)`;
//! - **the GPU after a move**, its record's bytes moved over the link by
//!   device work behind the pass's wait (a prompt group's fill from the
//!   tier, or its converts' reads of the conveyor's records), then computed
//!   from the GPU's memory: `t_copy + t_gpu(r)`, `t_copy` the link line's
//!   record (`GpuLines::link`'s weights: the same link bytes), none on
//!   unified memory, whose bank has no host tier to move from.
//!
//! Which sites a miss may take is the pass's graph's ([`Pass`]): a decode
//! step's route assigns each miss to the CPU or in place; a prompt group's
//! pass moves every cold expert and computes it resident.
//!
//! The GPU's lines are its kernels' measured costs (`ExecutorCosts`) at the
//! expert's bytes and arithmetic, each storage's own route ([`GpuLines`]).
//!
//! A layer's misses can be served only after its router, and the only GPU
//! work of the layer that does not wait on them is its hits. Each distinct
//! missed expert, the most rows first (ties in their order), goes to the
//! site it ends soonest on, the CPU's queue from the router and the GPU's
//! from its hits, its reads in place no sooner than the link is free of the
//! step's moves beside them ([`Moves`]); the CPU wins a tie, then the link
//! (principle 2). So each miss goes where the makespan of the GPU, the link
//! and the CPU ends least, host memory's reads being the same whichever
//! serves it. The layer then adds
//!
//! ```text
//! exposed = end − hits
//! ```
//!
//! to its step: nothing done after the router hides more than the hits. Its
//! parts ([`Exposed`]) are what held the end: the GPU's queue past its hits
//! (`gpu_end − hits`), and the CPU's past the GPU's (`max(0, cpu_end −
//! gpu_end)`), split into its experts' part and the handshake its queue
//! carries, the rest.
//!
//! Every line is in whole nanoseconds ([`Nanoseconds`]), each within a
//! policy word, so at whole rows the function's sums are exact integers, the
//! same the route's u64 arithmetic computes from the policy words: the host
//! states the rule, the route computes it, and a host test runs the route's
//! reference body against [`ExpertLines::serve`].

use crate::lines::{FileLine, ReadLine, RecordLine};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A line's whole nanoseconds, the nearest to what was measured, at most a
/// policy word's (`i32::MAX`, 2.1 s): every line of the cost function, so
/// the route reads each as it is ([`ExpertLines::policy`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct Nanoseconds(u32);

/// Why seconds make no line: not finite and non-negative, or past a policy
/// word.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineError(pub f64);

impl fmt::Display for LineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} s makes no line of whole nanoseconds within a policy word", self.0)
    }
}

impl std::error::Error for LineError {}

impl Nanoseconds {
    /// The whole nanoseconds nearest `seconds`.
    pub fn of(seconds: f64) -> Result<Self, LineError> {
        let nanoseconds = (seconds * 1e9).round();
        match nanoseconds >= 0. && nanoseconds <= i32::MAX as f64 {
            true => Ok(Self(nanoseconds as u32)),
            false => Err(LineError(seconds)),
        }
    }

    /// The whole nanoseconds nearest `seconds`, a line's term (finite and
    /// not negative, which a fit's terms are by construction), at most a
    /// policy word's: a term past a word is priced at the most a word says,
    /// the dearest the route weighs.
    pub fn saturating(seconds: f64) -> Self {
        assert!(seconds.is_finite() && seconds >= 0., "a line's term of {seconds} s");
        Self((seconds * 1e9).round().min(i32::MAX as f64) as u32)
    }

    /// As a float, in nanoseconds: exact.
    pub fn get(self) -> f64 {
        f64::from(self.0)
    }

    /// As its policy word.
    pub fn word(self) -> i32 {
        self.0 as i32
    }
}

impl TryFrom<u32> for Nanoseconds {
    type Error = LineError;

    fn try_from(nanoseconds: u32) -> Result<Self, LineError> {
        match nanoseconds <= i32::MAX as u32 {
            true => Ok(Self(nanoseconds)),
            false => Err(LineError(f64::from(nanoseconds) * 1e-9)),
        }
    }
}

impl From<Nanoseconds> for u32 {
    fn from(nanoseconds: Nanoseconds) -> Self {
        nanoseconds.0
    }
}

/// A CPU line in rows: an expert of `r` rows takes `one + (r − 1) · row`,
/// `row` its cost's bound ([`RowCost::bound`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RowLine {
    pub one: Nanoseconds,
    pub row: RowCost,
}

impl RowLine {
    /// An expert's nanoseconds at `rows` rows (a mean, where it is one).
    pub fn nanoseconds(&self, rows: f64) -> f64 {
        self.one.get() + (rows - 1.) * self.row.bound().get()
    }
}

/// What a row costs the CPU, as measured: a slope above its points'
/// resolution, or one within it, which the measurement bounds by the
/// resolution and no finer. Either is priced at its bound: the slope, or
/// the most the measurement admits.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum RowCost {
    Measured(Nanoseconds),
    AtMost(Nanoseconds),
}

impl RowCost {
    /// What the cost function prices a row at.
    pub fn bound(self) -> Nanoseconds {
        match self {
            Self::Measured(row) | Self::AtMost(row) => row,
        }
    }
}

/// A measured request: the fastest of its samples, each on records no
/// sample before it read, and the resolution of that fastest, the second
/// fastest less it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub seconds: f64,
    pub resolution: f64,
}

impl Sample {
    /// The sample of `seconds`, a request's times; none of fewer than two,
    /// whose fastest has no resolution.
    pub fn of(seconds: &[f64]) -> Option<Self> {
        let mut sorted = seconds.to_vec();
        sorted.sort_by(f64::total_cmp);
        match sorted[..] {
            [fastest, next, ..] => Some(Self { seconds: fastest, resolution: next - fastest }),
            _ => None,
        }
    }
}

/// A GPU line: an expert of `r` rows takes the slower of reading its record
/// and its arithmetic, `max(weights, r · row)`, as the GPU's kernels stream
/// the one and compute the other at once.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PassLine {
    pub weights: Nanoseconds,
    pub row: Nanoseconds,
}

impl PassLine {
    /// An expert's nanoseconds at `rows` rows.
    pub fn nanoseconds(&self, rows: f64) -> f64 {
        self.weights.get().max(rows * self.row.get())
    }

    /// The line of an expert whose record reads in `weights` seconds and
    /// whose arithmetic takes `row` seconds a row.
    pub fn of(weights: f64, row: f64) -> Result<Self, LineError> {
        Ok(Self { weights: Nanoseconds::of(weights)?, row: Nanoseconds::of(row)? })
    }
}

/// The CPU pool's lines over a layer's records: an expert (its rows read
/// from host memory), and the handshake a request adds whatever it
/// computes: the device's request noticed and its answer published, as a
/// request of no entries takes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CpuLines {
    pub expert: RowLine,
    pub handshake: Nanoseconds,
}

/// Why the CPU's requests make no lines.
#[derive(Clone, Debug, PartialEq)]
pub enum CpuLinesError {
    /// The pool could not answer its requests.
    Unmeasured(String),
    /// A layer's lines take an expert and two rows; it has `experts`
    /// choices and `rows` rows.
    Narrow { experts: usize, rows: usize },
    /// Its points `(experts, rows, seconds)`, beyond the handshake's
    /// seconds, give an expert no cost.
    Falling { handshake: f64, points: [(usize, usize, f64); 3] },
    /// A line past a policy word.
    Line(LineError),
}

impl fmt::Display for CpuLinesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unmeasured(reason) => write!(formatter, "the CPU pool answered no request: {reason}"),
            Self::Narrow { experts, rows } => {
                write!(formatter, "a layer of {experts} choices and {rows} rows makes no CPU line")
            }
            Self::Falling { handshake, points } => {
                write!(formatter, "the CPU's points {points:?} beyond a handshake of {handshake} s give an expert no cost")
            }
            Self::Line(error) => write!(formatter, "a CPU line: {error}"),
        }
    }
}

impl std::error::Error for CpuLinesError {}

/// A measurement of CPU pool requests ([`CpuLines::measured`]): each
/// request's `(experts, rows)` to its [`Sample`], in order.
pub type Measure<'a> = dyn Fn(&[(usize, usize)]) -> Result<Vec<Sample>, String> + 'a;

impl CpuLines {
    /// The lines through requests the pool answered (`measure`: each
    /// request's `(experts, rows)`, each expert chosen by that many rows,
    /// its fastest seconds from its publication to its answer and their
    /// resolution, [`Sample`]). The handshake is a request of no entries,
    /// measured as itself. The lines are taken at the
    /// decode's operating point, the handshake removed: an expert's seconds
    /// are what `experts` experts of one row take beyond it, each; a row's
    /// what one expert of `rows` rows (the widest decode's, so the slope's
    /// signal is the largest the shape has) takes beyond one of one row,
    /// each, measured where it exceeds the two points' resolutions over the
    /// rows between, else at most that ([`RowCost`]). Only an expert of no
    /// cost refuses the lines.
    pub fn measured(
        experts: usize,
        rows: usize,
        measure: &Measure<'_>,
    ) -> Result<Self, CpuLinesError> {
        if experts == 0 || rows < 2 {
            return Err(CpuLinesError::Narrow { experts, rows });
        }
        let requests = [(1, 1), (experts, 1), (1, rows)];
        let samples = measure(&[(0, 0), requests[0], requests[1], requests[2]]).map_err(CpuLinesError::Unmeasured)?;
        let handshake = samples[0].seconds;
        let ns = |seconds| Nanoseconds::of(seconds).map_err(CpuLinesError::Line);
        let line = |samples: &[Sample]| {
            let expert = (samples[1].seconds - handshake) / experts as f64;
            if expert.is_nan() || expert <= 0. {
                let points = std::array::from_fn(|at| (requests[at].0, requests[at].1, samples[at].seconds));
                return Err(CpuLinesError::Falling { handshake, points });
            }
            let between = (rows - 1) as f64;
            let row = (samples[2].seconds - samples[0].seconds) / between;
            let resolution = (samples[2].resolution + samples[0].resolution) / between;
            let row = match row > resolution {
                true => RowCost::Measured(ns(row)?),
                false => RowCost::AtMost(ns(resolution)?),
            };
            Ok(RowLine { one: ns(expert)?, row })
        };
        Ok(Self { expert: line(&samples[1..])?, handshake: ns(handshake)? })
    }
}

/// The GPU's lines over a layer's expert: its record read in place where
/// the host tier holds it (over the link; on unified memory, the GPU's own
/// memory), and from the GPU's memory; and the moves the link carries
/// beside a step's reads of the layer ([`Moves`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuLines {
    pub link: PassLine,
    pub resident: PassLine,
    pub moves: Moves,
}

/// The moves (promotions: host tier records copied into the GPU's memory)
/// beside a layer's reads, a decode step's share of a layer, as the load
/// measures them: the time they hold the link (`link`), which the layer's
/// reads in place wait behind, and the records they read from host memory
/// (`records`), which its CPU and link reads share. The copy engine moves
/// them whatever layer the step is at, so each layer bears an even share.
/// None where nothing moves: unified memory, which has no link, and a load
/// before it measured a move.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Moves {
    pub link: Nanoseconds,
    pub records: f64,
}

impl Moves {
    /// No move.
    pub const NONE: Self = Self { link: Nanoseconds(0), records: 0. };
}

/// A streamed layer's lines on the host's side: the bank's (the executor's `BankLines`),
/// and where the bank has a host tier, one tier record's read from host
/// memory at the memory's whole read rate (every CPU reading,
/// the executor's `MemoryCosts`), none without one (unified memory's bank has no
/// tier).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostLines {
    pub cpu: Option<CpuLines>,
    pub memory: Option<Nanoseconds>,
    /// One tier record's read time (`b / B_joint`) from the shared `HostRam`
    /// bus under concurrent `DMA ∥ CPU` traffic, when measured by the
    /// topology-grounded contention probe (`TierCopyCosts::concurrent_host_byte_seconds`).
    ///
    /// `None` on unified memory (where `memory` is also `None`) or before the
    /// concurrent probe has been recorded, in which case shared-bus contention
    /// falls back conservatively to `memory` (`B_joint <= B_mem`).
    #[serde(default)]
    pub joint_memory: Option<Nanoseconds>,
    /// What a request's file misses add past their reads' end as they land,
    /// at the one site its bank lands them ([`LandingLine`]); none before
    /// any line of that site is had.
    pub landing: Option<LandingLine>,
    pub record_bytes: u64,
}

impl HostLines {
    /// One tier record's read time (`b / B_joint`) on the shared `HostRam` bus
    /// under concurrent `DMA ∥ CPU` traffic: the measured `joint_memory` (at
    /// least `memory`, since concurrent traffic cannot exceed uncontended
    /// `HostRam` bandwidth), or `memory` as the conservative analytical
    /// fallback when `joint_memory` is unprobed. `None` on unified memory
    /// (`memory == None`).
    pub fn contended_memory(&self) -> Option<Nanoseconds> {
        let mem = self.memory?;
        Some(self.joint_memory.map_or(mem, |joint| joint.max(mem)))
    }
}

/// Where a layer's file misses land, by who places its bank's records, and
/// the line of what that adds past their reads' end:
/// - `Host`: into a host tier beside a discrete device, the host's landing
///   line ([`FileLine`]): `n` landings `fixed + n · record` past the last
///   read.
/// - `Device`: into the bank on unified memory, by the step's landing
///   launch (`routed_land`), which converts each record as its read lands:
///   past the last read byte the last record's conversion remains, one
///   record of the device's convert line ([`RecordLine::seconds`] of 1),
///   unless conversion is the slower of the two, when it also holds what
///   `count` records' conversions run past the reads' (the first record
///   lands, and conversion begins, one record's read in).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum LandingLine {
    Host(FileLine),
    Device(RecordLine),
}

impl LandingLine {
    /// What `count` file misses requested together (one or more) add past
    /// their reads' end, each read taking `record_read_seconds` after the
    /// one before. `count` is at least one: nothing missed lands nothing.
    pub fn seconds(&self, count: u64, record_read_seconds: f64) -> f64 {
        assert!(count > 0, "a landing prices at least one miss");
        match self {
            Self::Host(file) => file.seconds(count),
            Self::Device(convert) => {
                let behind = convert.seconds(count) - (count - 1) as f64 * record_read_seconds;
                behind.max(convert.seconds(1))
            }
        }
    }
}

/// Where the route's policy words lie (`streaming_words.seismic`'s policy
/// after its home): how many there are, and the offset of each word the
/// cost function writes. The kernels' sources own the layout; the executor
/// states it from their constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolicyLayout {
    pub words: usize,
    pub cpu: usize,
    pub cpu_row: usize,
    pub handshake: usize,
    pub link: usize,
    pub link_row: usize,
    pub link_busy: usize,
    pub hit: usize,
    pub hit_row: usize,
    pub priced: usize,
}

/// A layer's CPU site as the route's policy reads it: none, its lines
/// pending, or its lines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SiteLines<'a> {
    Unsited,
    Pending,
    Measured(&'a CpuLines),
}

/// A streamed layer's lines: the host's and the GPU's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExpertLines {
    pub host: HostLines,
    pub gpu: GpuLines,
}

/// Some of the GPU's hits of a layer: `experts` distinct experts of `rows`
/// rows each (a mean, where it is one).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hits {
    pub experts: f64,
    pub rows: f64,
}

/// One distinct missed expert of a layer: its rows, and when its record is
/// whole in host memory, from the router: at once for a tier record, after
/// its read and landing for a file miss ([`ExpertLines::filed`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Miss {
    pub rows: f64,
    pub ready_seconds: f64,
}

/// Where a miss is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    Cpu,
    Link,
    Copy,
}

/// The pass a layer's misses are served in, which names the sites its graph
/// has: a decode step's route assigns each miss to the CPU or the GPU in
/// place (`assign`); a prompt group's pass moves each cold expert into the
/// GPU's memory behind its wait and computes it there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    Decode,
    Group,
}

/// Where each of a layer's misses was computed (in their order), when the
/// layer ended from its router, what that added to its step, and of the
/// GPU's queue the resident compute of the misses it moved first (their
/// `t_gpu(r)`, which a measured pass's compute already holds).
#[derive(Clone, Debug, PartialEq)]
pub struct ServedMisses {
    pub sites: Vec<Site>,
    pub end_seconds: f64,
    pub exposed: Exposed,
    pub moved_compute_seconds: f64,
}

/// What a layer's misses added to its step, by what held its end past its
/// hits ([`ExpertLines::serve`]): the GPU's queue (misses read over the
/// link in place), the CPU's experts past the GPU's queue, and the
/// handshake the CPU's queue carries past it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Exposed {
    pub gpu_seconds: f64,
    pub cpu_seconds: f64,
    pub handshake_seconds: f64,
}

impl Exposed {
    pub fn total(&self) -> f64 {
        self.gpu_seconds + self.cpu_seconds + self.handshake_seconds
    }
}

impl std::ops::Add for Exposed {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            gpu_seconds: self.gpu_seconds + other.gpu_seconds,
            cpu_seconds: self.cpu_seconds + other.cpu_seconds,
            handshake_seconds: self.handshake_seconds + other.handshake_seconds,
        }
    }
}

#[cfg(test)]
impl ServedMisses {
    /// The misses served at `site`.
    pub fn at(&self, site: Site) -> usize {
        self.sites.iter().filter(|&&at| at == site).count()
    }
}

/// Result of scheduling decode misses onto the CPU or the GPU link
/// ([`ExpertLines::serve_decode`]).
pub(crate) struct DecodeSchedule {
    pub gpu_ns: f64,
    pub cpu_ns: f64,
    pub served_cpu: bool,
    pub link_count: u64,
}

impl ExpertLines {
    /// The lines with `observed`, the landing line the load observes of
    /// its landings' site beside the lines it was priced on, where it has
    /// one.
    pub fn landed_by(self, observed: Option<LandingLine>) -> Self {
        let landing = observed.or(self.host.landing);
        Self { host: HostLines { landing, ..self.host }, ..self }
    }

    /// The lines with the CPU site's line as the load observes its
    /// requests take (`[fixed, per expert]` seconds, an expert of one row,
    /// its rows past the first at the job's row cost, which it keeps), in
    /// place of the line the job measured on an idle machine, where a site
    /// serves: what its requests take beside the machine's other work,
    /// which the route then weighs against the link. A term past a policy
    /// word is priced at a word's most ([`Nanoseconds::saturating`]).
    pub fn sited(self, observed: Option<[f64; 2]>) -> Self {
        let (Some(cpu), Some([fixed, expert])) = (self.host.cpu, observed) else {
            return self;
        };
        let (handshake, one) = (Nanoseconds::saturating(fixed), Nanoseconds::saturating(expert));
        let cpu = CpuLines { handshake, expert: RowLine { one, ..cpu.expert } };
        Self { host: HostLines { cpu: Some(cpu), ..self.host }, ..self }
    }

    /// The lines with the GPU's per-expert costs as the load observes its
    /// sampled steps' blocks take (`[fixed, per hit, per link miss, per
    /// sited miss]` seconds, an expert of one row, its rows past the first
    /// at the job's lines, which it keeps): a hit's in place of the
    /// resident line's weights, a link miss's in place of the link line's,
    /// what each costs the device beside the machine's other work. A term
    /// past a policy word is priced at a word's most
    /// ([`Nanoseconds::saturating`]).
    pub fn passed(self, observed: Option<[f64; 4]>) -> Self {
        let Some([_, hit, link, _]) = observed else {
            return self;
        };
        let refined = |line: PassLine, seconds: f64| PassLine { weights: Nanoseconds::saturating(seconds), ..line };
        Self { gpu: GpuLines { link: refined(self.gpu.link, link), resident: refined(self.gpu.resident, hit), ..self.gpu }, ..self }
    }

    /// The lines with the link's moves as the load measures them
    /// ([`Moves`]), where it measured any: the layer's reads in place then
    /// wait behind them. A term past a policy word is priced at a word's
    /// most ([`Nanoseconds::saturating`]).
    pub fn moving(self, observed: Option<(f64, f64)>) -> Self {
        let Some((link, records)) = observed else {
            return self;
        };
        assert!(records.is_finite() && records >= 0., "a step's moved records: {records}");
        Self { gpu: GpuLines { moves: Moves { link: Nanoseconds::saturating(link), records }, ..self.gpu }, ..self }
    }

    /// What `rows` rows of one expert take past its first along `line`,
    /// in seconds: what a block's time holds beyond the expert of one row
    /// its line's term is.
    pub fn rows_past_first(line: &PassLine, rows: f64) -> f64 {
        (line.nanoseconds(rows) - line.nanoseconds(1.)) * 1e-9
    }

    /// What `count` file misses of the layer requested together add past
    /// their reads' end, at the one site its bank lands them
    /// ([`LandingLine`]; none had yet: nothing to price).
    pub fn landed(&self, count: u64, record_read_seconds: f64) -> f64 {
        match (count, self.host.landing) {
            (0, _) | (_, None) => 0.,
            (count, Some(landing)) => landing.seconds(count, record_read_seconds),
        }
    }

    /// When `count` file misses of the layer requested together are whole
    /// in the tier: their one read along `line` and their landing (along
    /// the landing line where it was measured; their read alone before).
    pub fn filed(&self, line: &ReadLine, count: u64) -> f64 {
        let record_read = line.transfer_seconds(self.host.record_bytes as f64);
        line.seconds(count * self.host.record_bytes) + self.landed(count, record_read)
    }

    /// Effective CPU lines for decode assignment, accounting for shared
    /// `HostRam` bus contention between PCIe DMA (`Link`) and CPU-site (`Cpu`).
    ///
    /// When `host.contended_memory()` is `Some(t_joint)` (discrete GPU sharing
    /// `HostRam` with the CPU):
    /// - `B_joint = b / t_joint` is the aggregate `HostRam` read throughput
    ///   under concurrent `DMA ∥ CPU` traffic (measured, or conservatively
    ///   bounded by `B_mem = b / host.memory` when unprobed).
    /// - `B_link = b / gpu.link.weights` is the uncontended link throughput.
    /// - The spare `HostRam` headroom available to CPU-site alongside saturated
    ///   link traffic is `B_spare = B_joint - B_link`.
    /// - When `B_spare <= 0` (`t_joint >= t_link`), offloading to CPU-site adds
    ///   zero net throughput and degrades wall time, so `Site::Cpu` is
    ///   suppressed (`None`).
    /// - Otherwise, the marginal shared-bus cost of a CPU expert is
    ///   `t_spare = (t_joint * t_link) / (t_link - t_joint)`, and the CPU's
    ///   effective per-expert cost is `max(cpu.expert.one, t_spare)`.
    ///
    /// On unified memory (`host.memory == None`), there is no separate `Link`
    /// sharing `HostRam` with the CPU, so `cpu` is returned unchanged.
    pub fn effective_cpu_for(&self, cpu: CpuLines) -> Option<CpuLines> {
        let Some(t_joint) = self.host.contended_memory() else {
            return Some(cpu);
        };
        let t_joint_ns = t_joint.get();
        let t_link_ns = self.gpu.link.weights.get();
        if t_joint_ns <= 0. || t_link_ns <= 0. {
            return Some(cpu);
        }
        if t_joint_ns >= t_link_ns {
            return None;
        }
        let t_spare_ns = (t_joint_ns * t_link_ns) / (t_link_ns - t_joint_ns);
        let effective_one = Nanoseconds::saturating(cpu.expert.one.get().max(t_spare_ns) * 1e-9);
        Some(CpuLines {
            expert: RowLine { one: effective_one, ..cpu.expert },
            handshake: cpu.handshake,
        })
    }

    /// [`Self::effective_cpu_for`] applied to `self.host.cpu`.
    pub fn effective_cpu(&self) -> Option<CpuLines> {
        self.host.cpu.and_then(|cpu| self.effective_cpu_for(cpu))
    }

    /// Assign each decode miss in `misses` (already ordered most-rows-first) to
    /// the CPU or the GPU in place, starting the GPU queue at `hit_ns` and
    /// notifying `on_site(tag, site)` for each assigned miss.
    ///
    /// Single owner of the decode-miss tie-breaking, handshake, shared-bus
    /// contention and link-busy rules shared by [`Self::serve`],
    /// [`Self::policy`] and [`crate::step::StepInputs`].
    pub(crate) fn serve_decode<T>(
        &self,
        hit_ns: f64,
        misses: impl IntoIterator<Item = (T, Miss)>,
        mut on_site: impl FnMut(T, Site),
    ) -> DecodeSchedule {
        let link_busy = self.gpu.moves.link.get();
        let cpu_lines = self.effective_cpu();
        let (mut gpu_ns, mut cpu_ns) = (hit_ns, 0f64);
        let mut served_cpu = false;
        let mut link_count = 0u64;
        for (tag, miss) in misses {
            let ready_ns = miss.ready_seconds * 1e9;
            let on_cpu = cpu_lines.map(|lines| {
                let handshake = if served_cpu { 0. } else { lines.handshake.get() };
                cpu_ns.max(ready_ns) + lines.expert.nanoseconds(miss.rows) + handshake
            });
            let in_place = gpu_ns.max(ready_ns).max(link_busy) + self.gpu.link.nanoseconds(miss.rows);
            match on_cpu {
                Some(end) if end <= in_place => {
                    cpu_ns = end;
                    served_cpu = true;
                    on_site(tag, Site::Cpu);
                }
                _ => {
                    gpu_ns = in_place;
                    link_count += 1;
                    on_site(tag, Site::Link);
                }
            }
        }
        DecodeSchedule { gpu_ns, cpu_ns, served_cpu, link_count }
    }

    /// The layer's distinct missed experts served in `pass` beside its GPU
    /// `hits` (each expert at its resident line): each, the most rows first
    /// (ties in their order), at the site of the pass it ends soonest on,
    /// the CPU's queue from the router (its first paying the handshake) and
    /// the GPU's from its hits, a read in place no sooner than the link is
    /// free of the step's moves, the CPU winning a tie, then in place; and
    /// what that adds to the step (module doc). Without a CPU site a decode
    /// reads every miss in place; a group moves every miss, its fill the
    /// link's own (a decode step's moves are no group pass's).
    pub fn serve(&self, pass: Pass, hits: &[Hits], misses: &[Miss]) -> ServedMisses {
        let hit = hits.iter().map(|hits| hits.experts * self.gpu.resident.nanoseconds(hits.rows)).sum::<f64>();
        let mut order = (0..misses.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| misses[b].rows.total_cmp(&misses[a].rows));
        if pass == Pass::Group {
            // The fill moves each expert once it is whole in host memory,
            // over the link from a host tier (none without one: unified
            // memory's pass reads the bank's records in place), then the
            // pass computes the hits and the moved experts.
            let move_ns = match self.host.memory {
                Some(_) => self.gpu.link.weights.get(),
                None => 0.,
            };
            let moves = order.iter().fold(0f64, |moves, &index| moves.max(misses[index].ready_seconds * 1e9) + move_ns);
            let moved = misses.iter().map(|miss| self.gpu.resident.nanoseconds(miss.rows)).sum::<f64>();
            let exposed = Exposed { gpu_seconds: (moves + moved) * 1e-9, ..Exposed::default() };
            let end_seconds = (moves + hit + moved) * 1e-9;
            return ServedMisses { sites: vec![Site::Copy; misses.len()], end_seconds, exposed, moved_compute_seconds: moved * 1e-9 };
        }
        let mut sites = vec![Site::Link; misses.len()];
        let DecodeSchedule { gpu_ns: gpu, cpu_ns: cpu, served_cpu: served, .. } =
            self.serve_decode(hit, order.into_iter().map(|index| (index, misses[index])), |index, site| sites[index] = site);
        // The CPU's queue carries the handshake once it served any: of what
        // it ran past the GPU's, its experts' part shown first, the
        // handshake's the rest.
        let handshake = match (served, self.host.cpu) {
            (true, Some(lines)) => lines.handshake.get(),
            _ => 0.,
        };
        let past = (cpu - gpu).max(0.);
        let experts = (cpu - handshake - gpu).max(0.);
        let exposed = Exposed { gpu_seconds: (gpu - hit) * 1e-9, cpu_seconds: experts * 1e-9, handshake_seconds: (past - experts) * 1e-9 };
        ServedMisses { sites, end_seconds: gpu.max(cpu) * 1e-9, exposed, moved_compute_seconds: 0. }
    }

    /// When a prompt group's pass has `misses`, its layer's cold experts,
    /// in the GPU's memory, from their requests: their group pass served
    /// with no hits ([`Self::serve`]), less the resident compute it carries,
    /// which the pass's measured compute holds. No miss's rows enter it.
    pub fn staged(&self, misses: &[Miss]) -> f64 {
        let served = self.serve(Pass::Group, &[], misses);
        served.end_seconds - served.moved_compute_seconds
    }

    /// When a prompt group's pass has `tiered` host-tier misses (ready at 0)
    /// and `filed` file misses (ready after [`Self::filed`]) in the GPU's
    /// memory, from their requests: [`Self::staged`] (`Pass::Group`) without
    /// resident compute. Single owner shared by runtime `Stages` and plan-time
    /// [`crate::BankPrices::prompt_seconds`].
    pub fn group_stage(&self, line: &ReadLine, tiered: u64, filed: u64) -> f64 {
        if tiered == 0 && filed == 0 {
            return 0.;
        }
        let filed_seconds = match filed {
            0 => 0.,
            count => self.filed(line, count),
        };
        let misses: Vec<Miss> = std::iter::repeat_n(0., tiered as usize)
            .chain(std::iter::repeat_n(filed_seconds, filed as usize))
            .map(|ready_seconds| Miss { rows: 1., ready_seconds })
            .collect();
        self.staged(&misses)
    }

    /// The policy words of the layer's lines (`streaming_words.seismic`'s
    /// policy after its home), each `(word, value)` at `layout`'s offsets:
    /// POLICY_PRICED set, and the GPU's words and a site's effective CPU words
    /// ([`Self::effective_cpu_for`]) written, once every line the assignment
    /// reads is measured (`lines`, and a site's own, `site`) and shared-bus
    /// contention leaves spare headroom for CPU-site; else POLICY_PRICED clear
    /// and no line written, so every miss is served on the GPU's path, the CPU
    /// site off.
    pub fn policy(layout: &PolicyLayout, site: SiteLines<'_>, lines: Option<&Self>) -> Vec<(usize, i32)> {
        let mut words = Vec::with_capacity(layout.words);
        let (cpu, lines, priced) = match (site, lines) {
            (SiteLines::Unsited, Some(lines)) => (None, Some(lines), true),
            (SiteLines::Measured(cpu), Some(lines)) => match lines.effective_cpu_for(*cpu) {
                Some(effective) => (Some(effective), Some(lines), true),
                None => (None, None, false),
            },
            (SiteLines::Pending, _) | (_, None) => (None, None, false),
        };
        if let Some(cpu) = cpu {
            words.extend([(layout.cpu, cpu.expert.one), (layout.cpu_row, cpu.expert.row.bound()), (layout.handshake, cpu.handshake)]);
        }
        if let Some(lines) = lines {
            words.extend([
                (layout.link, lines.gpu.link.weights),
                (layout.link_row, lines.gpu.link.row),
                (layout.link_busy, lines.gpu.moves.link),
                (layout.hit, lines.gpu.resident.weights),
                (layout.hit_row, lines.gpu.resident.row),
            ]);
        }
        let mut words = words.into_iter().map(|(word, line)| (word, line.word())).collect::<Vec<_>>();
        words.push((layout.priced, i32::from(priced)));
        words
    }
}

#[cfg(any(test, feature = "fixtures"))]
impl ExpertLines {
    /// The record bytes [`Self::reference`] states its bandwidth terms at.
    pub const REFERENCE_RECORD: u64 = 1_930_000;

    /// The reference RTX 3090 host's lines over records of `record_bytes` bytes,
    /// stored and resident alike, which fixtures price by. Its bandwidth
    /// terms are the machine's rates over the record, as a load derives or
    /// measures them over each group's own (the executor's `host_lines`,
    /// `ExpertPass::lines`), here stated at its 1.93 MB records: the CPU 40
    /// µs an expert of one row from RAM; the link 86 µs in place; host
    /// memory 38 µs a record (51.2 GB/s); the GPU 5 µs a resident expert.
    /// Its other terms hold at any record: the CPU 1 µs a row more and a
    /// 4 µs handshake, the GPU 0.1 µs a row over the link or resident; no
    /// landing line.
    pub fn reference(record_bytes: u64) -> Self {
        let ns = |nanoseconds: u32| Nanoseconds::try_from(nanoseconds).expect("a fixture line");
        // A bandwidth term: `nanoseconds` at the reference record, at its
        // rate over `record_bytes`.
        let rate = |nanoseconds: u64| {
            let record = Self::REFERENCE_RECORD;
            let scaled = (nanoseconds * record_bytes + record / 2) / record;
            ns(u32::try_from(scaled).expect("a fixture line within u32 nanoseconds"))
        };
        Self {
            host: HostLines {
                cpu: Some(CpuLines {
                    expert: RowLine { one: rate(40_000), row: RowCost::Measured(ns(1_000)) },
                    handshake: ns(4_000),
                }),
                memory: Some(rate(38_000)),
                joint_memory: None,
                landing: None,
                record_bytes,
            },
            gpu: GpuLines {
                link: PassLine { weights: rate(86_000), row: ns(100) },
                resident: PassLine { weights: rate(5_000), row: ns(100) },
                moves: Moves::NONE,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines() -> ExpertLines {
        ExpertLines::reference(ExpertLines::REFERENCE_RECORD)
    }

    /// Reference lines without shared `HostRam` bus contention (`memory: None`),
    /// for testing pure two-queue makespan and handshake arithmetic.
    fn uncontended_lines() -> ExpertLines {
        let base = lines();
        ExpertLines { host: HostLines { memory: None, joint_memory: None, ..base.host }, ..base }
    }

    fn misses(count: usize, rows: f64) -> Vec<Miss> {
        vec![Miss { rows, ready_seconds: 0. }; count]
    }

    /// `experts` hits of one row: 5 µs each.
    fn hits(experts: f64) -> [Hits; 1] {
        [Hits { experts, rows: 1. }]
    }

    /// The CPU takes misses while it ends no later than the link: three of
    /// one row end at 4 + 3 · 40 = 124 µs against 45 µs of hits, the link
    /// 131; a fourth would end the CPU at 164, so the link takes it.
    #[test]
    fn the_cpu_takes_misses_while_it_ends_no_later_than_the_link() {
        let lines = uncontended_lines();
        let three = lines.serve(Pass::Decode, &hits(9.), &misses(3, 1.));
        assert_eq!(three.at(Site::Cpu), 3, "{three:?}");
        assert!((three.exposed.total() - (124e-6 - 45e-6)).abs() < 1e-12, "{three:?}");
        // The CPU's queue held the end past the GPU's hits: its experts
        // 75 µs of it, the handshake the other 4.
        assert_eq!(three.exposed.gpu_seconds, 0.);
        assert!((three.exposed.cpu_seconds - 75e-6).abs() < 1e-12, "{three:?}");
        assert!((three.exposed.handshake_seconds - 4e-6).abs() < 1e-12, "{three:?}");
        let four = lines.serve(Pass::Decode, &hits(9.), &misses(4, 1.));
        assert_eq!((four.at(Site::Cpu), four.at(Site::Link)), (3, 1), "{four:?}");
        assert!((four.end_seconds - 131e-6).abs() < 1e-12, "{four:?}");
        // The link's queue held the end: the GPU's part alone.
        assert!((four.exposed.gpu_seconds - (131e-6 - 45e-6)).abs() < 1e-12, "{four:?}");
        assert_eq!((four.exposed.cpu_seconds, four.exposed.handshake_seconds), (0., 0.));
        let none = lines.serve(Pass::Decode, &hits(9.), &[]);
        assert_eq!(none.exposed, Exposed::default());
    }

    /// The link's moves hold it before the layer's reads in place: four
    /// misses beside 45 µs of hits split three to the CPU and one to the
    /// link on an idle link (131 µs), but behind 100 µs of moves a read in
    /// place ends at 186 µs, so the CPU takes the fourth too (164 µs); a
    /// layer that reads nothing in place never waits for the moves.
    #[test]
    fn moves_on_the_link_send_misses_to_the_cpu() {
        let lines = uncontended_lines().moving(Some((100e-6, 0.5)));
        assert_eq!(lines.gpu.moves, Moves { link: Nanoseconds(100_000), records: 0.5 });
        let four = lines.serve(Pass::Decode, &hits(9.), &misses(4, 1.));
        assert_eq!(four.at(Site::Cpu), 4, "{four:?}");
        assert!((four.end_seconds - 164e-6).abs() < 1e-12, "{four:?}");
        let unsited = ExpertLines { host: HostLines { cpu: None, ..lines.host }, ..lines };
        let one = unsited.serve(Pass::Decode, &hits(9.), &misses(1, 1.));
        assert!((one.end_seconds - 186e-6).abs() < 1e-12, "{one:?}");
        assert_eq!(unsited.serve(Pass::Decode, &hits(9.), &[]).exposed, Exposed::default());
        assert_eq!(lines.moving(None), lines);
    }

    /// The handshake is the CPU's first miss's: one miss beside 45 µs of
    /// hits ends at 44 µs, hidden whole.
    #[test]
    fn the_hits_hide_the_cpus_work_and_its_handshake() {
        let lines = uncontended_lines();
        let one = lines.serve(Pass::Decode, &hits(9.), &misses(1, 1.));
        assert_eq!(one.sites, [Site::Cpu]);
        assert_eq!(one.exposed, Exposed::default());
        // Beside 40 µs of hits the handshake shows, the expert hidden.
        let shown = lines.serve(Pass::Decode, &hits(8.), &misses(1, 1.));
        assert_eq!(shown.exposed.cpu_seconds, 0.);
        assert!((shown.exposed.handshake_seconds - 4e-6).abs() < 1e-12, "{shown:?}");
    }

    /// Wide experts (a prompt's) go to the GPU, the CPU's line rising with
    /// their rows; without a CPU site every miss is read in place.
    #[test]
    fn wide_experts_and_layers_without_a_site_read_in_place() {
        let lines = lines();
        let wide = lines.serve(Pass::Decode, &[], &misses(2, 512.));
        assert_eq!(wide.at(Site::Link), 2, "{wide:?}");
        let unsited = ExpertLines { host: HostLines { cpu: None, ..lines.host }, ..lines };
        let in_place = unsited.serve(Pass::Decode, &hits(9.), &misses(3, 1.));
        assert_eq!(in_place.at(Site::Link), 3);
        assert!((in_place.exposed.gpu_seconds - 3. * 86e-6).abs() < 1e-12);
        assert_eq!((in_place.exposed.cpu_seconds, in_place.exposed.handshake_seconds), (0., 0.));
    }

    /// A prompt group's fill moves each cold expert over the link behind its
    /// pass's wait, then the pass computes hits and moved experts resident:
    /// two tier experts of 300 rows beside 9 hits (45 µs) end at 2 · 86 +
    /// 45 + 2 · 30 µs. Their stage is when both are moved, their compute
    /// aside: 2 · 86 µs, and a file expert's after its landing.
    #[test]
    fn a_group_moves_its_cold_experts_and_computes_them_resident() {
        let lines = lines();
        let group = lines.serve(Pass::Group, &hits(9.), &misses(2, 300.));
        assert_eq!(group.at(Site::Copy), 2, "{group:?}");
        assert!((group.end_seconds - (45e-6 + 2. * 116e-6)).abs() < 1e-12, "{group:?}");
        assert!((group.moved_compute_seconds - 60e-6).abs() < 1e-12, "{group:?}");
        assert!((lines.staged(&misses(2, 300.)) - 172e-6).abs() < 1e-12);
        let filed = [Miss { rows: 300., ready_seconds: 1e-3 }];
        assert!((lines.staged(&filed) - (1e-3 + 86e-6)).abs() < 1e-12);
        assert_eq!(lines.staged(&[]), 0.);
    }

    /// A file miss waits for its read and landing before any site.
    #[test]
    fn a_file_miss_waits_for_its_landing() {
        let lines = uncontended_lines();
        let line = ReadLine::new(200e-6, 1. / 6e9).unwrap();
        let ready = lines.filed(&line, 1);
        assert!((ready - (200e-6 + 1_930_000. / 6e9)).abs() < 1e-12);
        let served = lines.serve(Pass::Decode, &hits(9.), &[Miss { rows: 1., ready_seconds: ready }]);
        assert!((served.end_seconds - (ready + 44e-6)).abs() < 1e-9, "{served:?}");
    }

    /// The GPU's per-hit and per-link-miss costs as sampled blocks took
    /// replace the resident and link lines' weights, their row costs kept,
    /// a term past a policy word at a word's most; none observed keeps the
    /// job's.
    #[test]
    fn the_gpu_lines_are_what_sampled_blocks_took() {
        let lines = lines();
        let passed = lines.passed(Some([100e-6, 20e-6, 300e-6, 50e-6]));
        assert_eq!((passed.gpu.resident.weights.get(), passed.gpu.link.weights.get()), (20_000., 300_000.));
        assert_eq!((passed.gpu.resident.row, passed.gpu.link.row), (lines.gpu.resident.row, lines.gpu.link.row));
        assert_eq!(lines.passed(Some([100e-6, 20e-6, 5., 0.])).gpu.link.weights.word(), i32::MAX);
        assert_eq!(lines.passed(None), lines);
    }

    /// Rows past an expert's first along a GPU line are what its arithmetic
    /// takes past its record's read: none while the read is the slower.
    #[test]
    fn rows_past_an_experts_first_are_its_arithmetic_past_its_read() {
        let line = PassLine::of(10e-6, 4e-6).unwrap();
        assert_eq!(ExpertLines::rows_past_first(&line, 1.), 0.);
        assert_eq!(ExpertLines::rows_past_first(&line, 2.), 0.);
        assert!((ExpertLines::rows_past_first(&line, 5.) - 10e-6).abs() < 1e-15);
    }

    /// The CPU site's line as its requests took replaces the job's where a
    /// site serves, its row cost kept, a term past a policy word at a word's
    /// most; a layer without a site, or none observed, keeps its lines.
    #[test]
    fn the_site_line_is_what_its_requests_took() {
        let lines = lines();
        let sited = lines.sited(Some([30e-6, 400e-6]));
        let cpu = sited.host.cpu.unwrap();
        assert_eq!((cpu.handshake.get(), cpu.expert.one.get()), (30_000., 400_000.));
        assert_eq!(cpu.expert.row, lines.host.cpu.unwrap().expert.row);
        let slow = lines.sited(Some([30e-6, 5.])).host.cpu.unwrap();
        assert_eq!(slow.expert.one.word(), i32::MAX);
        assert_eq!(lines.sited(None), lines);
        let unsited = ExpertLines { host: HostLines { cpu: None, ..lines.host }, ..lines };
        assert_eq!(unsited.sited(Some([30e-6, 400e-6])), unsited);
    }

    /// The policy words are the lines' nanoseconds; the GPU's and
    /// POLICY_PRICED only once every line the assignment reads is priced,
    /// the CPU's only of a site, whose pending lines leave the layer
    /// unpriced (every miss on the GPU's path).
    #[test]
    fn the_policy_words_are_the_measured_lines() {
        let lines = uncontended_lines();
        // A layout of its own: each word at an offset no other takes.
        let layout = PolicyLayout { words: 10, cpu: 0, cpu_row: 1, handshake: 2, link: 3, link_row: 4, link_busy: 8, hit: 5, hit_row: 6, priced: 7 };
        let words = |site, priced| ExpertLines::policy(&layout, site, priced).into_iter().collect::<std::collections::BTreeMap<_, _>>();
        let word = |offset: usize| offset;
        let sited = SiteLines::Measured(lines.host.cpu.as_ref().unwrap());
        let both = words(sited, Some(&lines));
        assert_eq!(both[&word(layout.cpu)], 40_000);
        assert_eq!(both[&word(layout.cpu_row)], 1_000);
        assert_eq!(both[&word(layout.handshake)], 4_000);
        assert_eq!(both[&word(layout.link)], 86_000);
        assert_eq!(both[&word(layout.link_busy)], 0, "nothing moves");
        assert_eq!(both[&word(layout.hit_row)], 100);
        assert_eq!(both[&word(layout.priced)], 1);
        // With shared HostRam contention (`memory = 38 µs`, `link = 86 µs`),
        // POLICY_CPU_NS reflects the spare-headroom marginal cost `68_083 ns`.
        let contended = self::lines();
        let contended_words = words(SiteLines::Measured(contended.host.cpu.as_ref().unwrap()), Some(&contended));
        assert_eq!(contended_words[&word(layout.cpu)], 68_083);
        assert_eq!(contended_words[&word(layout.priced)], 1);
        let unpriced = words(sited, None);
        assert_eq!(unpriced[&word(layout.priced)], 0);
        assert!(!unpriced.contains_key(&word(layout.link)));
        assert!(!unpriced.contains_key(&word(layout.cpu)));
        let pending = words(SiteLines::Pending, Some(&lines));
        assert_eq!(pending[&word(layout.priced)], 0);
        assert!(!pending.contains_key(&word(layout.link)) && !pending.contains_key(&word(layout.cpu)));
        let unsited = words(SiteLines::Unsited, Some(&lines));
        assert_eq!(unsited[&word(layout.priced)], 1);
        assert!(!unsited.contains_key(&word(layout.cpu)));
        assert_eq!(Nanoseconds::of(1.6e-9), Ok(Nanoseconds(2)));
        assert_eq!(Nanoseconds::of(10.), Err(LineError(10.)));
    }

    /// Exact samples of the given resolution.
    fn sampled(seconds: impl Fn(usize, usize) -> f64, resolution: f64) -> impl Fn(&[(usize, usize)]) -> Result<Vec<Sample>, String> {
        move |requests: &[(usize, usize)]| Ok(requests.iter().map(|&(experts, rows)| Sample { seconds: seconds(experts, rows), resolution }).collect())
    }

    /// The CPU's requests give its terms: the handshake a request of no
    /// entries, an expert the decode's experts of one row beyond it, a row
    /// one expert's rows beyond its first.
    #[test]
    fn the_cpu_lines_run_through_its_requests() {
        let measure = sampled(|experts, rows| 3e-6 + experts as f64 * (40e-6 + rows.saturating_sub(1) as f64 * 2e-6), 0.);
        let lines = CpuLines::measured(8, 4, &measure).unwrap();
        assert_eq!(lines.handshake.get(), 3_000.);
        assert_eq!((lines.expert.one.get(), lines.expert.row), (40_000., RowCost::Measured(Nanoseconds(2_000))));
        assert_eq!(CpuLines::measured(8, 1, &measure), Err(CpuLinesError::Narrow { experts: 8, rows: 1 }));
        // An expert that costs nothing past the handshake refuses the lines.
        let free = sampled(|_, _| 2e-6, 0.);
        assert!(matches!(CpuLines::measured(8, 4, &free), Err(CpuLinesError::Falling { .. })));
    }

    /// A request's sample is its fastest time, resolved to the next
    /// fastest: noise past them (a slow first request, a preempted one)
    /// moves neither.
    #[test]
    fn a_sample_is_the_fastest_resolved_to_the_next() {
        let noisy = [412e-6, 383e-6, 391e-6, 2.1e-3, 382e-6, 395e-6];
        assert_eq!(Sample::of(&noisy), Some(Sample { seconds: 382e-6, resolution: 383e-6 - 382e-6 }));
        assert_eq!(Sample::of(&[382e-6]), None);
    }

    /// A row slope within its points' resolution is that resolution's
    /// bound, not a refusal: an RTX 3090's Vulkan group (one expert of one
    /// row 382 µs, eight 1418 µs, one of eight rows 337 µs, a handshake of
    /// 8.4 µs), whose eight rows measured faster than one, with each point
    /// resolved to 60 µs, is a row of at most 120/7 µs, priced so. A slope
    /// past the resolution is measured.
    #[test]
    fn a_row_within_its_resolution_is_bounded_by_it() {
        let point = |experts: usize, rows: usize| match (experts, rows) {
            (0, 0) => 8.4e-6,
            (1, 1) => 382e-6,
            (8, 1) => 1418e-6,
            _ => 337e-6,
        };
        let lines = CpuLines::measured(8, 8, &sampled(point, 60e-6)).unwrap();
        assert_eq!(lines.expert.row, RowCost::AtMost(Nanoseconds::of(120e-6 / 7.).unwrap()));
        assert_eq!(lines.expert.nanoseconds(8.), lines.expert.one.get() + 7. * lines.expert.row.bound().get());
        // 25 µs a row over 7 rows, 175 µs, past a 2 µs resolution.
        let sloped = |experts: usize, rows: usize| point(experts, 1).max(1e-6) + rows.saturating_sub(1) as f64 * 25e-6 * (experts.min(1) as f64);
        let lines = CpuLines::measured(8, 8, &sampled(sloped, 2e-6)).unwrap();
        assert_eq!(lines.expert.row, RowCost::Measured(Nanoseconds(25_000)));
    }

    /// Build `ExpertLines` for a 3 MB expert given link bandwidth (`b_link_gbs`),
    /// uncontended HostRam bandwidth (`b_mem_gbs`, `None` on unified memory),
    /// measured concurrent `DMA ∥ CPU` aggregate bandwidth (`b_joint_gbs`), and
    /// uncontended CPU-site time (`cpu_ns`).
    fn topology_lines(
        b_link_gbs: f64,
        b_mem_gbs: Option<f64>,
        b_joint_gbs: Option<f64>,
        cpu_ns: u32,
    ) -> ExpertLines {
        const RECORD: u64 = 3_000_000;
        let ns_of_gbs = |gbs: f64| Nanoseconds::of(RECORD as f64 / (gbs * 1e9)).unwrap();
        let ns = |v: u32| Nanoseconds::try_from(v).unwrap();
        ExpertLines {
            host: HostLines {
                cpu: Some(CpuLines {
                    expert: RowLine { one: ns(cpu_ns), row: RowCost::Measured(ns(1_000)) },
                    handshake: ns(4_000),
                }),
                memory: b_mem_gbs.map(ns_of_gbs),
                joint_memory: b_joint_gbs.map(ns_of_gbs),
                landing: None,
                record_bytes: RECORD,
            },
            gpu: GpuLines {
                link: PassLine { weights: ns_of_gbs(b_link_gbs), row: ns(100) },
                resident: PassLine { weights: ns(5_000), row: ns(100) },
                moves: Moves::NONE,
            },
        }
    }

    /// 1. Gen5 + 2-ch DDR5 (`B_link = 43.6 GB/s`, `B_mem = 47.8 GB/s`,
    ///    contended `B_joint <= 47.8 GB/s`): CPU-site is naturally suppressed
    ///    (0 CPU experts assigned), both under measured contention (`B_joint <= B_link`)
    ///    and under the analytical upper bound (`B_joint = B_mem = 47.8 GB/s`).
    #[test]
    fn gen5_dual_channel_ddr5_suppresses_cpu_site() {
        let layout = PolicyLayout { words: 10, cpu: 0, cpu_row: 1, handshake: 2, link: 3, link_row: 4, link_busy: 8, hit: 5, hit_row: 6, priced: 7 };
        let words = |lines: &ExpertLines| {
            ExpertLines::policy(&layout, SiteLines::Measured(lines.host.cpu.as_ref().unwrap()), Some(lines))
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>()
        };

        // Measured arbitration slowdown: `B_joint = 42.0 GB/s < B_link = 43.6 GB/s`.
        let contended = topology_lines(43.6, Some(47.8), Some(42.0), 75_000);
        assert_eq!(contended.effective_cpu(), None);
        let served = contended.serve(Pass::Decode, &hits(2.), &misses(8, 1.));
        assert_eq!(served.at(Site::Cpu), 0, "{served:?}");
        assert_eq!(served.at(Site::Link), 8, "{served:?}");
        let pol = words(&contended);
        assert_eq!(pol[&layout.priced], 0, "POLICY_PRICED must be cleared when CPU-site has zero headroom");
        assert!(!pol.contains_key(&layout.cpu));

        // Conservative analytical fallback (unprobed `joint_memory: None`, `B_joint = B_mem = 47.8 GB/s`):
        // spare headroom is only `4.2 GB/s` (`t_spare ≈ 714 µs`), so 8 decode misses all go to Link.
        let analytical = topology_lines(43.6, Some(47.8), None, 75_000);
        let served = analytical.serve(Pass::Decode, &hits(2.), &misses(8, 1.));
        assert_eq!(served.at(Site::Cpu), 0, "{served:?}");
        assert_eq!(served.at(Site::Link), 8, "{served:?}");
        let pol = words(&analytical);
        assert_eq!(pol[&layout.priced], 1);
        assert!(pol[&layout.cpu] > 700_000, "marginal CPU cost must reflect 4.2 GB/s spare headroom: {}", pol[&layout.cpu]);
    }

    /// 2. Gen4 + 2-ch DDR5 (`B_link = 20 GB/s`, `B_mem = 47.8 GB/s`,
    ///    contended `B_joint = 44 GB/s`): CPU-site is naturally enabled and
    ///    receives the headroom-bounded share of misses (`B_spare = 24 GB/s` ->
    ///    `t_spare = 125 µs`), strictly less than uncontended `75 µs`.
    #[test]
    fn gen4_dual_channel_ddr5_bounds_cpu_to_spare_headroom() {
        let contended = topology_lines(20.0, Some(47.8), Some(44.0), 75_000);
        let uncontended = topology_lines(20.0, None, None, 75_000);

        let eff = contended.effective_cpu().expect("Gen4 + DDR5 has spare HostRam headroom");
        assert!(
            (eff.expert.one.get() - 125_000.).abs() < 200.,
            "expected ~125 µs for 24 GB/s spare headroom, got {}",
            eff.expert.one.get()
        );

        let served = contended.serve(Pass::Decode, &hits(3.), &misses(8, 1.));
        let served_uncontended = uncontended.serve(Pass::Decode, &hits(3.), &misses(8, 1.));
        assert_eq!(served.at(Site::Cpu), 4, "4 of 8 misses (~50% ≈ 24/44) should go to CPU: {served:?}");
        assert_eq!(served.at(Site::Link), 4, "{served:?}");
        assert!(
            served.at(Site::Cpu) < served_uncontended.at(Site::Cpu),
            "contended CPU share ({}) must be strictly below uncontended ({})",
            served.at(Site::Cpu),
            served_uncontended.at(Site::Cpu),
        );
    }

    /// 3. Gen5 + 8-ch EPYC DDR5 (`B_link = 43.6 GB/s`, `B_mem = 300 GB/s`):
    ///    spare headroom (`~256 GB/s`, `t_spare ≈ 11.7 µs`) far exceeds CPU-site
    ///    rate (`75 µs`), so CPU-site runs at its full uncontended rate in parallel.
    #[test]
    fn gen5_eight_channel_epyc_runs_cpu_uncontended() {
        let epyc = topology_lines(43.6, Some(300.0), Some(295.0), 75_000);
        let uncontended = topology_lines(43.6, None, None, 75_000);

        assert_eq!(epyc.effective_cpu(), epyc.host.cpu, "EPYC spare headroom must not throttle CPU-site");
        let served = epyc.serve(Pass::Decode, &hits(2.), &misses(8, 1.));
        let served_uncontended = uncontended.serve(Pass::Decode, &hits(2.), &misses(8, 1.));
        assert_eq!(served.sites, served_uncontended.sites);
        assert!(served.at(Site::Cpu) >= 3, "CPU should run in parallel with Gen5 DMA on EPYC: {served:?}");
    }

    /// 4. Unified memory (Apple Silicon / GB10): `HostLines::memory == None`
    ///    -> `contended_memory()` is `None` and CPU pricing is completely unaffected.
    #[test]
    fn unified_memory_has_no_shared_link_contention() {
        let unified = topology_lines(43.6, None, Some(40.0), 75_000);
        assert_eq!(unified.host.contended_memory(), None);
        assert_eq!(unified.effective_cpu(), unified.host.cpu);
    }

    /// 5. Backward compatibility: existing cached `HostLines` JSON without
    ///    `joint_memory` deserializes cleanly with `joint_memory: None`.
    #[test]
    fn host_lines_deserializes_without_joint_memory_field() {
        let legacy_json = r#"{"cpu":null,"memory":38000,"landing":null,"record_bytes":1930000}"#;
        let host: HostLines = serde_json::from_str(legacy_json).expect("legacy HostLines JSON must deserialize");
        assert_eq!(host.memory, Some(Nanoseconds(38_000)));
        assert_eq!(host.joint_memory, None);
        assert_eq!(host.contended_memory(), Some(Nanoseconds(38_000)));
    }
}
