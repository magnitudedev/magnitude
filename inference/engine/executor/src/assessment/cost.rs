//! The one cost function of where a missed expert is computed: what a
//! streamed layer's misses add to its step, given the
//! GPU's hits, from lines this machine measured. The route's assignment
//! (through the policy words the host writes between flights,
//! [`ExpertLines::policy`], which the route computes this function from:
//! streaming_words.seismic's assignment), the bank's sizing
//! ([`super::MissRecord::saving`]), assessment's streamed estimate
//! ([`super::MissRecord::step_seconds`]), a prompt group's stages, which its
//! size, merge sets and places are formed by ([`super::streaming`]), and the
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
//! from its hits; the CPU wins a tie, then the link (principle 2). The layer
//! then adds
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

use super::estimate::ExpertPass;
use super::streaming_costs::{FileLine, ReadLine, RecordLine};
use crate::planning::ExpertBankPlan;
use magnitude_kernels::constants;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

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

/// What a load read of its CPU sites from the records (`CpuCosts`): the
/// workers the job sized its pool to, and the lines of each streamed layer
/// that has them stored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StoredCpuSite {
    pub workers: Option<usize>,
    pub lines: std::collections::BTreeMap<usize, CpuLines>,
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
        measure: &dyn Fn(&[(usize, usize)]) -> Result<Vec<Sample>, String>,
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
            if !(expert > 0.) {
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
/// memory), and from the GPU's memory.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuLines {
    pub link: PassLine,
    pub resident: PassLine,
}

/// What the bank holds of a streamed layer's lines on the host's side: its
/// CPU pool's lines, none without a CPU site (unified memory, where the GPU
/// reads host memory at full speed); and one expert's stored bytes, what a
/// file miss reads.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BankLines {
    pub cpu: Option<CpuLines>,
    pub record_bytes: u64,
}

impl BankLines {
    /// Each streamed layer's lines as the bank opens, by its group
    /// (`plan`'s): one expert's stored bytes. The CPU's lines are measured
    /// where their pool is built.
    pub fn of_bank(plan: &ExpertBankPlan) -> Result<Arc<[Self]>, HostLinesError> {
        let groups = plan
            .groups
            .iter()
            .enumerate()
            .map(|(group, members)| {
                let record_bytes = members
                    .record
                    .iter()
                    .map(|(_, tensor)| tensor.stored_bytes())
                    .sum::<Result<u64, _>>()
                    .map_err(|reason| HostLinesError::Record { group, reason })?;
                Ok(Self { cpu: None, record_bytes })
            })
            .collect::<Result<Vec<_>, HostLinesError>>()?;
        Ok(plan.layers.iter().map(|layer| groups[layer.group]).collect())
    }
}

/// A streamed layer's lines on the host's side: the bank's ([`BankLines`]),
/// and where the bank has a host tier, one tier record's read from host
/// memory at the memory's whole read rate (every CPU reading,
/// [`super::MemoryCosts`]), none without one (unified memory's bank has no
/// tier).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HostLines {
    pub cpu: Option<CpuLines>,
    pub memory: Option<Nanoseconds>,
    pub file: Option<FileLine>,
    /// The device's convert of a request's records in their barrier, beside
    /// `file`, the host's: where the host places (unified memory) and the
    /// job measured it.
    pub convert: Option<RecordLine>,
    pub record_bytes: u64,
}

/// Why a bank group's host lines could not be had.
#[derive(Clone, Debug, PartialEq)]
pub enum HostLinesError {
    /// A routed tensor's stored or resident record bytes of group `group`.
    Record { group: usize, reason: String },
    /// Group `group`'s record read from host memory past a policy word.
    Line { group: usize, cause: LineError },
}

impl fmt::Display for HostLinesError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Record { group, reason } => write!(formatter, "bank group {group}'s record: {reason}"),
            Self::Line { group, cause } => write!(formatter, "bank group {group}'s record read from host memory: {cause}"),
        }
    }
}

impl std::error::Error for HostLinesError {}

impl HostLines {
    /// Each streamed layer's host lines: the bank's (`bank`, in decoder
    /// order) beside, where `plan`'s bank has a host tier, one tier record's
    /// read from host memory at `byte_seconds` (`MemoryCosts`), which only
    /// a bank with a tier has.
    pub fn of_bank(plan: &ExpertBankPlan, bank: &[BankLines], byte_seconds: Option<f64>) -> Result<Arc<[Self]>, HostLinesError> {
        assert_eq!(
            byte_seconds.is_some(),
            plan.tier_reservation().is_some(),
            "a host memory line prices exactly a bank with a host tier"
        );
        let memory = plan
            .groups
            .iter()
            .enumerate()
            .map(|(group, members)| {
                byte_seconds
                    .map(|byte_seconds| {
                        let bytes = members
                            .record
                            .iter()
                            .map(|(_, tensor)| tensor.record_bytes())
                            .sum::<Result<u64, _>>()
                            .map_err(|reason| HostLinesError::Record { group, reason })?;
                        Nanoseconds::of(bytes as f64 * byte_seconds).map_err(|cause| HostLinesError::Line { group, cause })
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>, HostLinesError>>()?;
        Ok(plan
            .layers
            .iter()
            .zip(bank)
            .map(|(layer, bank)| Self { cpu: bank.cpu, memory: memory[layer.group], file: None, convert: None, record_bytes: bank.record_bytes })
            .collect())
    }
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

impl ServedMisses {
    /// The misses served at `site`.
    pub fn at(&self, site: Site) -> usize {
        self.sites.iter().filter(|&&at| at == site).count()
    }
}

/// Why the lines of a load's layers could not be priced.
#[derive(Clone, Debug, PartialEq)]
pub enum PricingError {
    /// The executor's costs have not read host storage's routes, whose rate
    /// the link line is: nothing else stands in for it.
    UnreadLink,
    /// A GPU line past a policy word.
    Line(LineError),
}

impl fmt::Display for PricingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnreadLink => formatter.write_str("the executor's costs have not read host storage, the link line's"),
            Self::Line(error) => write!(formatter, "a GPU line: {error}"),
        }
    }
}

impl std::error::Error for PricingError {}

impl ExpertLines {
    /// Each streamed layer's lines: the host's the load measured (`host`,
    /// in decoder order) and the GPU's its kernels' costs price (`pass`),
    /// once they read every line.
    pub fn priced(host: &[HostLines], pass: &ExpertPass) -> Result<Arc<[Self]>, PricingError> {
        host.iter()
            .enumerate()
            .map(|(layer, &host)| Ok(Self { host, gpu: pass.lines(layer).ok_or(PricingError::UnreadLink)?.map_err(PricingError::Line)? }))
            .collect()
    }

    /// The lines with `file`, the landing line the load observes beside
    /// the lines it was priced on, and `convert`, the device's beside it.
    pub fn landing(self, file: Option<FileLine>, convert: Option<RecordLine>) -> Self {
        Self { host: HostLines { file, convert, ..self.host }, ..self }
    }

    /// What `count` file misses of the layer add past their reads' end, at
    /// the site that lands them sooner: the host's, or the device's convert
    /// in their barrier (none measured yet: nothing to price).
    pub fn landed(&self, count: u64) -> f64 {
        self.host.file.map_or(0., |file| match self.host.convert {
            Some(device) if file.device_lands(&device, count) => device.seconds(count),
            _ => file.seconds(count),
        })
    }

    /// When `count` file misses of the layer requested together are whole
    /// in the tier: their one read along `line` and their landing (along
    /// the landing line where it was measured; their read alone before).
    pub fn filed(&self, line: &ReadLine, count: u64) -> f64 {
        line.seconds(count * self.host.record_bytes) + self.landed(count)
    }

    /// The layer's distinct missed experts served in `pass` beside its GPU
    /// `hits` (each expert at its resident line): each, the most rows first
    /// (ties in their order), at the site of the pass it ends soonest on,
    /// the CPU's queue from the router (its first paying the handshake) and
    /// the GPU's from its hits, the CPU winning a tie, then in place; and
    /// what that adds to the step (module doc). Without a CPU site a decode
    /// reads every miss in place; a group moves every miss.
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
        let (mut gpu, mut cpu) = (hit, 0f64);
        let mut sites = vec![Site::Link; misses.len()];
        let mut served = false;
        for index in order {
            let miss = misses[index];
            let ready = miss.ready_seconds * 1e9;
            let on_cpu = self.host.cpu.map(|lines| {
                let handshake = if served { 0. } else { lines.handshake.get() };
                cpu.max(ready) + lines.expert.nanoseconds(miss.rows) + handshake
            });
            let in_place = gpu.max(ready) + self.gpu.link.nanoseconds(miss.rows);
            match on_cpu {
                Some(end) if end <= in_place => {
                    (cpu, served) = (end, true);
                    sites[index] = Site::Cpu;
                }
                _ => gpu = in_place,
            }
        }
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

    /// The policy words of the layer's lines (`streaming_words.seismic`'s
    /// policy after its home), each `(word, value)`: POLICY_PRICED set, and
    /// the GPU's words and a site's CPU words written, once every line the
    /// assignment reads is measured (`lines`, and a site's own, `site`);
    /// else POLICY_PRICED clear and no line written, so every miss is
    /// served on the GPU's path, the CPU site off.
    pub fn policy(site: SiteLines<'_>, lines: Option<&Self>) -> Vec<(usize, i32)> {
        let mut words = Vec::with_capacity(constants::POLICY_WORDS as usize);
        let (cpu, lines) = match (site, lines) {
            (SiteLines::Unsited, Some(lines)) => (None, Some(lines)),
            (SiteLines::Measured(cpu), Some(lines)) => (Some(cpu), Some(lines)),
            (SiteLines::Pending, _) | (_, None) => (None, None),
        };
        if let Some(cpu) = cpu {
            words.extend([
                (constants::POLICY_CPU_NS, cpu.expert.one),
                (constants::POLICY_CPU_ROW_NS, cpu.expert.row.bound()),
                (constants::POLICY_HANDSHAKE_NS, cpu.handshake),
            ]);
        }
        if let Some(lines) = lines {
            words.extend([
                (constants::POLICY_LINK_NS, lines.gpu.link.weights),
                (constants::POLICY_LINK_ROW_NS, lines.gpu.link.row),
                (constants::POLICY_HIT_NS, lines.gpu.resident.weights),
                (constants::POLICY_HIT_ROW_NS, lines.gpu.resident.row),
            ]);
        }
        let mut words = words.into_iter().map(|(word, line)| (word as usize, line.word())).collect::<Vec<_>>();
        words.push((constants::POLICY_PRICED as usize, i32::from(lines.is_some())));
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
    /// measures them over each group's own ([`HostLines::of_bank`],
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
                file: None,
                convert: None,
                record_bytes,
            },
            gpu: GpuLines {
                link: PassLine { weights: rate(86_000), row: ns(100) },
                resident: PassLine { weights: rate(5_000), row: ns(100) },
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
        let lines = lines();
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

    /// The handshake is the CPU's first miss's: one miss beside 45 µs of
    /// hits ends at 44 µs, hidden whole.
    #[test]
    fn the_hits_hide_the_cpus_work_and_its_handshake() {
        let lines = lines();
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
        let lines = lines();
        let line = ReadLine::new(200e-6, 1. / 6e9).unwrap();
        let ready = lines.filed(&line, 1);
        assert!((ready - (200e-6 + 1_930_000. / 6e9)).abs() < 1e-12);
        let served = lines.serve(Pass::Decode, &hits(9.), &[Miss { rows: 1., ready_seconds: ready }]);
        assert!((served.end_seconds - (ready + 44e-6)).abs() < 1e-9, "{served:?}");
    }

    /// The policy words are the lines' nanoseconds; the GPU's and
    /// POLICY_PRICED only once every line the assignment reads is priced,
    /// the CPU's only of a site, whose pending lines leave the layer
    /// unpriced (every miss on the GPU's path).
    #[test]
    fn the_policy_words_are_the_measured_lines() {
        let lines = lines();
        let words = |site, priced| ExpertLines::policy(site, priced).into_iter().collect::<std::collections::BTreeMap<_, _>>();
        let word = |name: u64| name as usize;
        let sited = SiteLines::Measured(lines.host.cpu.as_ref().unwrap());
        let both = words(sited, Some(&lines));
        assert_eq!(both[&word(constants::POLICY_CPU_NS)], 40_000);
        assert_eq!(both[&word(constants::POLICY_CPU_ROW_NS)], 1_000);
        assert_eq!(both[&word(constants::POLICY_HANDSHAKE_NS)], 4_000);
        assert_eq!(both[&word(constants::POLICY_LINK_NS)], 86_000);
        assert_eq!(both[&word(constants::POLICY_HIT_ROW_NS)], 100);
        assert_eq!(both[&word(constants::POLICY_PRICED)], 1);
        let unpriced = words(sited, None);
        assert_eq!(unpriced[&word(constants::POLICY_PRICED)], 0);
        assert!(!unpriced.contains_key(&word(constants::POLICY_LINK_NS)));
        assert!(!unpriced.contains_key(&word(constants::POLICY_CPU_NS)));
        let pending = words(SiteLines::Pending, Some(&lines));
        assert_eq!(pending[&word(constants::POLICY_PRICED)], 0);
        assert!(!pending.contains_key(&word(constants::POLICY_LINK_NS)) && !pending.contains_key(&word(constants::POLICY_CPU_NS)));
        let unsited = words(SiteLines::Unsited, Some(&lines));
        assert_eq!(unsited[&word(constants::POLICY_PRICED)], 1);
        assert!(!unsited.contains_key(&word(constants::POLICY_CPU_NS)));
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

    mod route {
        //! The route's assignment (`assign`, routed.seismic's reference
        //! body, which every native is admitted against) computes
        //! [`ExpertLines::serve`]: on the inputs the route has (one step's
        //! choices, each distinct expert's rows its choices, every record
        //! whole at the router, the policy words of the lines), the CPU's
        //! experts are exactly the function's.
        use super::super::*;
        use seismic_lang::checked::{CheckedModule, SourceFile, check_source};
        use seismic_lang::entry::ElementBindings;
        use seismic_lang::failure::SourceTermination;
        use seismic_lang::interp::{Arg, Interpreter, TensorData};
        use seismic_lang::types::DType;

        fn module() -> CheckedModule {
            let mut sources = seismic_std::sources();
            for (path, text) in [
                ("routed.seismic", include_str!("../../../kernels/kernels/routed.seismic")),
                ("streaming_words.seismic", include_str!("../../../kernels/kernels/streaming_words.seismic")),
                ("functions.seismic", include_str!("../../../kernels/kernels/functions.seismic")),
            ] {
                sources.push(SourceFile { path: path.into(), text: text.into() });
            }
            check_source(sources).unwrap()
        }

        /// The sources `assign` leaves of `routes` (`M` rows of `K`)
        /// resolved to `records` under `policy`.
        fn assign(module: &CheckedModule, k: usize, routes: &[i32], records: &[i32], policy: &[i32]) -> Vec<i32> {
            let m = routes.len() / k;
            let entry = module.entry(module.entry_named("assign").unwrap(), &ElementBindings::new()).unwrap();
            let mut interpreter = Interpreter::new(&entry);
            let mut tensor = |shape: Vec<usize>, values: &[i32]| {
                Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::I32, shape, values.iter().map(|&v| f64::from(v)).collect())))
            };
            let sources = magnitude_kernels::source_words(records, &vec![-1; records.len()]);
            let arguments = [
                tensor(vec![m, k], routes),
                tensor(vec![constants::POLICY_WORDS as usize], policy),
                tensor(vec![constants::SOURCE_ROWS as usize, m, k], &sources),
            ];
            let outcome = interpreter.run(&arguments).unwrap();
            assert!(!matches!(outcome.termination(), SourceTermination::Failed(_)), "assign failed");
            let input = outcome.inputs().find(|input| input.ordinal() == 2).unwrap();
            let sources = input.tensor();
            (0..sources.element_count()).map(|at| sources.read(at).unwrap() as i32).collect()
        }

        /// A small deterministic generator (xorshift).
        struct Draw(u64);

        impl Draw {
            fn below(&mut self, bound: u64) -> u64 {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                self.0 % bound
            }
        }

        #[test]
        fn the_routes_assignment_is_the_cost_functions() {
            let module = module();
            let (k, experts) = (4usize, 12i32);
            let mut draw = Draw(0x9e37_79b9_7f4a_7c15);
            for case in 0..200 {
                let m = 1 + draw.below(4) as usize;
                // Each row's K distinct experts.
                let mut routes = Vec::with_capacity(m * k);
                for _ in 0..m {
                    let mut row = Vec::with_capacity(k);
                    while row.len() < k {
                        let expert = draw.below(experts as u64) as i32;
                        if !row.contains(&expert) {
                            row.push(expert);
                        }
                    }
                    routes.extend(row);
                }
                // Each expert a hit, a tier miss or absent.
                let state = (0..experts).map(|_| draw.below(3)).collect::<Vec<_>>();
                let records = routes
                    .iter()
                    .map(|&expert| match state[expert as usize] {
                        0 => 100 + expert,
                        1 => -2 - (50 + expert),
                        _ => -1,
                    })
                    .collect::<Vec<_>>();
                let ns = |low: u64, high: u64| Nanoseconds::try_from((low + draw.below(high - low)) as u32).unwrap();
                let mut ns = ns;
                let lines = ExpertLines {
                    host: HostLines {
                        cpu: Some(CpuLines {
                            expert: RowLine { one: ns(20_000, 90_000), row: RowCost::Measured(ns(0, 20_000)) },
                            handshake: ns(0, 10_000),
                        }),
                        memory: None,
                        file: None,
                        convert: None,
                        record_bytes: 1,
                    },
                    gpu: GpuLines {
                        link: PassLine { weights: ns(20_000, 150_000), row: ns(0, 60_000) },
                        resident: PassLine { weights: ns(1_000, 10_000), row: ns(0, 5_000) },
                    },
                };
                let mut policy = vec![0; constants::POLICY_WORDS as usize];
                for (word, value) in ExpertLines::policy(SiteLines::Measured(lines.host.cpu.as_ref().unwrap()), Some(&lines)) {
                    policy[word] = value;
                }
                let sources = assign(&module, k, &routes, &records, &policy);
                // The host function over the same step: each distinct
                // expert in first-choice order, its rows its choices.
                let mut distinct = Vec::<i32>::new();
                for &expert in &routes {
                    if !distinct.contains(&expert) {
                        distinct.push(expert);
                    }
                }
                let rows = |expert: i32| routes.iter().filter(|&&chosen| chosen == expert).count() as f64;
                let hit = distinct.iter().filter(|&&expert| state[expert as usize] == 0).map(|&expert| Hits { experts: 1., rows: rows(expert) }).collect::<Vec<_>>();
                let missed = distinct.iter().copied().filter(|&expert| state[expert as usize] != 0).collect::<Vec<_>>();
                let misses = missed.iter().map(|&expert| Miss { rows: rows(expert), ready_seconds: 0. }).collect::<Vec<_>>();
                let served = lines.serve(Pass::Decode, &hit, &misses);
                let count = routes.len();
                for (choice, &expert) in routes.iter().enumerate() {
                    let (record, site) = (sources[constants::SOURCE_RECORD as usize * count + choice], sources[constants::SOURCE_SITE as usize * count + choice]);
                    let expected = match missed.iter().position(|&miss| miss == expert).map(|at| served.sites[at]) {
                        Some(Site::Cpu) => (-1, match state[expert as usize] { 1 => 50 + expert, _ => -2 }),
                        _ => (records[choice], -1),
                    };
                    assert_eq!((record, site), expected, "case {case}: choice {choice} (expert {expert}) of {routes:?}, {served:?}");
                }
            }
        }
    }
}
