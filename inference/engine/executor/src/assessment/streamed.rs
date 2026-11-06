//! The streamed decode estimate: what a streamed plan's decode step takes on
//! this machine, from the plan's structure and the machine's stored figures
//! alone. No model's own measurement enters it, so a model never loaded is
//! estimated as one loaded many times.
//!
//! - **Misses** ([`MissModel`]): the one generic model of how a bank of a
//!   share `s` of a layer's experts misses its routed experts, the share
//!   `m(s)` of a step's distinct routed experts it lacks, of that layer's `E`
//!   and `k` alone. A layer reads `D(r)` distinct experts a step of `r` rows,
//!   so a bank of `slots` misses `m(slots/E)·D(r)` of them, spread over steps
//!   as evenly as they fall; of those misses, the ones a host tier also lacks
//!   (`m(host/E)·D(r)`, the tier inclusive of the bank) are read from the file
//!   and the rest from the tier. A layer's misses of a step are one request at
//!   its route, a share `q = 1 − (1 − m)^k` of a row's steps requesting.
//! - **Lines**: the GPU's over each layer's expert from the executor's
//!   costs ([`ExpertPass::lines`]), one expert's stored bytes from the plan,
//!   a host tier's record read at the host memory's line, which also binds
//!   a CPU site's misses where the plan has one, the file's reads at the
//!   model store's read line and, with no host tier, a read's landing at the
//!   device's convert, all measured by the measurement job
//!   ([`StreamedFigures`]).
//! - **The step** is the one step model's ([`super::step`]) on `C`, the
//!   resident plan's step, with no read ahead: a hint saves only what it
//!   lands before its layer, which no stored figure prices.
//!
//! A load prices its experts' decisions on the same step ([`DecodePrice`]):
//! the same generic model counted as one more step of what its own decode
//! read this session ([`LiveDecode`]), at its decode's rows and context.
//! Nothing it counts is stored.
//!
use super::cost::{BankLines, CpuLines, ExpertLines, HostLines, HostLinesError, LineError, Nanoseconds, RowCost, RowLine};
use super::estimate::ExpertPass;
use super::executor_costs::ExecutorCosts;
use super::miss_model::MissModel;
use super::misses::{HeldSlots, StreamedStep};
use super::priced::Priced;
use super::step::{LayerLoad, ReadAhead, StepInputs};
use super::streaming_costs::{CeilingError, ConvertCosts, ConvertForm, FileLine, MemoryCosts, MemoryError, ReadLine, RecordLine, StorageCosts, StorageError};
use crate::planning::ExpertBankPlan;
use seismic::measure::{InvalidLine, Record, RecordStore, StoreError};
use seismic::{DeviceTopology, Executor, PointKey};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

/// What a streamed estimate reads beside the executor's costs, each a
/// standing request of the measurement job (`requests::standing`): the
/// model store's read line, where a bank can have a host tier, the host
/// memory's seconds a byte with every CPU reading, and a stored byte's
/// seconds through the device's convert (`ConvertForm::reference`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamedFigures {
    pub read: ReadLine,
    pub memory: Option<f64>,
    pub convert: f64,
}

impl StreamedFigures {
    /// The figures `store`'s records give `executor` of the model store at
    /// `storage`, measuring nothing, with the records they were read from:
    /// pending while the job has not measured them, unavailable where they
    /// cannot be had.
    pub fn stored(
        topology: &DeviceTopology,
        executor: &Executor,
        storage: &Path,
        store: &dyn RecordStore<PointKey>,
    ) -> (Priced<Self>, Vec<(PointKey, Record)>) {
        let reading = Reading { store, read: RefCell::default() };
        let read = match StorageCosts::stored(topology, executor, storage, &reading) {
            Ok(line) => Priced::Measured(line),
            Err(error @ StorageError::Unrecorded { .. }) => Priced::Pending(error.to_string()),
            Err(error) => Priced::Unavailable(error.to_string()),
        };
        let memory = match topology.shared_ranges_lie_in_device_memory(executor) {
            true => Priced::Measured(None),
            false => match std::thread::available_parallelism() {
                Err(error) => Priced::Unavailable(format!("the host's processors: {error}")),
                Ok(readers) => match MemoryCosts::stored(topology, executor, readers.get() as u64, &reading) {
                    Ok(costs) => Priced::Measured(Some(costs.byte_seconds)),
                    Err(error @ MemoryError::Unrecorded { .. }) => Priced::Pending(error.to_string()),
                    Err(error) => Priced::Unavailable(error.to_string()),
                },
            },
        };
        let convert = match ConvertForm::reference(executor.backend()) {
            Err(error) => Priced::Unavailable(error.to_string()),
            Ok(None) => Priced::Unavailable(format!("{:?} streams nothing", executor.backend())),
            Ok(Some(form)) => match ConvertCosts::stored(topology, executor, form, &reading) {
                Ok(costs) => form.stored_byte_seconds(costs.record_seconds).map_or_else(|error| Priced::Unavailable(error.to_string()), Priced::Measured),
                Err(error @ CeilingError::Unrecorded { .. }) => Priced::Pending(error.to_string()),
                Err(error) => Priced::Unavailable(error.to_string()),
            },
        };
        let figures = read.and_then(|read| {
            memory.and_then(|memory| convert.and_then(|convert| Priced::Measured(Self { read, memory, convert })))
        });
        (figures, reading.read.into_inner())
    }
}

/// A store whose loads are kept: what a figure was read from.
struct Reading<'a> {
    store: &'a dyn RecordStore<PointKey>,
    read: RefCell<Vec<(PointKey, Record)>>,
}

impl RecordStore<PointKey> for Reading<'_> {
    fn load(&self, key: &PointKey) -> Result<Option<Record>, StoreError> {
        let record = self.store.load(key)?;
        if let Some(record) = &record {
            self.read.borrow_mut().push((key.clone(), record.clone()));
        }
        Ok(record)
    }

    fn store(&self, _: &PointKey, _: &Record) -> Result<(), StoreError> {
        unreachable!("a figure's reading stores nothing")
    }
}

/// Why a streamed plan's layers cannot be priced.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamedError {
    /// The executor's costs give no line over host storage's route.
    UnreadLink,
    Line(LineError),
    Host(HostLinesError),
    Costs(String),
    /// A landing's line from the convert's figure.
    Landing(InvalidLine),
}

impl fmt::Display for StreamedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnreadLink => write!(formatter, "the executor's costs read no host storage route"),
            Self::Line(error) => error.fmt(formatter),
            Self::Host(error) => error.fmt(formatter),
            Self::Costs(error) => write!(formatter, "the streamed layers' passes: {error}"),
            Self::Landing(error) => write!(formatter, "a landing's line: {error}"),
        }
    }
}

impl std::error::Error for StreamedError {}

/// What a bank of some size misses at a layer: the share of a step's routed
/// experts it lacks, and the share of a single row's steps that miss at all.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shares {
    misses: f64,
    requests: f64,
}

/// A layer's decode reads as a load counted them, each entry the count of
/// those at or past that depth in the ranking's stack: what a bank of that
/// many slots misses. The last entry counts first reads, which miss a bank of
/// any size.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerReads {
    misses: Vec<u64>,
    requests: Vec<u64>,
}

impl LayerReads {
    /// The reads of `misses` and steps of `requests` by depth, one bucket a
    /// depth and the last the first reads.
    pub fn of_depths(misses: &[u64], requests: &[u64]) -> Self {
        assert_eq!(misses.len(), requests.len(), "a layer's reads and steps are by one depth");
        let tails = |counts: &[u64]| {
            let mut tails = counts.to_vec();
            for depth in (0..tails.len().saturating_sub(1)).rev() {
                tails[depth] += tails[depth + 1];
            }
            tails
        };
        Self { misses: tails(misses), requests: tails(requests) }
    }

    /// The fewest slots past which a bigger bank catches no further read: the
    /// depth past the deepest read found in the stack; none of a layer that
    /// has found no read there yet.
    fn saturation(&self) -> Option<u64> {
        let first = *self.misses.last().expect("a layer's reads have a first-read bucket");
        let found = self.misses.iter().position(|&tail| tail == first).expect("the last bucket is its own tail");
        (found > 0).then_some(found as u64)
    }
}

/// What a load's counted decode read and ran this session
/// ([`crate::experts::directory::Stack::live`]): its layers' reads, its
/// steps, and their mean rows and context tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct LiveDecode {
    pub layers: Vec<LayerReads>,
    pub steps: u64,
    pub rows: f64,
    pub depth: u32,
}

impl LiveDecode {
    /// `prior` counted as one more step of what `layer`'s decode read at a
    /// bank of `slots`, so a load starts on the generic model and its own
    /// reads take over as it decodes. A step of several rows missing at all
    /// with a row's share `q` misses with `1 − (1 − q)^rows`, so its requests
    /// count at `q`.
    fn shares(&self, layer: usize, slots: usize, prior: Shares) -> Shares {
        let reads = &self.layers[layer];
        let at = slots.min(reads.misses.len() - 1);
        let steps = self.steps as f64;
        let over = |prior: f64, live: f64| (prior + steps * live) / (steps + 1.);
        let misses = match reads.misses[0] {
            0 => prior.misses,
            all => over(prior.misses, reads.misses[at] as f64 / all as f64),
        };
        let missing = (reads.requests[at] as f64 / steps).min(1.);
        Shares { misses, requests: over(prior.requests, 1. - (1. - missing).powf(self.rows.recip())) }
    }

    /// The fewest slots past which its reads stop falling
    /// ([`LayerReads::saturation`]): the deepest layer's; none before any
    /// layer found a read in its stack.
    fn saturation(&self) -> Option<u64> {
        self.layers.iter().filter_map(LayerReads::saturation).max()
    }
}

/// A streamed plan's layers as the estimate prices them: each layer's `E`,
/// the plan's `k`, and each layer's lines.
pub struct StreamedLayers {
    experts: Vec<u64>,
    choices: u64,
    /// Each layer's generic model.
    models: Vec<std::sync::Arc<MissModel>>,
    lines: std::sync::Arc<[ExpertLines]>,
}

impl StreamedLayers {
    /// `bank`'s layers on the executor `costs` price, a host tier's record
    /// read at `figures`' memory line.
    pub fn of_bank(bank: &ExpertBankPlan, costs: &ExecutorCosts, figures: &StreamedFigures) -> Result<Self, StreamedError> {
        let pass = ExpertPass::of_bank(bank, costs.clone()).map_err(|error| StreamedError::Costs(error.to_string()))?;
        let byte_seconds = bank.tier_reservation().and(figures.memory);
        let host = HostLines::of_bank(bank, &BankLines::of_bank(bank).map_err(StreamedError::Host)?, byte_seconds).map_err(StreamedError::Host)?;
        let lines = host
            .iter()
            .enumerate()
            .map(|(layer, &host)| {
                let gpu = pass.lines(layer).ok_or(StreamedError::UnreadLink)?.map_err(StreamedError::Line)?;
                // A CPU site's misses are bound by the host memory they read: its record at the
                // memory's whole rate, no form's line being the machine's.
                let cpu = host.memory.filter(|_| bank.sited(layer)).map(|one| CpuLines { handshake: Nanoseconds::saturating(0.), expert: RowLine { one, row: RowCost::AtMost(gpu.link.row) } });
                let file = landing(&host, figures.convert).map_err(StreamedError::Landing)?;
                Ok(ExpertLines { host: HostLines { cpu, file, ..host }, gpu })
            })
            .collect::<Result<Vec<_>, StreamedError>>()?;
        Ok(Self::new(bank.layers().iter().map(|layer| layer.experts).collect(), bank.selected(), lines.into()))
    }

    /// Layers of `experts` experts each, routing `choices` a row, each
    /// priced by its `lines` (a load's own, `ExpertLines::priced`), in
    /// decoder order.
    pub fn new(experts: Vec<u64>, choices: u64, lines: std::sync::Arc<[ExpertLines]>) -> Self {
        assert_eq!(lines.len(), experts.len(), "a streamed layer's lines are its own");
        let models = experts.iter().map(|&experts| MissModel::of(experts, choices)).collect();
        Self { experts, choices, models, lines }
    }

    /// `T` of a decode step of `rows` rows holding `held` of each layer's
    /// experts, on a resident step of `compute_seconds` (`C`), its file
    /// reads along `read`, its misses the generic model's ([`MissModel`]),
    /// a load's beside its own decode (`live`): each layer's misses and
    /// requests as whole counts over a period of the widest layer's `E`
    /// steps, each spread as evenly as integers allow, priced by the one step
    /// model, each distinct step of the period once.
    pub fn step_seconds(&self, live: Option<&LiveDecode>, read: ReadLine, held: HeldSlots, rows: f64, compute_seconds: f64) -> StreamedStep {
        let period = self.experts.iter().copied().max().expect("a streamed plan streams a layer") as usize;
        let layers = self
            .experts
            .iter()
            .enumerate()
            .map(|(layer, &experts)| {
                let model = &self.models[layer];
                let shares = |slots: usize| {
                    let prior = generic_shares(model, slots as f64 / experts as f64);
                    live.map_or(prior, |live| live.shares(layer, slots, prior))
                };
                layer_period(&shares, model.routed(rows), self.choices, held, rows, period)
            })
            .collect::<Vec<_>>();
        // A layer's loads differ from step to step only in their misses, so
        // a step is its layers' misses; each distinct one keeps its first
        // step and its count.
        let mut distinct = BTreeMap::<Vec<(u64, u64)>, (usize, usize)>::new();
        for step in 0..period {
            let misses = layers.iter().map(|loads| (loads[step].tier, loads[step].file)).collect();
            distinct.entry(misses).or_insert((step, 0)).1 += 1;
        }
        let steps = distinct
            .into_values()
            .map(|(step, count)| (layers.iter().map(|loads| loads[step]).collect::<Vec<_>>(), count))
            .collect::<Vec<_>>();
        let inputs = StepInputs {
            compute: compute_seconds,
            head: 0.,
            overlap: 1.,
            read,
            sustained: read.seconds_per_byte(),
            ahead: ReadAhead::NONE,
            lines: &self.lines,
        };
        let step = inputs.unhinted(steps.iter().map(|(loads, count)| (loads.as_slice(), *count)));
        StreamedStep { compute_seconds, exposed_seconds: step.total - compute_seconds, total_seconds: step.total }
    }
}

/// What a file miss of a layer of `host`'s lines takes past its read: with no
/// host tier it lands through the device's convert into its bank record before
/// its layer computes, its stored bytes at `byte_seconds`; beside a tier the
/// CPU or the link serves it where it landed.
fn landing(host: &HostLines, byte_seconds: f64) -> Result<Option<FileLine>, InvalidLine> {
    match host.memory {
        Some(_) => Ok(None),
        None => RecordLine::new(0., host.record_bytes as f64 * byte_seconds).map(|landing| Some(FileLine { landing })),
    }
}

/// A load's decode step as its experts' decisions price it at any holding:
/// its layers', on the generic model beside what its counted decode read
/// ([`LiveDecode`]), at that decode's rows, on `C`, the resident step of those
/// rows at its context, its file read along the model store's read line: the
/// one step assessment prices ([`StreamedLayers::step_seconds`]).
pub struct DecodePrice {
    layers: StreamedLayers,
    live: LiveDecode,
    read: ReadLine,
    compute_seconds: f64,
}

impl DecodePrice {
    /// `live`'s decode on `layers`, its resident step `compute_seconds`
    /// (`C` at `live`'s rows and context).
    pub fn new(layers: StreamedLayers, live: LiveDecode, read: ReadLine, compute_seconds: f64) -> Self {
        assert_eq!(layers.experts.len(), live.layers.len(), "a decode's reads are of its streamed layers");
        Self { layers, live, read, compute_seconds }
    }

    /// The step holding `held`.
    pub fn step(&self, held: HeldSlots) -> StreamedStep {
        self.layers.step_seconds(Some(&self.live), self.read, held, self.live.rows, self.compute_seconds)
    }

    /// What holding `to` rather than `from` saves a step, in seconds.
    pub fn saving(&self, from: HeldSlots, to: HeldSlots) -> f64 {
        self.step(from).total_seconds - self.step(to).total_seconds
    }

    /// The decode steps counted: what a saving a step is over.
    pub fn steps(&self) -> u64 {
        self.live.steps
    }

    /// The model store's read line its file reads take.
    pub fn read(&self) -> &ReadLine {
        &self.read
    }

    /// The fewest slots at which the decode's own reads stop falling; none
    /// before it found a read in its stack.
    pub fn saturation(&self) -> Option<u64> {
        self.live.saturation()
    }
}

/// What `model` gives a bank of a share `share` of its layer's experts, a
/// bank past them holding them all.
fn generic_shares(model: &MissModel, share: f64) -> Shares {
    let misses = model.misses(share.min(1.));
    Shares { misses, requests: model.requests(misses) }
}

/// A layer routing `routed` distinct experts a step and `choices` a row, its
/// loads over a period of `period` steps of `rows` rows holding `held`: its misses and
/// requests as `shares` give them at a bank of some slots, as whole counts
/// over the period, the requests spread as evenly over it as integers allow
/// and the misses as evenly over the requests, the file's among them as evenly
/// again.
fn layer_period(shares: &dyn Fn(usize) -> Shares, routed: f64, choices: u64, held: HeldSlots, rows: f64, period: usize) -> Vec<LayerLoad> {
    let misses = |slots: usize| (shares(slots).misses * routed * period as f64).round() as usize;
    let total = misses(held.slots);
    let filed = misses(held.slots.max(held.host)).min(total);
    let missing = 1. - (1. - shares(held.slots).requests).powf(rows);
    let requests = ((missing * period as f64).round() as usize).clamp(total.min(1), total.min(period));
    let quiet = LayerLoad { routed, rows: choices as f64 * rows / routed, ..LayerLoad::default() };
    let mut loads = vec![quiet; period];
    for request in 0..requests {
        let (first, last) = (request * total / requests, (request + 1) * total / requests);
        let file = last * filed / total - first * filed / total;
        loads[request * period / requests] = LayerLoad { tier: (last - first - file) as u64, file: file as u64, ..quiet };
    }
    loads
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bank's misses are the generic model's share of its routed experts,
    /// spread over every step while it misses more than a step's worth; a
    /// host tier takes those it holds and a whole bank misses none.
    #[test]
    fn a_layer_misses_as_the_generic_model_gives() {
        let model = MissModel::of(32, 4);
        let shares = |slots: usize| generic_shares(&model, slots as f64 / 32.);
        let period = |held| layer_period(&shares, 4., 4, held, 1., 320);
        // 8 of 32 experts, a quarter: 4 a row, one row.
        let misses = (model.misses(0.25) * 4. * 320.).round() as u64;
        let loads = period(HeldSlots::bank(8));
        assert_eq!(loads.iter().map(|load| load.tier + load.file).sum::<u64>(), misses);
        assert!(loads.iter().all(|load| load.tier == 0), "no host tier: every miss is the file's");
        let requests = (model.requests(model.misses(0.25)) * 320.).round() as usize;
        assert_eq!(loads.iter().filter(|load| load.file > 0).count(), requests, "a miss's steps are the requests'");
        let tiered = period(HeldSlots { slots: 8, host: 32 });
        assert!(tiered.iter().all(|load| load.file == 0));
        assert_eq!(tiered.iter().map(|load| load.tier).sum::<u64>(), misses);
        assert!(period(HeldSlots::bank(32)).iter().all(|load| load.tier + load.file == 0));
    }

    /// A file miss lands at the convert's rate only with no host tier.
    #[test]
    fn a_file_miss_lands_through_the_convert_with_no_host_tier() {
        let host = HostLines { cpu: None, memory: None, file: None, convert: None, record_bytes: 2_000_000 };
        let line = landing(&host, 1e-10).unwrap().expect("no tier: the device lands it");
        assert!((line.seconds(3) - 3. * 2e-4).abs() < 1e-12);
        let tier = HostLines { memory: Some(Nanoseconds::saturating(1.)), ..host };
        assert_eq!(landing(&tier, 1e-10).unwrap(), None);
    }

    /// A load prices on the generic model counted as one more step: at its
    /// first counted step half each, and its own reads as it decodes; a bank
    /// stops growing where its reads stop falling, first reads apart.
    #[test]
    fn a_load_starts_generic_and_its_own_decode_takes_over() {
        // Four experts: reads found at depths 0 and 1, and a first read.
        let reads = LayerReads::of_depths(&[2, 1, 0, 0, 1], &[1, 1, 0, 0, 1]);
        assert_eq!(reads.saturation(), Some(2));
        assert_eq!(LayerReads::of_depths(&[0, 0, 0, 0, 3], &[0, 0, 0, 0, 3]).saturation(), None);
        let live = |steps| LiveDecode { layers: vec![reads.clone()], steps, rows: 1., depth: 100 };
        let prior = Shares { misses: 0.5, requests: 1. };
        let first = live(1).shares(0, 2, prior);
        assert_eq!(first.misses, (0.5 + 0.25) / 2., "a read of 4 at or past 2 slots: the first read");
        assert!((live(1_000_000).shares(0, 2, prior).misses - 0.25).abs() < 1e-5);
        let none = LiveDecode { layers: vec![LayerReads::of_depths(&[0; 5], &[0; 5])], steps: 1, rows: 1., depth: 100 };
        assert_eq!(none.shares(0, 2, prior).misses, 0.5, "a layer that read nothing keeps its prior");
    }
}
