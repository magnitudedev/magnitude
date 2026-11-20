//! The streamed decode estimate: what a streamed plan's decode step takes on
//! this machine, from the plan's structure and the machine's stored figures
//! alone. No model's own measurement enters it, so a model never loaded is
//! estimated as one loaded many times.
//!
//! - **Misses** ([`MissModel`]): the one generic model of how a bank of a
//!   share `s` of a layer's experts misses its routed experts, the share
//!   `m(s)` of a step's distinct routed experts it lacks, of that layer's `E`
//!   and `k` alone: placement's fixed point at its bound, `1 − s`, the
//!   uninformed routing's. A layer reads `D(r)` distinct experts a step of
//!   `r` rows, so a bank of `slots` misses `m(slots/E)·D(r)` of them, spread
//!   over steps as evenly as they fall; of those misses, the ones a host tier
//!   also lacks (`m(host/E)·D(r)`, the tier inclusive of the bank) are read
//!   from the file and the rest from the tier. A layer's misses of a step are
//!   one request at its route, a share `q(s)` of a row's steps requesting.
//! - **Lines**: the GPU's over each layer's expert from the executor's
//!   costs ([`crate::ExpertPass::lines`]), one expert's stored bytes from the plan,
//!   a host tier's record read at the host memory's line, a CPU site's
//!   misses where the plan has one at the site's lines of a reference form
//!   scaled by the record's bytes, the file's reads at the
//!   model store's read line and, with no host tier, a request's landing
//!   past its last read at one record's device convert (its landing launch
//!   converts each record as its read lands), all measured by the
//!   measurement job
//!   (the executor's `StreamedFigures`).
//! - **The step** is the one step model's ([`crate::step`]) on `C`, the
//!   step's compute in the form its holding decodes in ([`FormSeconds`]),
//!   with no read ahead: a hint saves only what it lands before its layer,
//!   which no stored figure prices. A bank holding every streamed layer's
//!   experts decodes the resident forms and misses nothing
//!   ([`StreamedLayers::whole`]); any smaller one the streamed forms, whose
//!   route, landing launch and CPU site entries cost at any misses, none
//!   included, so the last slot that makes the bank whole saves them too.
//!
//! A load prices its experts' decisions on the same step ([`DecodePrice`]):
//! at each layer its own decode has read this session, what that decode
//! read ([`LiveDecode`]), and the generic model only at a layer it has not,
//! at its decode's rows and context.
//! Nothing it counts is stored. Whether two-row decode runs is no
//! projection of this step's: it is the measured regime's
//! ([`crate::regime`]).
//!
use crate::expert_lines::ExpertLines;
use crate::lines::ReadLine;
use crate::miss_model::MissModel;
use crate::step::{LayerLoad, LayerRun, ReadAhead, StepInputs};
use std::collections::HashMap;


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
/// (the executor's `experts::directory::Stack::live`): its layers' reads, its
/// steps, their mean rows (exact rows: a front row reads nothing of its own
/// step) and context tokens, and what its front rows foresaw of them
/// (counted by the ranking, which no price reads).
#[derive(Clone, Debug, PartialEq)]
pub struct LiveDecode {
    pub layers: Vec<LayerReads>,
    pub steps: u64,
    pub rows: f64,
    pub depth: u32,
}

impl LiveDecode {
    /// What `layer`'s decode read at a bank of `slots`: where the layer read
    /// anything, its measured shares; where it read nothing, `model`'s, the
    /// generic model's bound. A layer that read has a measured share at
    /// every depth of its stack: a read found at depth `d` misses every bank
    /// of `d` slots or fewer, and its first read misses a bank of any size,
    /// so every tail counts at least that read. The measurement is used
    /// whole, from its first read: no weight is set between it and the
    /// model. A step of several rows missing at all with a row's share `q`
    /// misses with `1 − (1 − q)^rows`, so its requests count at `q`.
    fn shares(&self, layer: usize, slots: usize, model: Shares) -> Shares {
        let reads = &self.layers[layer];
        let all = reads.misses[0];
        if all == 0 {
            return model;
        }
        let at = slots.min(reads.misses.len() - 1);
        let missing = (reads.requests[at] as f64 / self.steps as f64).min(1.);
        Shares { misses: reads.misses[at] as f64 / all as f64, requests: 1. - (1. - missing).powf(self.rows.recip()) }
    }

    /// The distinct experts a step of `layer` read, its mean over the
    /// counted steps (each read counted once, at its depth); none where it
    /// read nothing, where the generic model's `D(r)` stands.
    fn routed(&self, layer: usize) -> Option<f64> {
        match self.layers[layer].misses[0] {
            0 => None,
            all => Some(all as f64 / self.steps as f64),
        }
    }

    /// The fewest slots past which its reads stop falling
    /// ([`LayerReads::saturation`]): the deepest layer's; none before any
    /// layer found a read in its stack.
    fn saturation(&self) -> Option<u64> {
        self.layers.iter().filter_map(LayerReads::saturation).max()
    }
}

/// A banked plan's decode step compute in each form it decodes in
/// (`LaunchPrices::launch`): `resident` while its bank holds every
/// streamed layer's experts and decode runs the resident forms, `streamed`
/// otherwise, the streamed forms' entries and their submission boundaries
/// beside them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FormSeconds {
    pub resident: f64,
    pub streamed: f64,
}

/// A streamed plan's layers as the estimate prices them, in decoder order,
/// as runs of consecutive layers alike: each run's `E`, its generic model,
/// its lines and its layers, and the plan's `k`.
pub struct StreamedLayers {
    runs: Vec<StreamedRun>,
    /// Each layer's lines, in decoder order.
    lines: std::sync::Arc<[ExpertLines]>,
    choices: u64,
}

/// Consecutive streamed layers of one kind: `experts` each, priced by one
/// line, `layers` of them.
struct StreamedRun {
    experts: u64,
    model: MissModel,
    layers: usize,
}

impl StreamedLayers {
    /// Layers of `experts` experts each, routing `choices` a row, each
    /// priced by its `lines` (a load's own), in decoder order: consecutive
    /// layers of equal experts and lines one run.
    ///
    /// # Panics
    /// Where there is no layer, or a layer has no lines of its own.
    pub fn new(experts: Vec<u64>, choices: u64, lines: std::sync::Arc<[ExpertLines]>) -> Self {
        assert!(!experts.is_empty(), "a streamed plan streams a layer");
        assert_eq!(lines.len(), experts.len(), "a streamed layer's lines are its own");
        let mut runs = Vec::<StreamedRun>::new();
        for (layer, &experts) in experts.iter().enumerate() {
            match runs.last_mut() {
                Some(run) if run.experts == experts && lines[layer - 1] == lines[layer] => run.layers += 1,
                _ => runs.push(StreamedRun { experts, model: MissModel::of(experts, choices), layers: 1 }),
            }
        }
        Self { runs, lines, choices }
    }

    /// The streamed layers.
    pub fn layers(&self) -> usize {
        self.runs.iter().map(|run| run.layers).sum()
    }

    /// Whether holding `held` makes the bank whole: its slots hold every
    /// streamed layer's experts, so decode runs the resident forms and
    /// misses nothing. The one decision of which form a holding decodes in.
    pub fn whole(&self, held: HeldSlots) -> bool {
        self.runs.iter().all(|run| held.slots as u64 >= run.experts)
    }

    /// Prepare the deduplicated step patterns of a period of `rows` rows
    /// holding `held` (and `live`'s reads where counted), reusable across
    /// compute times (such as context depths) via
    /// [`StreamedPeriod::step_seconds`].
    pub fn prepare(&self, live: Option<&LiveDecode>, held: HeldSlots, rows: f64) -> StreamedPeriod<'_> {
        if self.whole(held) {
            return StreamedPeriod { layers: self, whole: true, patterns: Vec::new() };
        }
        StreamedPeriod { layers: self, whole: false, patterns: self.period(live, held, rows) }
    }

    /// `T` of a decode step of `rows` rows holding `held` of each layer's
    /// experts, on `compute` (`C` of each form): a whole bank's
    /// ([`Self::whole`]) the resident form's alone; any other's the streamed
    /// form's, its file reads along `read`, its misses those its own decode
    /// read (`live`) at the layers it read, the generic model's
    /// ([`MissModel`]) elsewhere: each
    /// layer's misses and requests as whole counts over a period of the
    /// widest layer's `E` steps, each spread as evenly as integers allow,
    /// priced by the one step model over the period ([`Self::period`]), with
    /// no read ahead.
    pub fn step_seconds(&self, live: Option<&LiveDecode>, read: ReadLine, held: HeldSlots, rows: f64, compute: FormSeconds) -> StreamedStep {
        self.prepare(live, held, rows).step_seconds(read, compute)
    }

    /// The period of the widest layer's `E` steps of `rows` rows holding
    /// `held`, aggregated into distinct step patterns in order first seen with
    /// their step counts: its layers' loads ([`fill_layer_period`]) as runs of
    /// consecutive layers whose inputs are one by construction, their misses
    /// and distinct experts `live`'s where it read and the generic model's
    /// elsewhere ([`LiveDecode::shares`], [`LiveDecode::routed`]).
    /// The generic model loads a kind's consecutive layers alike, so without
    /// `live` a run is its kind's; a load's own reads differ layer by layer,
    /// so with it each layer is a run of its own. Within a period every group's
    /// `routed`, `rows`, `recall` and `layers` are fixed, so `(tier, file)`
    /// across groups identifies each distinct step exactly.
    fn period(&self, live: Option<&LiveDecode>, held: HeldSlots, rows: f64) -> Vec<(Vec<LayerRun>, usize)> {
        let period = self.runs.iter().map(|run| run.experts).max().expect("a streamed plan streams a layer") as usize;
        let width = match live {
            None => self.runs.len(),
            Some(_) => self.layers(),
        };
        let mut flat = vec![LayerRun { load: LayerLoad::default(), layers: 0 }; period * width];
        let mut col = 0;
        let mut layer = 0;
        for run in &self.runs {
            let model = |slots: usize| generic_shares(&run.model, slots as f64 / run.experts as f64);
            match live {
                None => {
                    fill_layer_period(&model, run.model.routed(rows), self.choices, held, rows, period, run.layers, &mut flat, width, col);
                    col += 1;
                }
                Some(live) => {
                    for at in layer..layer + run.layers {
                        let routed = live.routed(at).unwrap_or_else(|| run.model.routed(rows));
                        fill_layer_period(&|slots| live.shares(at, slots, model(slots)), routed, self.choices, held, rows, period, 1, &mut flat, width, col);
                        col += 1;
                    }
                }
            }
            layer += run.layers;
        }
        const LINEAR_LIMIT: usize = 8;
        let mut patterns = Vec::<(Vec<LayerRun>, usize)>::new();
        let mut seen = HashMap::<Vec<(u64, u64)>, usize>::new();
        let mut key = Vec::<(u64, u64)>::with_capacity(width);
        for step in 0..period {
            let row = &flat[step * width..(step + 1) * width];
            key.clear();
            key.extend(row.iter().map(|run| (run.load.tier, run.load.file)));
            if seen.is_empty() {
                if let Some((_, count)) = patterns.iter_mut().find(|(runs, _)| {
                    runs.iter().zip(&key).all(|(run, &(tier, file))| run.load.tier == tier && run.load.file == file)
                }) {
                    *count += 1;
                    continue;
                }
                patterns.push((row.to_vec(), 1));
                if patterns.len() > LINEAR_LIMIT {
                    seen.reserve(patterns.len());
                    for (at, (runs, _)) in patterns.iter().enumerate() {
                        seen.insert(runs.iter().map(|run| (run.load.tier, run.load.file)).collect(), at);
                    }
                }
            } else if let Some(&at) = seen.get(key.as_slice()) {
                patterns[at].1 += 1;
            } else {
                seen.insert(key.clone(), patterns.len());
                patterns.push((row.to_vec(), 1));
            }
        }
        patterns
    }

    /// The seconds a prompt pass of `rows` rows waits on its misses' reads
    /// holding `held`: the step model's ([`Self::step_seconds`]) at those
    /// rows, the generic model's misses, with no compute beside them to
    /// cover them; none where `held` makes the bank whole.
    pub(crate) fn pass_reads(&self, read: ReadLine, held: HeldSlots, rows: f64) -> f64 {
        self.step_seconds(None, read, held, rows, FormSeconds { resident: 0., streamed: 0. }).total_seconds
    }

    /// The free path records needed for moves of `bytes` stored bytes along
    /// `read`, the bank holding `held` with `hint_leads` lookahead layers
    /// across `num_layers` routed layers and decode compute time `compute`.
    /// Single owner shared by [`crate::BankPrices::paths`] and assessment's
    /// path pricer.
    pub fn path_records(
        &self,
        read: ReadLine,
        compute: FormSeconds,
        bytes: u64,
        held: HeldSlots,
        hint_leads: u64,
        num_layers: u64,
    ) -> Option<crate::PathRecords> {
        let step = self.step_seconds(None, read, held, 1.0, compute).total_seconds;
        let reads = self.pass_reads(read, held, 1.0);
        let queue = if num_layers == 0 {
            0.0
        } else {
            hint_leads as f64 * (reads / num_layers as f64)
        };
        crate::PathRecords::of(read, bytes, queue, crate::PathRecords::release(step, reads))
    }
}

/// The deduplicated step patterns of a [`StreamedLayers`] over one period at a
/// fixed holding and row count ([`StreamedLayers::prepare`]), reusable across
/// compute times (such as context depths).
pub struct StreamedPeriod<'a> {
    layers: &'a StreamedLayers,
    whole: bool,
    patterns: Vec<(Vec<LayerRun>, usize)>,
}

impl StreamedPeriod<'_> {
    /// `T` of a decode step on `compute` (`C` of each form), its file reads
    /// along `read`: identical to [`StreamedLayers::step_seconds`] at the
    /// holding and row count this period was prepared for.
    pub fn step_seconds(&self, read: ReadLine, compute: FormSeconds) -> StreamedStep {
        if self.whole {
            return StreamedStep { compute_seconds: compute.resident, exposed_seconds: 0., total_seconds: compute.resident };
        }
        let compute_seconds = compute.streamed;
        let step = step_inputs(compute_seconds, read, ReadAhead::NONE, &self.layers.lines)
            .run(self.patterns.iter().map(|(runs, count)| (runs.as_slice(), *count)));
        StreamedStep { compute_seconds, exposed_seconds: step.total - compute_seconds, total_seconds: step.total }
    }
}

/// What the step model prices a streamed step on: `compute` with no readout
/// apart, its reads along `read` and read ahead as `ahead`, its layers'
/// `lines`.
pub fn step_inputs(compute: f64, read: ReadLine, ahead: ReadAhead, lines: &[ExpertLines]) -> StepInputs<'_> {
    StepInputs { compute, head: 0., overlap: 1., read, sustained: read.seconds_per_byte(), ahead, lines }
}

/// A load's decode step as its experts' decisions price it at any holding:
/// its layers', on what its counted decode read ([`LiveDecode`]) where it
/// read and the generic model elsewhere, at that decode's rows, on `C`, the
/// step of those rows at its context in each form, its file read along the
/// model store's read line: the one step assessment prices
/// ([`StreamedLayers::step_seconds`]).
pub struct DecodePrice {
    layers: StreamedLayers,
    /// The decode the load counted; none before it counted a step, when the
    /// generic model alone prices it ([`Self::prior`]).
    live: Option<LiveDecode>,
    rows: f64,
    depth: u32,
    read: ReadLine,
    compute: FormSeconds,
}

impl DecodePrice {
    /// `live`'s decode on `layers`, its step's compute `compute` (`C` of
    /// each form at `live`'s rows and context).
    pub fn new(layers: StreamedLayers, live: LiveDecode, read: ReadLine, compute: FormSeconds) -> Self {
        assert_eq!(layers.layers(), live.layers.len(), "a decode's reads are of its streamed layers");
        Self { layers, rows: live.rows, depth: live.depth, live: Some(live), read, compute }
    }

    /// The decode before the load counted a step: the generic model's misses
    /// alone ([`MissModel`]), at `rows` rows at the start of its context on
    /// `compute` of them, the price assessment gives the same layers from
    /// headers and the machine's figures, which the load's own reads
    /// replace at each layer it reads once it counts them ([`Self::new`]).
    pub fn prior(layers: StreamedLayers, rows: f64, read: ReadLine, compute: FormSeconds) -> Self {
        Self { layers, live: None, rows, depth: 0, read, compute }
    }

    /// The context tokens a row of the counted decode reads.
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Whether the bank holds every streamed layer's experts at `held`.
    pub fn whole(&self, held: HeldSlots) -> bool {
        self.layers.whole(held)
    }

    /// The step holding `held`.
    pub fn step(&self, held: HeldSlots) -> StreamedStep {
        self.layers.step_seconds(self.live.as_ref(), self.read, held, self.rows, self.compute)
    }

    /// The fewest slots at which the decode's own reads stop falling; none
    /// before it found a read in its stack.
    pub fn saturation(&self) -> Option<u64> {
        self.live.as_ref().and_then(LiveDecode::saturation)
    }
}

/// What `model` gives a bank of a share `share` of its layer's experts, a
/// bank past them holding them all.
fn generic_shares(model: &MissModel, share: f64) -> Shares {
    let share = share.min(1.);
    Shares { misses: model.misses(share), requests: model.requests(share) }
}

/// Fill column `col` of row-major `flat` (`period × width`) with a layer's
/// loads over `period` steps of `rows` rows holding `held`: its misses and
/// requests as `shares` give them at a bank of some slots, as whole counts
/// over the period, the requests spread as evenly over it as integers allow
/// and the misses as evenly over the requests, the file's among them as evenly
/// again.
#[allow(clippy::too_many_arguments)]
fn fill_layer_period(
    shares: &dyn Fn(usize) -> Shares,
    routed: f64,
    choices: u64,
    held: HeldSlots,
    rows: f64,
    period: usize,
    layers: usize,
    flat: &mut [LayerRun],
    width: usize,
    col: usize,
) {
    let misses = |slots: usize| (shares(slots).misses * routed * period as f64).round() as usize;
    let total = misses(held.slots);
    let filed = misses(held.slots.max(held.host)).min(total);
    let missing = 1. - (1. - shares(held.slots).requests).powf(rows);
    let requests = ((missing * period as f64).round() as usize).clamp(total.min(1), total.min(period));
    let quiet = LayerLoad { routed, rows: choices as f64 * rows / routed, ..LayerLoad::default() };
    let quiet_run = LayerRun { load: quiet, layers };
    for step in 0..period {
        flat[step * width + col] = quiet_run;
    }
    for request in 0..requests {
        let (first, last) = (request * total / requests, (request + 1) * total / requests);
        let file = last * filed / total - first * filed / total;
        flat[(request * period / requests) * width + col] = LayerRun {
            load: LayerLoad { tier: (last - first - file) as u64, file: file as u64, ..quiet },
            layers,
        };
    }
}

/// A layer routing `routed` distinct experts a step and `choices` a row, its
/// loads over a period of `period` steps of `rows` rows holding `held`.
#[cfg(test)]
fn layer_period(shares: &dyn Fn(usize) -> Shares, routed: f64, choices: u64, held: HeldSlots, rows: f64, period: usize) -> Vec<LayerLoad> {
    let mut flat = vec![LayerRun { load: LayerLoad::default(), layers: 1 }; period];
    fill_layer_period(shares, routed, choices, held, rows, period, 1, &mut flat, 1, 0);
    flat.into_iter().map(|run| run.load).collect()
}

/// A streamed decode step as the one cost function prices it
/// ([`StreamedLayers::step_seconds`]): `C`, what its
/// layers' misses add, and `T`, their sum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StreamedStep {
    pub compute_seconds: f64,
    pub exposed_seconds: f64,
    pub total_seconds: f64,
}

/// What a load holds of each streamed layer's experts: `slots` in the bank,
/// and `host` in a host tier, inclusive of the bank's (none on unified
/// memory): the tier holds the bank's experts too, so it holds more only
/// past `slots`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeldSlots {
    pub slots: usize,
    pub host: usize,
}

impl HeldSlots {
    /// `slots` in the bank and nothing in a host tier.
    pub fn bank(slots: usize) -> Self {
        Self { slots, host: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A read of 0.4 ms and 2 GB/s.
    const READ: ReadLine = match ReadLine::new(4e-4, 1. / 2e9) {
        Ok(line) => line,
        Err(_) => panic!("the tests' read line"),
    };

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
        let requests = (model.requests(0.25) * 320.).round() as usize;
        assert_eq!(loads.iter().filter(|load| load.file > 0).count(), requests, "a miss's steps are the requests'");
        let tiered = period(HeldSlots { slots: 8, host: 32 });
        assert!(tiered.iter().all(|load| load.file == 0));
        assert_eq!(tiered.iter().map(|load| load.tier).sum::<u64>(), misses);
        assert!(period(HeldSlots::bank(32)).iter().all(|load| load.tier + load.file == 0));
    }

    /// The last slot that makes a bank whole saves the streamed form's own
    /// compute beside its misses: at every expert but one the step runs the
    /// streamed form, at every expert the resident form and misses nothing.
    #[test]
    fn a_whole_bank_saves_the_streamed_forms_compute() {
        let lines = ExpertLines::reference(1_000_000);
        let layers = StreamedLayers::new(vec![32, 32], 4, std::sync::Arc::from([lines, lines]));
        let read = READ;
        let compute = FormSeconds { resident: 0.010, streamed: 0.0115 };
        let step = |slots| layers.step_seconds(None, read, HeldSlots::bank(slots), 1., compute).total_seconds;
        assert_eq!(step(32), compute.resident, "a whole bank decodes the resident forms and misses nothing");
        assert!(
            step(31) - step(32) >= compute.streamed - compute.resident,
            "{} s at 31 slots, {} s at 32",
            step(31),
            step(32)
        );
    }

    /// A load prices a layer on its own reads from the first it counts, at
    /// every bank size, and on the generic model only where it read
    /// nothing; a bank stops growing where its reads stop falling, first
    /// reads apart.
    #[test]
    fn a_layers_own_reads_price_it_once_it_has_read() {
        // Four experts: reads found at depths 0 and 1, and a first read.
        let reads = LayerReads::of_depths(&[2, 1, 0, 0, 1], &[1, 1, 0, 0, 1]);
        assert_eq!(reads.saturation(), Some(2));
        assert_eq!(LayerReads::of_depths(&[0, 0, 0, 0, 3], &[0, 0, 0, 0, 3]).saturation(), None);
        let live = LiveDecode { layers: vec![reads], steps: 1, rows: 1., depth: 100 };
        let model = Shares { misses: 0.5, requests: 1. };
        let measured = |slots| live.shares(0, slots, model);
        assert_eq!(measured(2), Shares { misses: 0.25, requests: 1. }, "a read of 4 at or past 2 slots: the first read");
        assert_eq!(measured(0), Shares { misses: 1., requests: 1. });
        assert_eq!(measured(4).misses, 0.25, "past its deepest read, its first read");
        let none = LiveDecode { layers: vec![LayerReads::of_depths(&[0; 5], &[0; 5])], steps: 1, rows: 1., depth: 100 };
        assert_eq!(none.shares(0, 2, model), model, "a layer that read nothing is the model's");
        assert_eq!((live.routed(0), none.routed(0)), (Some(4.), None), "its 4 reads in its one step");
    }

    /// Deduplicating a period into weighted distinct patterns preserves the
    /// exact step time (both in the small linear path and the hash-map path)
    /// and a prepared period can be reused across compute times.
    #[test]
    fn prepared_period_deduplicates_steps_and_matches_expanded_run() {
        let lines = ExpertLines::reference(1_000_000);
        let layers = StreamedLayers::new(vec![512; 4], 8, std::sync::Arc::from([lines; 4]));
        let held = HeldSlots { slots: 120, host: 300 };
        let prepared = layers.prepare(None, held, 1.);
        assert!(!prepared.whole);
        assert!(prepared.patterns.len() <= 8, "expected few distinct patterns, got {}", prepared.patterns.len());
        assert_eq!(prepared.patterns.iter().map(|(_, count)| *count).sum::<usize>(), 512);

        // Compare against un-deduplicated 512 weight-1 steps.
        let expanded = prepared
            .patterns
            .iter()
            .flat_map(|(runs, count)| std::iter::repeat_n((runs.as_slice(), 1usize), *count))
            .collect::<Vec<_>>();
        for compute in [
            FormSeconds { resident: 0.010, streamed: 0.012 },
            FormSeconds { resident: 0.020, streamed: 0.025 },
        ] {
            let direct = layers.step_seconds(None, READ, held, 1., compute);
            let reused = prepared.step_seconds(READ, compute);
            assert_eq!(reused, direct);
            let reference = step_inputs(compute.streamed, READ, ReadAhead::NONE, &layers.lines).run(expanded.iter().copied());
            assert_eq!(reused.total_seconds, reference.total);
        }

        // Whole bank skips period construction and returns resident compute.
        let whole = layers.prepare(None, HeldSlots::bank(512), 1.);
        assert!(whole.whole);
        assert!(whole.patterns.is_empty());
        let compute = FormSeconds { resident: 0.010, streamed: 0.012 };
        assert_eq!(whole.step_seconds(READ, compute), StreamedStep { compute_seconds: 0.010, exposed_seconds: 0., total_seconds: 0.010 });
    }
}
