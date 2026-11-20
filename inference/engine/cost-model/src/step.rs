//! The one model of a streamed decode step: a
//! timeline of the step's layers and the reads they wait for. Its instances
//! differ only in their inputs ([`StepInputs`], [`LayerLoad`]):
//!
//! | Instance | Misses | When a miss's read is issued |
//! |---|---|---|
//! | frontier | the ranking's ideal replay | every one at the step's start |
//! | reachable | the ranking's ideal replay | a hint's `ℓ` layers ahead at the run's recall and precision, the rest on demand |
//! | estimate | the record's steady-state curve | a hint's `ℓ` layers ahead at its recall, the rest on demand |
//! | live check | the step's own | the step's own hints |
//!
//! **One step.** A layer is one barrier. Its route names its misses and the
//! hints of the layers ahead, the device stops there until the misses are
//! whole, then computes the layer. So the step is its compute plus what its
//! barriers wait for, and a read overlaps the compute only of the layers
//! before the one that needs it, never that layer's own. Reads complete in
//! the order issued, one device:
//!
//! ```text
//! c_k = max(issue_k + λ + b/P, c_{k−1} + b/P)
//! ```
//!
//! `λ` and `P` the device's latency and peak rate, its raw line. A layer's
//! records are whole at `r_l`, its last read's completion and its landing's
//! line after it. Layer `l`'s route comes at `f_{l−1}`, where its demand
//! reads and the hints it names are issued: a hint `ℓ` layers ahead reads for
//! layer `l + ℓ`, and one that lands before that layer's route is a hit there,
//! which is all a hint ever saves. Compute runs at rate `η` while the device
//! reads and at rate 1 after (`t ⊕ w`), so with no host tier layer `l` ends
//! at `f_l = max(f_{l−1}, r_l) ⊕ τ_l`. Beside a host tier the layer ends at
//! `f_l = f_{l−1} ⊕ τ_l + s_l`, `s_l` what its misses served on the CPU or
//! over the link show past its hits ([`ExpertLines::serve`], host memory
//! bounding the CPU, the link and the moves' reads together). A miss read
//! in place is the GPU's own work at its layer's barrier, so it adds in
//! series; the link's moves run on the copy engine beside every layer.
//! `τ_l` is its share of the resident step `(C − h)/L`, and the step ends
//! at `T_s = f_{L−1} ⊕ h`, `h` the readout's part of `C`.
//!
//! **A run** of steps takes the slowest of its terms
//! ([`crate::Terms`]): `T = max(mean T_s, mean B_s / S, mean K_s)`. No burst
//! beats the drive's sustained rate `S` over a long run, `B_s` the step's
//! read bytes, wrong hints' among them; nor the link's, `K_s` the step's
//! reads in place and its moves on it.

use crate::expert_lines::ExpertLines;
use crate::lines::ReadLine;
use std::collections::HashMap;

/// One streamed layer's reads in one step: the distinct experts its route
/// chose, each of `rows` rows; of them those served from the host tier;
/// those a bank without read-ahead would read from the file; and the share
/// of those a hint reads ahead (`R`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LayerLoad {
    pub routed: f64,
    pub rows: f64,
    pub tier: u64,
    pub file: u64,
    pub recall: f64,
}

/// How far ahead hinted reads are issued.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Lead {
    /// From the route `ℓ` layers before the layer they are for, the hints'
    /// mean.
    Layers(f64),
    /// At the step's start: every route of the token known once it exists.
    Token,
}

/// What hinted reads are: issued at `lead`, `precision` (`P`) of them read
/// an expert their layer's route chose, the wrong ones read beside them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadAhead {
    pub precision: f64,
    pub lead: Lead,
}

impl ReadAhead {
    /// No read is predicted.
    pub const NONE: Self = Self { precision: 1., lead: Lead::Layers(0.) };
}

/// What every step is priced with.
#[derive(Clone, Copy, Debug)]
pub struct StepInputs<'a> {
    /// `C`, the resident step.
    pub compute: f64,
    /// `h`, the readout's part of `C`, after the last layer.
    pub head: f64,
    /// `η`, compute retained while reads run.
    pub overlap: f64,
    /// `λ + b/P`, the device's raw line at its peak.
    pub read: ReadLine,
    /// `1/S`, the device's sustained seconds a byte.
    pub sustained: f64,
    pub ahead: ReadAhead,
    /// Each streamed layer's lines, in decoder order.
    pub lines: &'a [ExpertLines],
}

/// A run's step and its terms, each a mean over the run's steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepTime {
    pub compute: f64,
    /// The step's own schedule: `mean T_s`.
    pub scheduled: f64,
    /// The device's sustained bound: `mean B_s / S`.
    pub sustained: f64,
    /// Of the schedule, what the layers' CPU and link service added.
    pub served: f64,
    /// The link's term: its reads in place and its moves, `mean K_s`.
    pub link: f64,
    /// Read bytes a step.
    pub bytes: f64,
    pub total: f64,
}

/// A run of `layers` consecutive streamed layers with identical inputs in
/// one step: the load every layer of it has, its lines one
/// ([`StepInputs::lines`]); a kind's consecutive layers, under a prior that
/// loads each alike.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LayerRun {
    pub load: LayerLoad,
    pub layers: usize,
}

/// The one device a step's reads queue on: reads complete in the order
/// issued, `c_k = max(issue_k + λ + b/P, c_{k−1} + b/P)`.
#[derive(Clone, Copy, Debug)]
struct Device {
    latency: f64,
    byte: f64,
    /// When the reads issued so far are done.
    free: f64,
}

impl Device {
    fn new(read: ReadLine) -> Self {
        Self { latency: read.latency_seconds(), byte: read.seconds_per_byte(), free: 0. }
    }

    /// `records` reads of `bytes` each issued at `at`: when they are done.
    fn read(&mut self, at: f64, records: f64, bytes: f64) -> f64 {
        if records > 0. {
            self.free = (at + self.latency + bytes * self.byte).max(self.free + bytes * self.byte) + (records - 1.) * bytes * self.byte;
        }
        self.free
    }
}

/// How closely a run's queue must repeat itself, against the step's own
/// time, for its remaining layers to be taken as repeating it: the
/// rounding of the timeline's sums.
const SETTLED: f64 = 4. * f64::EPSILON;

impl StepInputs<'_> {
    /// `w` of compute from `t` while the step's reads run until `busy`.
    fn advance(&self, t: f64, w: f64, busy: f64) -> f64 {
        if t >= busy {
            t + w
        } else if t + w / self.overlap <= busy {
            t + w / self.overlap
        } else {
            busy + (w - (busy - t) * self.overlap)
        }
    }

    /// What a layer of `lines`' misses of `load` served on the CPU or over
    /// the link show past its hits, its file misses whole `ready` after the
    /// layer begins, and the link's seconds of the layer: its reads in place
    /// and its moves (none without a host tier: the GPU reads them in place
    /// from its own memory).
    fn service(&self, lines: &ExpertLines, load: LayerLoad, ready: f64) -> (f64, f64) {
        let Some(memory) = lines.host.memory else {
            return (0., 0.);
        };
        let total_misses = load.tier + load.file;
        let hits_experts = (load.routed - total_misses as f64).max(0.);
        let hit_ns = hits_experts * lines.gpu.resident.nanoseconds(load.rows);
        let (mut gpu, mut cpu) = (hit_ns, 0f64);
        let mut served_cpu = false;
        let mut link_count = 0u64;
        if total_misses > 0 {
            let link_busy = lines.gpu.moves.link.get();
            let link_ns = lines.gpu.link.nanoseconds(load.rows);
            let cpu_params = lines.host.cpu.map(|c| (c.expert.nanoseconds(load.rows), c.handshake.get()));
            // Identical to `ExpertLines::serve(Pass::Decode, ...)` for uniform `load.rows`:
            // `load.tier` misses ready at 0.0, followed by `load.file` misses ready at `ready`.
            for (count, ready_ns) in [(load.tier, 0.0), (load.file, ready * 1e9)] {
                for _ in 0..count {
                    let on_cpu = cpu_params.map(|(expert_ns, handshake)| {
                        let h = if served_cpu { 0. } else { handshake };
                        cpu.max(ready_ns) + expert_ns + h
                    });
                    let in_place = gpu.max(ready_ns).max(link_busy) + link_ns;
                    match on_cpu {
                        Some(end) if end <= in_place => {
                            cpu = end;
                            served_cpu = true;
                        }
                        _ => {
                            gpu = in_place;
                            link_count += 1;
                        }
                    }
                }
            }
        }
        let end_seconds = gpu.max(cpu) * 1e-9;
        // Host memory bounds the CPU, the link and the moves together:
        // every miss and every moved record leaves it at `t_ram`.
        let leave = (total_misses as f64 + lines.gpu.moves.records) * memory.get() * 1e-9;
        let hit = hit_ns * 1e-9;
        let in_place = link_count as f64 * lines.gpu.link.nanoseconds(load.rows);
        ((end_seconds.max(leave) - hit).max(0.), (in_place + lines.gpu.moves.link.get()) * 1e-9)
    }

    /// When a layer of `lines` routed at `route` ends: beside a host tier
    /// its compute and its `service` (its file misses land before the CPU
    /// or the link serves them, within it); without one its compute once
    /// its misses are whole at `ready`; `busy` the device's reads' end.
    fn finish(&self, lines: &ExpertLines, route: f64, ready: f64, tau: f64, busy: f64, service: f64) -> f64 {
        match lines.host.memory {
            Some(_) => self.advance(route, tau, busy) + service,
            None => self.advance(route.max(ready), tau, busy),
        }
    }

    /// When a layer of `lines`' `demand` file misses, read on demand at its
    /// route `route` on `device`, are whole: their reads' end and their
    /// landing past it; none without such misses.
    fn demanded(lines: &ExpertLines, device: &mut Device, route: f64, demand: u64) -> f64 {
        match demand {
            0 => 0.,
            _ => {
                let bytes = lines.host.record_bytes as f64;
                device.read(route, demand as f64, bytes) + lines.landed(demand, bytes * device.byte)
            }
        }
    }

    /// One layer of `lines` with `load`, routed at `route`, its reads on
    /// `device`: its misses whole at `ready` beside its `demand` file misses
    /// read on demand at its route, after which it issues what `ahead`
    /// reads (the hints it names of the layers ahead); its end, its CPU
    /// and link service, and the link's seconds of it ([`Self::service`]).
    #[allow(clippy::too_many_arguments)]
    fn layer(&self, lines: &ExpertLines, load: LayerLoad, route: f64, ready: f64, demand: u64, tau: f64, device: &mut Device, ahead: impl FnOnce(&mut Device)) -> (f64, f64, f64) {
        let ready = ready.max(Self::demanded(lines, device, route, demand));
        ahead(device);
        let busy = device.read(route, 0., 0.);
        let (service, link) = self.service(lines, load, (ready - route).max(0.));
        (self.finish(lines, route, ready, tau, busy, service), service, link)
    }

    /// One step of `loads` (each streamed layer's, in decoder order): `T_s`,
    /// the CPU and link service in it, its read bytes, and its link's
    /// seconds. A share of a layer's misses reads ahead in whole records:
    /// `carried` holds what each layer's rounding left, for its next step,
    /// so a run's share is its recall. [`Self::step`] of the layers as runs
    /// of one.
    pub fn one(&self, loads: &[LayerLoad], carried: &mut [f64]) -> (f64, f64, f64, f64) {
        let runs = loads.iter().map(|&load| LayerRun { load, layers: 1 }).collect::<Vec<_>>();
        self.step(&runs, &Self::hinted(&runs, carried))
    }

    /// Each layer's whole hinted reads of `runs`, each run one layer: its
    /// recall of its file misses beside the rounding `carried` holds of the
    /// steps before, which it leaves what this step's rounding left.
    ///
    /// # Panics
    /// Where a run is not one layer of `carried`'s.
    fn hinted(runs: &[LayerRun], carried: &mut [f64]) -> Vec<u64> {
        assert!(runs.len() == carried.len() && runs.iter().all(|run| run.layers == 1), "invariant violated: hinted reads are rounded layer by layer");
        runs.iter()
            .zip(carried.iter_mut())
            .map(|(run, carried)| {
                let due = *carried + run.load.recall * run.load.file as f64;
                let hinted = (due.floor() as u64).min(run.load.file);
                *carried = due - hinted as f64;
                hinted
            })
            .collect()
    }

    /// One step of `runs` (consecutive streamed layers, in decoder order),
    /// each run reading `hinted` of each of its layers' file misses ahead:
    /// `T_s`, the CPU and link service in it, its read bytes, and its
    /// link's seconds. One timeline through [`Self::layer`]: positional where every run is one
    /// layer, each layer's demand reads at its route, its hinted reads for
    /// the layer its lead is ahead, their wrong reads after them, or under
    /// a token's lead every read at the step's start in layer order, or
    /// under a pass's lead the hinted reads there and the rest on demand;
    /// collapsed where a run holds several identical layers, which nothing
    /// in the step reading ahead allows, a run's rest closed once its queue
    /// settles (a layer leaving the queue as it found it: its reads' lead
    /// on its end the same after it, or no reads and none queued), every
    /// layer after it repeating it, one layer's evaluation for a settled
    /// run, the queue's transient before it one layer each.
    ///
    /// # Panics
    /// Where the runs' layers are not the step's, a run's layers have other
    /// lines, a run's hinted reads are not its own, or a run of several
    /// layers is in a step that reads ahead.
    fn step(&self, runs: &[LayerRun], hinted: &[u64]) -> (f64, f64, f64, f64) {
        let layers = runs.iter().map(|run| run.layers).sum::<usize>();
        assert_eq!(layers, self.lines.len(), "invariant violated: a step's runs are its layers");
        assert_eq!(hinted.len(), runs.len(), "invariant violated: a step's hinted reads are its runs'");
        if runs.iter().any(|run| run.layers > 1) {
            let reads_ahead = self.ahead != ReadAhead::NONE || runs.iter().zip(hinted).any(|(run, &hinted)| run.load.recall != 0. || hinted > 0);
            assert!(!reads_ahead, "invariant violated: a step reading ahead is evaluated layer by layer");
        }
        let tau = (self.compute - self.head) / layers as f64;
        // Each run's lines, its hinted reads, its wrong hints', and its
        // demand reads, a layer.
        let mut first = 0;
        let reads = runs
            .iter()
            .zip(hinted)
            .map(|(run, &hinted)| {
                let lines = &self.lines[first];
                assert!(self.lines[first..first + run.layers].iter().all(|layer| layer == lines), "invariant violated: a run's layers have one line");
                first += run.layers;
                let wrong = hinted as f64 * (1. / self.ahead.precision - 1.);
                (lines, hinted, wrong, run.load.file - hinted)
            })
            .collect::<Vec<_>>();
        let bytes = runs.iter().zip(&reads).map(|(run, &(lines, _, wrong, _))| run.layers as f64 * lines.host.record_bytes as f64 * (run.load.file as f64 + wrong)).sum::<f64>();
        let mut device = Device::new(self.read);
        let byte = device.byte;
        // What `count` misses of run `at`'s layer add past their reads' end.
        let landing = |at: usize, count: u64| reads[at].0.landed(count, reads[at].0.host.record_bytes as f64 * byte);
        let mut ready = vec![0f64; runs.len()];
        // Under a token's lead every read is issued at the step's start, in
        // layer order, wrong hints after each layer's own.
        let (lead, on_demand) = match self.ahead.lead {
            Lead::Token => {
                for (at, &(lines, hinted, wrong, demand)) in reads.iter().enumerate() {
                    let bytes = lines.host.record_bytes as f64;
                    let started = hinted + demand;
                    if started > 0 {
                        ready[at] = device.read(0., started as f64, bytes) + landing(at, started);
                    }
                    device.read(0., wrong, bytes);
                }
                (None, false)
            }
            Lead::Layers(lead) => (Some(lead.round() as usize), true),
        };
        let (mut end, mut served_sum, mut link_sum) = (0f64, 0f64, 0f64);
        for (at, run) in runs.iter().enumerate() {
            let lines = reads[at].0;
            let mut left = run.layers;
            while left > 0 {
                // The layer's route: its demand reads, then the hints it
                // names of the layers ahead, their wrong reads after them;
                // under a token's lead, nothing issued at it.
                let (route, queued) = (end, device.free - end);
                let demand = if on_demand { reads[at].3 } else { 0 };
                let whole = ready[at];
                let ahead = |device: &mut Device| {
                    let Some((ahead, &(lines, hinted, wrong, _))) = lead.and_then(|lead| Some((at + lead, reads.get(at + lead)?))) else {
                        return;
                    };
                    let bytes = lines.host.record_bytes as f64;
                    if hinted > 0 {
                        ready[ahead] = ready[ahead].max(device.read(route, hinted as f64, bytes) + landing(ahead, hinted));
                    }
                    device.read(route, wrong, bytes);
                };
                let (layer_end, service, link) = self.layer(lines, run.load, route, whole, demand, tau, &mut device, ahead);
                (end, left) = (layer_end, left - 1);
                served_sum += service;
                link_sum += link;
                let settled = match run.load.file {
                    0 => queued <= 0.,
                    _ => ((device.free - end) - queued).abs() <= SETTLED * end.abs().max(tau),
                };
                if settled && left > 0 {
                    let repeated = left as f64;
                    let queue = device.free - end;
                    end += repeated * (layer_end - route);
                    served_sum += repeated * service;
                    link_sum += repeated * link;
                    if run.load.file > 0 {
                        device.free = end + queue;
                    }
                    left = 0;
                }
            }
        }
        let busy = device.free;
        (self.advance(end, self.head, busy), served_sum, bytes, link_sum)
    }

    /// Allocation-free fast path of [`Self::step`] when nothing reads ahead
    /// (`self.ahead == ReadAhead::NONE` and every run has `hinted == 0`).
    fn step_no_ahead(&self, runs: &[LayerRun]) -> (f64, f64, f64, f64) {
        let layers = runs.iter().map(|run| run.layers).sum::<usize>();
        assert_eq!(layers, self.lines.len(), "invariant violated: a step's runs are its layers");
        let tau = (self.compute - self.head) / layers as f64;
        let mut device = Device::new(self.read);
        let (mut end, mut served_sum, mut link_sum, mut bytes) = (0f64, 0f64, 0f64, 0f64);
        let mut first = 0;
        for run in runs {
            let lines = &self.lines[first];
            assert!(self.lines[first..first + run.layers].iter().all(|layer| layer == lines), "invariant violated: a run's layers have one line");
            first += run.layers;
            bytes += run.layers as f64 * lines.host.record_bytes as f64 * run.load.file as f64;
            let mut left = run.layers;
            while left > 0 {
                let (route, queued) = (end, device.free - end);
                let (layer_end, service, link) = self.layer(lines, run.load, route, 0., run.load.file, tau, &mut device, |_| {});
                (end, left) = (layer_end, left - 1);
                served_sum += service;
                link_sum += link;
                let settled = match run.load.file {
                    0 => queued <= 0.,
                    _ => ((device.free - end) - queued).abs() <= SETTLED * end.abs().max(tau),
                };
                if settled && left > 0 {
                    let repeated = left as f64;
                    let queue = device.free - end;
                    end += repeated * (layer_end - route);
                    served_sum += repeated * service;
                    link_sum += repeated * link;
                    if run.load.file > 0 {
                        device.free = end + queue;
                    }
                    left = 0;
                }
            }
        }
        let busy = device.free;
        (self.advance(end, self.head, busy), served_sum, bytes, link_sum)
    }

    /// A run of `steps`, each its runs of layers run its count of times in a
    /// row: `T`, the slowest of its schedule's, the drive's sustained rate's
    /// and the link's means.
    /// Where any step reads ahead (a recall, or a read ahead other than
    /// none), every run is evaluated as its layers one by one, else as it
    /// is ([`Self::step`]): the one choice, by the steps alone. Its hinted
    /// reads are whole, each layer's rounding carried from step to step in
    /// the run's order, so a run's hinted share is its recall; each distinct
    /// step, its runs and hinted reads alike, is priced once and weighed by
    /// its count, in the order first seen. A step reading nothing ahead
    /// rounds nothing, so its repetitions are one distinct step at once.
    pub fn run<'s>(&self, steps: impl IntoIterator<Item = (&'s [LayerRun], usize)>) -> StepTime {
        let steps = steps.into_iter().collect::<Vec<_>>();
        let ahead = self.ahead != ReadAhead::NONE || steps.iter().any(|(runs, _)| runs.iter().any(|run| run.load.recall != 0.));
        if !ahead {
            let mut seen = HashMap::<RunKey<'s>, usize>::with_capacity(steps.len());
            let mut unique = Vec::<(&'s [LayerRun], usize)>::with_capacity(steps.len());
            for (runs, count) in steps {
                let key = RunKey(runs);
                match seen.get(&key) {
                    Some(&at) => unique[at].1 += count,
                    None => {
                        seen.insert(key, unique.len());
                        unique.push((runs, count));
                    }
                }
            }
            return self.mean(unique.into_iter().map(|(runs, count)| (self.step_no_ahead(runs), count)));
        }
        let steps = steps
            .into_iter()
            .map(|(runs, count)| {
                let runs = runs.iter().flat_map(|run| std::iter::repeat_n(LayerRun { load: run.load, layers: 1 }, run.layers)).collect::<Vec<_>>();
                (runs, count)
            })
            .collect::<Vec<_>>();
        let mut carried = vec![0.; self.lines.len()];
        let mut rounded = Vec::new();
        for (runs, count) in &steps {
            let (rounds, each) = match runs.iter().all(|run| run.load.recall == 0.) {
                true => (1, *count),
                false => (*count, 1),
            };
            for _ in 0..rounds {
                let hinted = Self::hinted(runs, &mut carried);
                let key = (runs.iter().zip(&hinted))
                    .map(|(run, &hinted)| [run.load.routed.to_bits(), run.load.rows.to_bits(), run.load.tier, run.load.file, run.load.recall.to_bits(), run.layers as u64, hinted])
                    .collect::<Vec<_>>();
                rounded.push((key, (runs, hinted), each));
            }
        }
        self.mean(distinct(rounded).into_iter().map(|((runs, hinted), count)| (self.step(runs, &hinted), count)))
    }

    /// The run of `steps`' terms (`T_s`, service, bytes, link), each with
    /// its count: their means, and `T` the slowest of the schedule's, the
    /// drive's sustained rate's and the link's.
    fn mean(&self, steps: impl Iterator<Item = ((f64, f64, f64, f64), usize)>) -> StepTime {
        let (mut count, mut scheduled, mut served, mut bytes, mut link) = (0f64, 0., 0., 0., 0.);
        for ((step, service, read, moved), times) in steps {
            let times = times as f64;
            count += times;
            scheduled += step * times;
            served += service * times;
            bytes += read * times;
            link += moved * times;
        }
        let (scheduled, served, bytes, link) = (scheduled / count, served / count, bytes / count, link / count);
        let sustained = bytes * self.sustained;
        StepTime {
            compute: self.compute,
            scheduled,
            sustained,
            served,
            link,
            bytes,
            total: scheduled.max(sustained).max(link),
        }
    }
}

/// Zero-allocation hash/equality key over a step's `&[LayerRun]` when nothing
/// reads ahead (bit-exact match to the `[u64; 7]` key with `hinted = 0`).
#[derive(Clone, Copy)]
struct RunKey<'a>(&'a [LayerRun]);

impl PartialEq for RunKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self.0.iter().zip(other.0).all(|(a, b)| {
                a.layers == b.layers
                    && a.load.tier == b.load.tier
                    && a.load.file == b.load.file
                    && a.load.routed.to_bits() == b.load.routed.to_bits()
                    && a.load.rows.to_bits() == b.load.rows.to_bits()
                    && a.load.recall.to_bits() == b.load.recall.to_bits()
            })
    }
}

impl Eq for RunKey<'_> {}

impl std::hash::Hash for RunKey<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for run in self.0 {
            run.load.routed.to_bits().hash(state);
            run.load.rows.to_bits().hash(state);
            run.load.tier.hash(state);
            run.load.file.hash(state);
            run.load.recall.to_bits().hash(state);
            run.layers.hash(state);
        }
    }
}

/// `items`' distinct ones by their key, each with its counts summed, in the
/// order first seen: what a run prices each of once.
fn distinct<K: std::hash::Hash + Eq, T>(items: impl IntoIterator<Item = (K, T, usize)>) -> Vec<(T, usize)> {
    let mut seen = HashMap::<K, usize>::new();
    let mut distinct = Vec::<(T, usize)>::new();
    for (key, item, count) in items {
        match seen.get(&key) {
            Some(&at) => distinct[at].1 += count,
            None => {
                seen.insert(key, distinct.len());
                distinct.push((item, count));
            }
        }
    }
    distinct
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_lines::{GpuLines, Hits, HostLines, LandingLine, Miss, Moves, Nanoseconds, Pass, PassLine, Site};
    use crate::lines::{FileLine, RecordLine};

    /// A record alone in 1 ms at 2 GB/s: 2 MB records, `λ` 0.
    const READ: ReadLine = match ReadLine::new(0., 1. / 2e9) {
        Ok(line) => line,
        Err(_) => panic!("the tests' read line"),
    };

    /// A unified layer: no CPU and no host tier; hits and in-place compute
    /// of nothing.
    fn unified() -> ExpertLines {
        let none = PassLine { weights: Nanoseconds::try_from(0).unwrap(), row: Nanoseconds::try_from(0).unwrap() };
        ExpertLines { host: HostLines { cpu: None, memory: None, landing: None, record_bytes: 2_000_000 }, gpu: GpuLines { link: none, resident: none, moves: Moves::NONE } }
    }

    fn landed_host() -> FileLine {
        FileLine { landing: RecordLine::new(0.2e-3, 0.1e-3).unwrap() }
    }

    fn landed_line(landing: LandingLine) -> ExpertLines {
        let mut landed = unified();
        landed.host.landing = Some(landing);
        landed
    }

    fn load(file: u64, recall: f64) -> LayerLoad {
        LayerLoad { routed: 8., rows: 1., file, recall, ..LayerLoad::default() }
    }

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} ≠ {expected}");
    }

    /// Four layers of 1 ms compute and no readout.
    fn inputs<'a>(lines: &'a [ExpertLines], lead: Lead, overlap: f64) -> StepInputs<'a> {
        StepInputs { compute: 4e-3, head: 0., overlap, read: READ, sustained: 1. / 2e9, ahead: ReadAhead { precision: 1., lead }, lines }
    }

    fn one(inputs: &StepInputs<'_>, loads: &[LayerLoad]) -> (f64, f64, f64, f64) {
        inputs.one(loads, &mut vec![0.; loads.len()])
    }

    /// `loads` as runs of one layer each.
    fn runs(loads: &[LayerLoad]) -> Vec<LayerRun> {
        loads.iter().map(|&load| LayerRun { load, layers: 1 }).collect()
    }

    /// A layer's barrier waits for its misses whole, then computes: its
    /// reads overlap only earlier layers' compute. Under a token's lead every
    /// read is issued at the step's start, so only what the device cannot
    /// finish by each layer's turn is exposed; a run is never faster than its
    /// bytes at the sustained rate.
    #[test]
    fn a_layers_barrier_adds_what_its_reads_have_not_finished() {
        let lines = [unified(); 4];
        let demand = inputs(&lines, Lead::Layers(0.), 1.);
        // Layer 1's route comes at 1 ms, its 2 records take 2 ms, then its
        // compute and the two layers after.
        let two = [load(0, 0.), load(2, 0.), load(0, 0.), load(0, 0.)];
        close(one(&demand, &two).0, 1e-3 + 2e-3 + 3e-3);
        // Under a token's lead its reads end at 2 ms, 1 ms past its route.
        let token = inputs(&lines, Lead::Token, 1.);
        close(one(&token, &[load(0, 1.), load(2, 1.), load(0, 1.), load(0, 1.)]).0, 5e-3);
        // Eight records at layer 0: 8 ms before it computes.
        close(one(&token, &[load(8, 1.), load(0, 1.), load(0, 1.), load(0, 1.)]).0, 8e-3 + 4e-3);
        // At η = 0.5 compute runs at half rate until the reads end at 4 ms:
        // two layers by then, two more after.
        close(one(&inputs(&lines, Lead::Token, 0.5), &[load(0, 1.), load(0, 1.), load(0, 1.), load(4, 1.)]).0, 6e-3);
        // The sustained bound: 40 records a step at 2 GB/s.
        let flood = runs(&[load(10, 1.); 4]);
        let run = token.run([(&flood[..], 1)]);
        close(run.sustained, 40e-3);
        assert!(run.total >= 40e-3);
    }

    /// Beside a host tier the link is a term of its own: moves holding it
    /// 10 ms a layer run on the copy engine beside four layers' 4 ms of
    /// compute, so the run takes their 40 ms, not the schedule's; a miss
    /// read in place waits behind them, so the CPU takes it.
    #[test]
    fn the_links_moves_are_a_term_of_the_run() {
        let lines = [ExpertLines::reference(2_000_000).moving(Some((10e-3, 0.))); 4];
        let token = inputs(&lines, Lead::Token, 1.);
        let quiet = runs(&[load(0, 1.); 4]);
        let run = token.run([(&quiet[..], 1)]);
        close(run.scheduled, 4e-3);
        close(run.link, 40e-3);
        close(run.total, 40e-3);
        let tier = LayerLoad { tier: 1, ..load(0, 1.) };
        let served = ExpertLines::reference(2_000_000).moving(Some((10e-3, 0.))).serve(Pass::Decode, &[Hits { experts: 7., rows: 1. }], &[Miss { rows: 1., ready_seconds: 0. }]);
        assert_eq!(served.sites, [Site::Cpu]);
        close(token.run([(&runs(&[tier; 4])[..], 1)]).link, 40e-3);
    }

    /// Without hints a run's terms are its distinct steps', each priced
    /// once and weighed by its count, whatever their order.
    #[test]
    fn an_unhinted_run_prices_each_distinct_step_once() {
        let lines = [unified(); 4];
        let demand = inputs(&lines, Lead::Layers(0.), 1.);
        let quiet = runs(&[load(0, 0.); 4]);
        let missing = runs(&[load(0, 0.), load(2, 0.), load(0, 0.), load(1, 0.)]);
        let run = demand.run([&quiet[..], &missing[..], &quiet[..], &quiet[..], &missing[..]].map(|loads| (loads, 1)));
        let counted = demand.run([(&missing[..], 2), (&quiet[..], 3)]);
        for (actual, expected) in [(counted.scheduled, run.scheduled), (counted.sustained, run.sustained), (counted.bytes, run.bytes), (counted.total, run.total)] {
            close(actual, expected);
        }
    }

    /// A hinted step's repetitions are its rounding's distinct steps, each
    /// priced once and weighed by its count: a counted run is the same run
    /// stepped one by one.
    #[test]
    fn a_counted_hinted_run_is_its_steps_one_by_one() {
        let lines = [unified(); 4];
        let hinted = inputs(&lines, Lead::Layers(1.), 1.);
        let step = [load(0, 0.5), load(3, 0.5), load(1, 0.5), load(2, 0.5)];
        let laid = runs(&step);
        let counted = hinted.run([(&laid[..], 5)]);
        let stepped = hinted.run([(&laid[..], 1); 5]);
        let mut carried = [0.; 4];
        let each = (0..5).map(|_| hinted.one(&step, &mut carried).0).sum::<f64>() / 5.;
        for (actual, expected) in [(counted.scheduled, stepped.scheduled), (counted.bytes, stepped.bytes), (counted.scheduled, each)] {
            close(actual, expected);
        }
    }

    /// A miss's records are whole only once they land, which its layer's
    /// computes wait for too: on the host, a request's every landing past
    /// its reads' end; on the device, whose landing launch converts each
    /// record as its read lands, the last record's convert alone, however
    /// many the request reads.
    #[test]
    fn a_layer_waits_for_its_landing() {
        let lines = [landed_line(LandingLine::Host(landed_host())); 4];
        let demand = inputs(&lines, Lead::Layers(0.), 1.);
        close(one(&demand, &[load(0, 0.), load(2, 0.), load(0, 0.), load(0, 0.)]).0, 1e-3 + 2e-3 + 0.4e-3 + 3e-3);
        let device = [landed_line(LandingLine::Device(RecordLine::new(0.05e-3, 0.03e-3).unwrap())); 4];
        let demand = inputs(&device, Lead::Layers(0.), 1.);
        for misses in [1, 2, 6] {
            close(one(&demand, &[load(0, 0.), load(misses, 0.), load(0, 0.), load(0, 0.)]).0, 1e-3 + misses as f64 * 1e-3 + 0.08e-3 + 3e-3);
        }
        // Conversion slower than the reads (2 ms a record against their 1 ms)
        // is what holds the layer: six records convert 12 ms from the first
        // landing, 7 ms past the last read (at 6 ms, 5 ms after the first).
        let slow = [landed_line(LandingLine::Device(RecordLine::new(0., 2e-3).unwrap())); 4];
        let demand = inputs(&slow, Lead::Layers(0.), 1.);
        close(one(&demand, &[load(0, 0.), load(6, 0.), load(0, 0.), load(0, 0.)]).0, 1e-3 + 6e-3 + 7e-3 + 3e-3);
    }

    /// A hint `ℓ` layers ahead is issued at the route `ℓ` before and hides
    /// its miss once it lands by that layer's route; the wrong reads of an
    /// imprecise hint queue with it.
    #[test]
    fn hints_read_ahead_of_their_layer() {
        let lines = [unified(); 4];
        let ahead = inputs(&lines, Lead::Layers(1.), 1.);
        // Layer 1's record, hinted at layer 0's route (0), lands at 1 ms, at
        // layer 1's route.
        close(one(&ahead, &[load(0, 1.), load(1, 1.), load(0, 1.), load(0, 1.)]).0, 4e-3);
        // Three records land at 3 ms: 2 ms past that route.
        close(one(&ahead, &[load(0, 1.), load(3, 1.), load(0, 1.), load(0, 1.)]).0, 6e-3);
        let wrong = StepInputs { ahead: ReadAhead { precision: 0.5, lead: Lead::Layers(1.) }, ..ahead };
        let (step, _, bytes, _) = one(&wrong, &[load(0, 1.), load(1, 1.), load(0, 1.), load(0, 1.)]);
        // Its wrong read queues behind its right one and hides nothing.
        close(bytes, 2. * 2e6);
        close(step, 4e-3);
    }

    /// Whole hinted reads carry their rounding across a run's steps, so the
    /// run's hinted share is its recall: a lone miss hinted at 0.5 is
    /// read ahead every other step.
    #[test]
    fn a_run_hints_its_recall_in_whole_records() {
        let lines = [unified(); 2];
        let ahead = inputs(&lines, Lead::Layers(1.), 1.);
        let half = [load(0, 0.), load(1, 0.5)];
        let (mut carried, mut first) = (vec![0.; 2], Vec::new());
        for _ in 0..4 {
            first.push(ahead.one(&half, &mut carried).0);
        }
        // Each layer computes 2 ms. On demand layer 1 waits 1 ms at its
        // route; hinted at layer 0's, its read has landed by then.
        close(first[0], 5e-3);
        close(first[1], 4e-3);
        close(first[2], 5e-3);
        close(first[3], 4e-3);
    }

    /// A cell's step from its own inputs, within 10%: a resident step, its
    /// read line and landing line, and its misses, recall, lead and
    /// precision, as a GB10 under Vulkan and an M4 Pro recorded them. The
    /// first waits on its layers' barriers; the second is held by its
    /// bytes as much as by them.
    #[test]
    fn a_cells_own_inputs_reproduce_its_step() {
        struct Cell {
            compute: f64,
            head: f64,
            latency: f64,
            rate: f64,
            sustained: f64,
            landing: (f64, f64),
            misses: f64,
            recall: f64,
            precision: f64,
            lead: f64,
            measured: f64,
        }
        let cells = [
            Cell { compute: 21.48e-3, head: 2.24e-3, latency: 0.453e-3, rate: 9.0e9, sustained: 10.86e9, landing: (0.271e-3, 0.126e-3), misses: 81.5, recall: 0.46, precision: 0.54, lead: 1.51, measured: 61.8e-3 },
            Cell { compute: 13.54e-3, head: 0.87e-3, latency: 0.284e-3, rate: 2.57e9, sustained: 2.78e9, landing: (0.052e-3, 0.052e-3), misses: 83.7, recall: 0.41, precision: 0.62, lead: 1.42, measured: 77.0e-3 },
        ];
        for cell in cells {
            let mut line = unified();
            line.host.record_bytes = 1_900_544;
            line.host.landing = Some(LandingLine::Host(FileLine { landing: RecordLine::new(cell.landing.0, cell.landing.1).unwrap() }));
            let lines = [line; 40];
            let step = StepInputs {
                compute: cell.compute,
                head: cell.head,
                overlap: 1.,
                read: ReadLine::new(cell.latency, 1. / cell.rate).unwrap(),
                sustained: 1. / cell.sustained,
                ahead: ReadAhead { precision: cell.precision, lead: Lead::Layers(cell.lead) },
                lines: &lines,
            };
            // A layer's misses a step, as even as integers allow.
            let steps = (0..400u64)
                .map(|step| {
                    (0..40u64)
                        .map(|layer| {
                            let at = step + 7 * layer;
                            let share = cell.misses / 40.;
                            load(((at + 1) as f64 * share).floor() as u64 - (at as f64 * share).floor() as u64, cell.recall)
                        })
                        .collect::<Vec<_>>()
                })
                .map(|loads| runs(&loads))
                .collect::<Vec<_>>();
            let total = step.run(steps.iter().map(|runs| (runs.as_slice(), 1))).total;
            assert!((total / cell.measured - 1.).abs() < 0.1, "{:.1} ms ≠ {:.1} ms", total * 1e3, cell.measured * 1e3);
        }
    }
}
