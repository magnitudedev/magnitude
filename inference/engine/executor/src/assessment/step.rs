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
//! bounding the CPU and the link together). `τ_l` is its share of the
//! resident step `(C − h)/L`, and the step ends at `T_s = f_{L−1} ⊕ h`, `h`
//! the readout's part of `C`.
//!
//! **A run** of steps takes `T = max(mean T_s, mean B_s / S)`: no burst
//! beats the device's sustained rate `S` over a long run, `B_s` the step's
//! read bytes, wrong hints' among them.

use super::cost::{ExpertLines, Hits, Miss, Pass};
use super::streaming_costs::ReadLine;

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
    /// Read bytes a step.
    pub bytes: f64,
    pub total: f64,
}

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

    /// What layer `layer`'s misses served on the CPU or over the link show
    /// past its hits, its file misses whole `ready` after the layer begins
    /// (none without a host tier: the GPU reads them in place).
    fn service(&self, layer: usize, load: LayerLoad, ready: f64) -> f64 {
        let lines = &self.lines[layer];
        let Some(memory) = lines.host.memory else {
            return 0.;
        };
        let misses = std::iter::repeat_n(0., load.tier as usize)
            .chain(std::iter::repeat_n(ready, load.file as usize))
            .map(|ready_seconds| Miss { rows: load.rows, ready_seconds })
            .collect::<Vec<_>>();
        let hits = Hits { experts: (load.routed - misses.len() as f64).max(0.), rows: load.rows };
        let served = lines.serve(Pass::Decode, &[hits], &misses);
        // Host memory bounds the CPU and the link together: every miss
        // leaves it at `t_ram`.
        let leave = (load.tier + load.file) as f64 * memory.get() * 1e-9;
        let hit = hits.experts * lines.gpu.resident.nanoseconds(load.rows) * 1e-9;
        (served.end_seconds.max(leave) - hit).max(0.)
    }

    /// One step of `loads` (each streamed layer's): `T_s`, the CPU and link
    /// service in it, and its read bytes. A share of a layer's misses reads
    /// ahead in whole records: `carried` holds what each layer's rounding
    /// left, for its next step, so a run's share is its recall.
    pub fn one(&self, loads: &[LayerLoad], carried: &mut [f64]) -> (f64, f64, f64) {
        let layers = self.lines.len();
        let tau = (self.compute - self.head) / layers as f64;
        let byte = self.read.seconds_per_byte();
        let latency = self.read.latency_seconds();
        // Each layer's hinted reads, its wrong hints', and its demand reads.
        let reads = loads
            .iter()
            .zip(carried.iter_mut())
            .enumerate()
            .map(|(layer, (load, carried))| {
                let bytes = self.lines[layer].host.record_bytes as f64;
                let due = *carried + load.recall * load.file as f64;
                let hinted = (due.floor() as u64).min(load.file);
                *carried = due - hinted as f64;
                let wrong = hinted as f64 * (1. / self.ahead.precision - 1.);
                (bytes, hinted, wrong, load.file - hinted)
            })
            .collect::<Vec<_>>();
        let bytes = loads.iter().zip(&reads).map(|(load, &(bytes, _, wrong, _))| bytes * (load.file as f64 + wrong)).sum::<f64>();
        // One device, reads completing in the order issued.
        let mut free = 0f64;
        let mut read = |at: f64, records: f64, bytes: f64| -> f64 {
            if records > 0. {
                free = (at + latency + bytes * byte).max(free + bytes * byte) + (records - 1.) * bytes * byte;
            }
            free
        };
        // What `count` misses of `layer` add past their reads' end.
        let landing = |layer: usize, count: u64| self.lines[layer].landed(count, reads[layer].0 * byte);
        let mut ready = vec![0f64; layers];
        // Under a token's lead every read is issued at the step's start, in
        // layer order, wrong hints after each layer's own.
        let lead = match self.ahead.lead {
            Lead::Token => {
                for (layer, &(bytes, hinted, wrong, demand)) in reads.iter().enumerate() {
                    if hinted + demand > 0 {
                        ready[layer] = read(0., (hinted + demand) as f64, bytes) + landing(layer, hinted + demand);
                    }
                    read(0., wrong, bytes);
                }
                None
            }
            Lead::Layers(lead) => Some(lead.round() as usize),
        };
        let mut busy = read(0., 0., 0.);
        let (mut end, mut served_sum) = (0f64, 0f64);
        for layer in 0..layers {
            // The layer's route: its demand reads, then the hints it names
            // of the layers ahead, their wrong reads after them.
            let route = end;
            if let Some(lead) = lead {
                let (bytes, _, _, demand) = reads[layer];
                if demand > 0 {
                    ready[layer] = ready[layer].max(read(route, demand as f64, bytes) + landing(layer, demand));
                }
                if let Some(&(bytes, hinted, wrong, _)) = reads.get(layer + lead) {
                    if hinted > 0 {
                        ready[layer + lead] = ready[layer + lead].max(read(route, hinted as f64, bytes) + landing(layer + lead, hinted));
                    }
                    read(route, wrong, bytes);
                }
                busy = read(route, 0., 0.);
            }
            let service = self.service(layer, loads[layer], (ready[layer] - route).max(0.));
            served_sum += service;
            end = match self.lines[layer].host.memory {
                // A host tier's file misses land before the CPU or the link
                // serves them, within the layer's service.
                Some(_) => self.advance(route, tau, busy) + service,
                None => self.advance(route.max(ready[layer]), tau, busy),
            };
        }
        (self.advance(end, self.head, busy), served_sum, bytes)
    }

    /// A run of `steps`, each its layers' loads: `T`, its schedule's and the
    /// sustained rate's means, the slower.
    pub fn run<'s>(&self, steps: impl IntoIterator<Item = &'s [LayerLoad]>) -> StepTime {
        let mut carried = vec![0.; self.lines.len()];
        self.mean(steps.into_iter().map(|loads| (self.one(loads, &mut carried), 1)))
    }

    /// What [`Self::run`] gives a run holding each of `steps` its count of
    /// times, in any order, none reading ahead, each priced once: with no
    /// hint, no rounding carries from step to step, so a step's terms are
    /// its loads' alone and the run's means weigh each by its count.
    pub fn unhinted<'s>(&self, steps: impl IntoIterator<Item = (&'s [LayerLoad], usize)>) -> StepTime {
        let mut carried = vec![0.; self.lines.len()];
        self.mean(steps.into_iter().map(|(loads, count)| {
            assert!(loads.iter().all(|load| load.recall == 0.), "an unhinted step reads ahead");
            (self.one(loads, &mut carried), count)
        }))
    }

    /// The run of `steps`' terms (`T_s`, service, bytes), each with its
    /// count: their means, and `T` the slower of the schedule's and the
    /// sustained rate's.
    fn mean(&self, steps: impl Iterator<Item = ((f64, f64, f64), usize)>) -> StepTime {
        let (mut count, mut scheduled, mut served, mut bytes) = (0f64, 0., 0., 0.);
        for ((step, service, read), times) in steps {
            let times = times as f64;
            count += times;
            scheduled += step * times;
            served += service * times;
            bytes += read * times;
        }
        let (scheduled, served, bytes) = (scheduled / count, served / count, bytes / count);
        let sustained = bytes * self.sustained;
        StepTime {
            compute: self.compute,
            scheduled,
            sustained,
            served,
            bytes,
            total: scheduled.max(sustained),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assessment::cost::{GpuLines, HostLines, LandingLine, Nanoseconds, PassLine};
    use crate::assessment::streaming_costs::{FileLine, RecordLine};

    /// A record alone in 1 ms at 2 GB/s: 2 MB records, `λ` 0.
    const READ: ReadLine = match ReadLine::new(0., 1. / 2e9) {
        Ok(line) => line,
        Err(_) => panic!("the tests' read line"),
    };

    /// A unified layer: no CPU and no host tier; hits and in-place compute
    /// of nothing.
    fn unified() -> ExpertLines {
        let none = PassLine { weights: Nanoseconds::try_from(0).unwrap(), row: Nanoseconds::try_from(0).unwrap() };
        ExpertLines { host: HostLines { cpu: None, memory: None, landing: None, record_bytes: 2_000_000 }, gpu: GpuLines { link: none, resident: none } }
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

    fn one(inputs: &StepInputs<'_>, loads: &[LayerLoad]) -> (f64, f64, f64) {
        inputs.one(loads, &mut vec![0.; loads.len()])
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
        let flood = [load(10, 1.); 4];
        let run = token.run([&flood[..]]);
        close(run.sustained, 40e-3);
        assert!(run.total >= 40e-3);
    }

    /// Without hints a run's terms are its distinct steps', each priced
    /// once and weighed by its count, whatever their order.
    #[test]
    fn an_unhinted_run_prices_each_distinct_step_once() {
        let lines = [unified(); 4];
        let demand = inputs(&lines, Lead::Layers(0.), 1.);
        let quiet = [load(0, 0.); 4];
        let missing = [load(0, 0.), load(2, 0.), load(0, 0.), load(1, 0.)];
        let run = demand.run([&quiet[..], &missing[..], &quiet[..], &quiet[..], &missing[..]]);
        let counted = demand.unhinted([(&missing[..], 2), (&quiet[..], 3)]);
        for (actual, expected) in [(counted.scheduled, run.scheduled), (counted.sustained, run.sustained), (counted.bytes, run.bytes), (counted.total, run.total)] {
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
        let (step, _, bytes) = one(&wrong, &[load(0, 1.), load(1, 1.), load(0, 1.), load(0, 1.)]);
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
                .collect::<Vec<_>>();
            let total = step.run(steps.iter().map(Vec::as_slice)).total;
            assert!((total / cell.measured - 1.).abs() < 0.1, "{:.1} ms ≠ {:.1} ms", total * 1e3, cell.measured * 1e3);
        }
    }
}
