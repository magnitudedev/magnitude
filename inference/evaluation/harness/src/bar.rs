//! `bar`: the ceiling of a streamed run's steps on the machine it ran on: the
//! best any implementation could reach, from the machine's measured rates, the
//! model and the run's own routing, never from our design's numbers.
//!
//! **Decode** ([`Ceiling`]). Each input's status is stated, because a term
//! that is only our design's would hide its flaw:
//!
//! - **Misses:** Bélády's over the run's routing at the most records the
//!   cap can hold ([`records_at_cap`]): the cap less the weights every step
//!   touches, in records of their stored bytes, with no staging or layout.
//!   No online limit on eviction is proven, so none is taken.
//! - **Reads:** every miss read ahead of its layer, so reads only ever bound
//!   the step by their bytes: at the device's sustained rate on unified
//!   memory, over the link's ceiling beside a host tier. No online limit on
//!   prediction is proven either. A route is known with certainty only once
//!   its gate has run, but a read needs only a prediction.
//! - **Compute:** `C`, the resident step. It is our kernels' number, not a
//!   hardware rate: a ceiling on `C` itself is not derived. `η`, measured
//!   under reads on unified memory, can only slow it.
//! - **Convert:** none: a ceiling takes the record's kernels reading its
//!   stored form, or a convert that overlaps compute.
//!
//! The online frontier and `reachable` are diagnostics, priced by the step
//! model ([`magnitude_executor::assessment::step`]) at the bank's own
//! ranking's misses: they split a gap between ranking, prediction and
//! implementation, and neither is a bar. They read at the device's own line
//! ([`crate::machine::StorageMachine`]), nothing of our read path, one
//! request a layer.
//!
//! **Prefill, every backend** ([`PrefillBar`]). A streamed prompt's prompt
//! groups, each cold expert moved once, against the slowest of the three
//! resources they share, each priced from the run's own measurements:
//!
//! - **storage:** the groups' cold stored bytes at what the storage device
//!   holds over a read that long ([`ReadRate`]);
//! - **device:** the resident prefill at the same context and concurrency,
//!   plus the cold records' convert at the convert's ceiling (`ConvertCosts`:
//!   the device's own memory to its own memory), both on the one device;
//! - **link** (a discrete GPU's host tier): every byte that crosses it, the
//!   tier's records pulled and the file's records converted from the host,
//!   at the link's ceiling (the placement matrix's copy peak from pinned
//!   host memory into device memory).
//!
//! It assumes the three overlap whole, across layers and groups. No term is
//! priced from the run's own convert or copy, whose flaws a bar would then
//! hide: the harness reports those beside it.

use magnitude_executor::assessment::{CpuLines, ExpertLines, LayerLoad, LineError, Nanoseconds, RowCost, RowLine, StepInputs, StepTime};

/// One streamed layer of one decode step as the bar prices it: the experts
/// its route chose and their choices, the replay's misses, and the share of
/// them predicted (hinted ahead of their read on unified memory).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OracleLayer {
    pub routed: u64,
    pub choices: u64,
    pub misses: u64,
    pub predictable: f64,
}

/// `lines` with the CPU at machine peak: an expert's read `t_ram`
/// (`HostLines::memory`) bounded by `arithmetic`, its FLOPs at `Φ`; a row
/// more, its arithmetic; the load's handshake; and its record read in place
/// over the link in `link`, the link's ceiling. Lines without a host tier
/// are unchanged.
pub fn peak_lines(lines: &ExpertLines, arithmetic: f64, handshake: Nanoseconds, link: Nanoseconds) -> Result<ExpertLines, LineError> {
    let Some(memory) = lines.host.memory else {
        return Ok(*lines);
    };
    let row = Nanoseconds::of(arithmetic)?;
    let expert = RowLine { one: Nanoseconds::of((memory.get() * 1e-9).max(arithmetic))?, row: RowCost::Measured(row) };
    let mut peak = *lines;
    peak.host.cpu = Some(CpuLines { expert, handshake });
    peak.gpu.link.weights = link;
    Ok(peak)
}

/// The run of `steps` under `inputs` ([`StepInputs::run`]), each layer's
/// misses where its lines say: from the host tier beside a tier; from the
/// file without one, hinted as predicted.
pub fn step(inputs: &StepInputs<'_>, steps: &[Vec<OracleLayer>]) -> StepTime {
    let loads = steps
        .iter()
        .map(|step| {
            step.iter()
                .enumerate()
                .map(|(layer, oracle)| {
                    let (routed, rows) = (oracle.routed as f64, oracle.choices as f64 / oracle.routed as f64);
                    match inputs.lines[layer].host.memory {
                        Some(_) => LayerLoad { routed, rows, tier: oracle.misses, ..LayerLoad::default() },
                        None => LayerLoad { routed, rows, file: oracle.misses, recall: oracle.predictable, ..LayerLoad::default() },
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    inputs.run(loads.iter().map(Vec::as_slice))
}

/// The most records a cap can hold: `records` of each group, each of
/// `record_bytes` stored, scaled alike to fill the `free` bytes left beside
/// the weights every step touches. Never fewer than the bank holds.
pub fn records_at_cap(records: &[usize], record_bytes: &[u64], free: u64) -> Vec<usize> {
    let held = records.iter().zip(record_bytes).map(|(&records, &bytes)| records as u64 * bytes).sum::<u64>();
    let scale = (free as f64 / held as f64).max(1.);
    records.iter().map(|&records| (records as f64 * scale) as usize).collect()
}

/// The decode ceiling's two terms, a step's seconds (module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ceiling {
    /// `C`, slowed by `η` for as long as the reads run.
    pub compute: f64,
    /// The step's missed bytes through what they cross.
    pub moved: f64,
}

impl Ceiling {
    /// A run at the resident step `compute`, each step's misses of each
    /// layer in `misses` (`record_bytes` the layers', a step their count),
    /// a byte taking `byte` seconds, compute retained at `overlap` while
    /// they move.
    pub fn of(compute: f64, overlap: f64, byte: f64, record_bytes: &[u64], misses: &[u64]) -> Self {
        let steps = misses.len() / record_bytes.len();
        let bytes = misses.chunks(record_bytes.len()).flat_map(|step| step.iter().zip(record_bytes)).map(|(&misses, &bytes)| misses as f64 * bytes as f64).sum::<f64>();
        let moved = bytes / steps as f64 * byte;
        // Compute finishes inside the reads, or what is left runs after them.
        let compute = if compute <= overlap * moved { compute / overlap } else { moved + compute - overlap * moved };
        Self { compute, moved }
    }

    /// The step: its compute or its bytes, the slower.
    pub fn seconds(&self) -> f64 {
        self.compute.max(self.moved)
    }
}

/// A prompt group layer's cold experts, summed over the groups: those
/// conveyed (read from the file and converted) and their stored bytes, and
/// those pulled from the host tier and their resident bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PrefillLayer {
    pub conveyed: u64,
    pub cold_bytes: u64,
    pub pulled: u64,
    pub tier_bytes: u64,
}

/// What a storage device reads over a long read, a token bucket: `peak`
/// until its `burst` bytes beyond `sustained` have drained, then `sustained`,
/// each in bytes a second.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadRate {
    pub peak: f64,
    pub sustained: f64,
    pub burst: f64,
}

impl ReadRate {
    /// `bytes`' seconds, whichever of the peak and the bucket binds.
    pub fn seconds(&self, bytes: f64) -> f64 {
        (bytes / self.peak).max((bytes - self.burst) / self.sustained.min(self.peak))
    }
}

/// What a streamed prompt's prefill bar is priced from, or why an input is
/// unmeasured: its streamed layers' cold experts, the storage device's
/// read rate, the resident prefill's seconds, each
/// layer's convert ceiling (a record's seconds) and, where the bank has a
/// host tier, the link's ceiling (a byte's seconds; none on unified
/// memory).
#[derive(Clone, Debug, PartialEq)]
pub struct PrefillInputs {
    pub layers: Vec<PrefillLayer>,
    pub read_rate: Result<ReadRate, String>,
    pub resident_seconds: Result<f64, String>,
    pub convert: Result<Vec<f64>, String>,
    pub link: Option<Result<f64, String>>,
}

/// A streamed prompt's prefill bar (module doc): its storage, device and
/// link terms, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefillBar {
    pub read: f64,
    pub resident: f64,
    pub convert: f64,
    pub link: f64,
}

impl PrefillBar {
    /// The bar's terms from `inputs`: unavailable, with why, without an
    /// input a term needs (the read rate only where the file is read, the
    /// convert's ceiling only where a record is conveyed).
    pub fn of(inputs: &PrefillInputs) -> Result<Self, String> {
        let total = |field: fn(&PrefillLayer) -> u64| inputs.layers.iter().map(field).sum::<u64>();
        let read = match total(|layer| layer.cold_bytes) {
            0 => 0.,
            bytes => inputs.read_rate.clone()?.seconds(bytes as f64),
        };
        let convert = match total(|layer| layer.conveyed) {
            0 => 0.,
            _ => {
                let ceilings = inputs.convert.clone().map_err(|reason| format!("the convert's ceiling: {reason}"))?;
                inputs.layers.iter().zip(&ceilings).map(|(layer, record)| layer.conveyed as f64 * record).sum()
            }
        };
        let link = match &inputs.link {
            None => 0.,
            Some(link) => {
                let byte = link.clone().map_err(|reason| format!("the link's ceiling: {reason}"))?;
                (total(|layer| layer.tier_bytes) + total(|layer| layer.cold_bytes)) as f64 * byte
            }
        };
        Ok(Self { read, resident: inputs.resident_seconds.clone()?, convert, link })
    }

    /// `max(storage, device, link)`.
    pub fn seconds(&self) -> f64 {
        self.read.max(self.resident + self.convert).max(self.link)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_executor::{EvictionRecency, RoutedRead, replay_routing};
    use std::collections::{BTreeSet, HashMap};

    use magnitude_executor::assessment::{Lead, ReadAhead, ReadLine};

    /// An M4 Pro volume's lines: a record alone in 0.971 ms, 3.172 GB/s
    /// peak, 2.510 GB/s sustained.
    const PEAK: f64 = 3.172e9;
    const READ: ReadLine = match ReadLine::new(0.971e-3 - 1_900_544. / PEAK, 1. / PEAK) {
        Ok(line) => line,
        Err(_) => panic!("the tests' read line"),
    };

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-9, "{actual} ≠ {expected}");
    }

    fn layer(routed: u64, misses: u64, predictable: f64) -> OracleLayer {
        OracleLayer { routed, choices: routed, misses, predictable }
    }

    /// The reference RTX 3090 host's at machine peak: 38 µs a record from RAM, 1 µs of
    /// arithmetic a row, a 4 µs handshake.
    fn discrete() -> ExpertLines {
        let reference = ExpertLines::reference(ExpertLines::REFERENCE_RECORD);
        let handshake = Nanoseconds::try_from(4_000).unwrap();
        peak_lines(&reference, 1e-6, handshake, reference.gpu.link.weights).unwrap()
    }

    fn inputs<'a>(compute: f64, lines: &'a [ExpertLines]) -> StepInputs<'a> {
        StepInputs { compute, head: 0., overlap: 1., read: READ, sustained: 1. / 2.51e9, ahead: ReadAhead { precision: 1., lead: Lead::Token }, lines }
    }

    /// One unpredicted miss beside 7 hits (35 µs) shows its read's 38 µs and
    /// handshake past them; 30 hits hide them whole.
    #[test]
    fn the_discrete_bar_exposes_only_what_the_hits_cannot_hide() {
        let lines = [discrete()];
        let cpu = lines[0].host.cpu.unwrap();
        let (read, handshake, hit) = (cpu.expert.nanoseconds(1.) * 1e-9, cpu.handshake.get() * 1e-9, lines[0].gpu.resident.nanoseconds(1.) * 1e-9);
        let at = |one: OracleLayer| step(&inputs(0.008, &lines), &[vec![one]]).total;
        close(at(layer(8, 1, 0.)), 0.008 + read + handshake - 7. * hit);
        close(at(layer(8, 0, 0.)), 0.008);
        close(at(layer(31, 1, 0.)), 0.008);
    }

    /// Host memory bounds the CPU and the link: 30 misses leave RAM at
    /// 38 µs each however they split.
    #[test]
    fn host_memory_bounds_the_service() {
        let mut slow = discrete();
        slow.gpu.link.weights = Nanoseconds::try_from(1_000).unwrap();
        let lines = [slow];
        close(step(&inputs(0.008, &lines), &[vec![layer(30, 30, 0.)]]).served, 30. * lines[0].host.memory.unwrap().get() * 1e-9);
    }

    /// A unified ceiling: every read issued
    /// at the step's start, one device; a step is its schedule or its bytes
    /// at the sustained rate, the slower. 38 misses a step over 40 layers of
    /// 1.9 MB records, on a 13.4 ms step: the sustained rate binds.
    #[test]
    fn a_unified_ceiling_is_its_schedule_or_its_bytes_at_the_sustained_rate() {
        let mut record = ExpertLines::reference(1_900_544);
        record.host.cpu = None;
        record.host.memory = None;
        let lines = vec![record; 40];
        let model = inputs(13.4e-3, &lines);
        let busy = vec![layer(8, 1, 1.); 38].into_iter().chain([layer(8, 0, 1.); 2]).collect::<Vec<_>>();
        let ceiling = step(&model, &[busy]);
        close(ceiling.sustained, 38. * 1_900_544. / 2.51e9);
        close(ceiling.total, ceiling.scheduled.max(ceiling.sustained));
        assert!(ceiling.sustained > ceiling.scheduled, "{ceiling:?}");
        // No misses: the resident step.
        close(step(&model, &[vec![layer(8, 0, 1.); 40]]).total, 13.4e-3);
        // One miss at layer 0 only, which nothing earlier in the token
        // predicts: layer 0 waits for its read to land, then the step's
        // compute.
        let mut first = vec![layer(8, 0, 1.); 40];
        first[0].misses = 1;
        close(step(&model, &[first]).scheduled, 0.971e-3 + 13.4e-3);
    }

    /// The prefill bar's three terms, each from its ceiling: unified memory
    /// (no link) reads and converts its conveyed records; a discrete GPU
    /// moves its tier's records and its conveyed ones over the link.
    #[test]
    fn a_prefill_bar_is_its_slowest_resource_at_its_ceiling() {
        let layer = |conveyed, cold_bytes, pulled, tier_bytes| PrefillLayer { conveyed, cold_bytes, pulled, tier_bytes };
        let inputs = PrefillInputs {
            layers: vec![layer(4, 4_000_000, 0, 0), layer(2, 2_000_000, 0, 0)],
            read_rate: Ok(ReadRate { peak: 1e9, sustained: 1e9, burst: 0. }),
            resident_seconds: Ok(3e-3),
            convert: Ok(vec![1e-4, 2e-4]),
            link: None,
        };
        // Read 6 ms; device 3 + 0.4 + 0.4 ms.
        let unified = PrefillBar::of(&inputs).unwrap();
        close(unified.read, 6e-3);
        close(unified.convert, 8e-4);
        close(unified.seconds(), 6e-3);
        // Over a link of 1 ns a byte, its 6 MB conveyed and 10 MB pulled.
        let discrete = PrefillInputs {
            layers: vec![layer(4, 4_000_000, 5, 5_000_000), layer(2, 2_000_000, 5, 5_000_000)],
            link: Some(Ok(1e-9)),
            ..inputs.clone()
        };
        close(PrefillBar::of(&discrete).unwrap().link, 16e-3);
        close(PrefillBar::of(&discrete).unwrap().seconds(), 16e-3);
    }

    /// Each ceiling is needed only where its resource carries something.
    #[test]
    fn a_prefill_bar_is_unavailable_only_without_an_input_a_term_needs() {
        let pulled = PrefillLayer { pulled: 4, tier_bytes: 4_000, ..PrefillLayer::default() };
        let inputs = PrefillInputs {
            layers: vec![pulled],
            read_rate: Err("no rate".into()),
            resident_seconds: Ok(1.),
            convert: Err("unformed".into()),
            link: Some(Ok(1e-6)),
        };
        close(PrefillBar::of(&inputs).unwrap().seconds(), 1.);
        assert!(PrefillBar::of(&PrefillInputs { link: Some(Err("no queue".into())), ..inputs.clone() }).unwrap_err().contains("no queue"));
        let conveyed = PrefillLayer { conveyed: 1, cold_bytes: 1, ..pulled };
        assert_eq!(PrefillBar::of(&PrefillInputs { layers: vec![conveyed], ..inputs.clone() }), Err("no rate".to_owned()));
        let rated = PrefillInputs { layers: vec![conveyed], read_rate: Ok(ReadRate { peak: 1., sustained: 1., burst: 0. }), ..inputs };
        assert!(PrefillBar::of(&rated).unwrap_err().contains("unformed"));
    }

    /// A read long past its burst takes the sustained rate, a short one the
    /// peak: an M4 Pro volume's 14.6 GB at a 3.17 GB/s peak holding 2.51.
    #[test]
    fn a_long_read_is_priced_at_what_the_device_sustains() {
        let rate = ReadRate { peak: 3.17e9, sustained: 2.51e9, burst: 1e9 };
        close(rate.seconds(14.6e9), 13.6e9 / 2.51e9);
        close(rate.seconds(1e9), 1e9 / 3.17e9);
        // A sustained rate read above the peak is the peak.
        close(ReadRate { peak: 1e9, sustained: 2e9, burst: 0. }.seconds(1e9), 1.);
    }

    /// The cap's free bytes in records of their stored bytes, each group
    /// scaled alike, and never below what the bank holds.
    #[test]
    fn records_at_the_cap_fill_it_in_stored_bytes() {
        assert_eq!(records_at_cap(&[100, 10], &[10, 20], 3_000), vec![250, 25]);
        assert_eq!(records_at_cap(&[100, 10], &[10, 20], 1), vec![100, 10]);
    }

    /// Compute binds until the bytes do, and only a read slows it.
    #[test]
    fn a_ceiling_is_its_compute_or_its_bytes() {
        let run = [4, 0];
        // 2 MB a step in 2 ms, beside 10 ms of compute.
        let fast = Ceiling::of(10e-3, 1., 1e-9, &[1_000_000], &run);
        close(fast.seconds(), 10e-3);
        // At η 0.5, 2 ms of reads cost 1 ms of compute.
        close(Ceiling::of(10e-3, 0.5, 1e-9, &[1_000_000], &run).seconds(), 11e-3);
        // 20 MB a step at 1 ns a byte: the bytes bind, and hide all compute.
        close(Ceiling::of(10e-3, 1., 10e-9, &[1_000_000], &run).seconds(), 20e-3);
    }

    /// The fewest misses any bank of `capacity` records could leave over
    /// `reads` of one group, from `held`, every miss placed: each read's
    /// absent experts placed, any records beyond the capacity evicted from
    /// those the read does not name, every choice of them tried.
    fn fewest(capacity: usize, held: BTreeSet<(usize, usize)>, reads: &[RoutedRead]) -> u64 {
        fn search(
            capacity: usize,
            position: usize,
            held: BTreeSet<(usize, usize)>,
            reads: &[RoutedRead],
            seen: &mut HashMap<(usize, BTreeSet<(usize, usize)>), u64>,
        ) -> u64 {
            let Some(read) = reads.get(position) else { return 0 };
            if let Some(&misses) = seen.get(&(position, held.clone())) {
                return misses;
            }
            let named = read.experts.iter().map(|&expert| (read.layer, expert)).collect::<BTreeSet<_>>();
            let absent = named.difference(&held).count();
            let mut all = held.clone();
            all.extend(&named);
            let over = all.len().saturating_sub(capacity);
            let candidates = held.difference(&named).copied().collect::<Vec<_>>();
            let mut best = u64::MAX;
            // Every `over`-subset of the candidates, by bitmask.
            for mask in 0u32..1 << candidates.len() {
                if mask.count_ones() as usize != over {
                    continue;
                }
                let mut next = all.clone();
                for (index, candidate) in candidates.iter().enumerate() {
                    if mask & 1 << index != 0 {
                        next.remove(candidate);
                    }
                }
                best = best.min(search(capacity, position + 1, next, reads, seen));
            }
            let misses = absent as u64 + best;
            seen.insert((position, held), misses);
            misses
        }
        search(capacity, 0, held, reads, &mut HashMap::new())
    }

    /// The oracle replay leaves exactly the fewest misses any bank could,
    /// group by group, on small traces of two groups (two layers sharing
    /// three records, one layer with two), from a held start.
    #[test]
    fn the_oracle_replay_leaves_the_fewest_misses_any_bank_could() {
        let (records, layers, experts) = ([3, 2], [0, 0, 1], 5);
        let held = vec![vec![0], vec![1], vec![2]];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |bound: usize| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize % bound
        };
        for _ in 0..24 {
            let mut reads = Vec::new();
            for step in 0..10 {
                for layer in 0..layers.len() {
                    let first = next(experts);
                    let second = (first + 1 + next(experts - 1)) % experts;
                    let mut chosen = vec![first, second];
                    chosen.sort_unstable();
                    reads.push(RoutedRead { sequence: step, layer, experts: chosen, choices: 2 });
                }
            }
            let replayed = replay_routing(&records, &layers, experts, &held, &reads, EvictionRecency::NextRead).unwrap();
            for (group, &capacity) in records.iter().enumerate() {
                let of_group = |layer: usize| layers[layer] == group;
                let group_reads = reads.iter().filter(|read| of_group(read.layer)).cloned().collect::<Vec<_>>();
                let start = held
                    .iter()
                    .enumerate()
                    .filter(|&(layer, _)| of_group(layer))
                    .flat_map(|(layer, experts)| experts.iter().map(move |&expert| (layer, expert)))
                    .collect();
                let oracle = reads
                    .iter()
                    .zip(&replayed.misses)
                    .filter(|(read, _)| of_group(read.layer))
                    .map(|(_, misses)| misses)
                    .sum::<u64>();
                assert_eq!(oracle, fewest(capacity, start, &group_reads), "group {group}: {reads:?}");
            }
        }
    }
}
