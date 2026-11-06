//! What a streamed decode step reads and what prices it: each streamed
//! layer's step as the watcher counted it ([`LayerStep`]), its file misses
//! anticipated by hints, a held bank's slots ([`HeldSlots`]), the step the
//! one cost function prices ([`StreamedStep`], [`super::streamed`]) and a
//! stored table's step ([`TableCurve`]).

use super::cost::ExpertLines;
use super::step::{LayerLoad, Lead, ReadAhead, StepInputs};
use super::streaming_costs::ReadLine;
use serde::{Deserialize, Serialize};

/// What one streamed layer of one decode step routed and missed: the
/// experts its route chose and their choices, the misses its host tier
/// held (`hosted`), and those read from the file and landed into the tier
/// (`misses`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LayerStep {
    pub routed: u64,
    pub choices: u64,
    pub hosted: u64,
    pub misses: u64,
    /// Of `misses`, those a hint had staged or begun reading by their
    /// request.
    pub hinted: u64,
    /// The reads this step's hints made for the layer; of them those of an
    /// expert its route chose, which saved a read whether they landed before
    /// the route (a hit) or after (of `hinted`); and their leads, in layers
    /// from the route that hinted to this layer's, summed over the reads.
    pub hint_reads: u64,
    pub used: u64,
    pub lead: u64,
}

impl LayerStep {
    /// What the layer's reads asked of the file and its hints answered: its
    /// misses on demand and every read a hint saved, hit or not, which is
    /// what a bank without read-ahead would have missed.
    pub fn anticipated(&self) -> Anticipated {
        Anticipated { file: self.misses - self.hinted + self.used, hinted: self.used }
    }

    /// What the layer's misses added to its step along `lines`, its file
    /// misses read along `read` as their one request, none hinted
    /// ([`super::step`]): its routed experts each of the step's mean rows
    /// an expert.
    pub fn exposed(&self, lines: &ExpertLines, read: &ReadLine) -> f64 {
        let all = [*lines];
        let inputs = StepInputs {
            compute: 0.,
            head: 0.,
            overlap: 1.,
            read: *read,
            sustained: read.seconds_per_byte(),
            ahead: ReadAhead::NONE,
            lines: &all,
        };
        inputs
            .one(
                &[LayerLoad {
                    routed: self.routed as f64,
                    rows: self.choices as f64 / self.routed as f64,
                    tier: self.hosted,
                    file: self.misses,
                    recall: 0.,
                }],
                &mut [0.],
            )
            .0
    }
}

/// A streamed layer's file reads a bank without read-ahead would miss (`file`)
/// and those a hint read (`hinted`): its recall `R`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Anticipated {
    pub file: u64,
    pub hinted: u64,
}

impl Anticipated {
    pub fn plus(self, other: Self) -> Self {
        Self { file: self.file + other.file, hinted: self.hinted + other.hinted }
    }

    pub fn recall(self) -> f64 {
        match self.file {
            0 => 0.,
            file => self.hinted as f64 / file as f64,
        }
    }
}

/// A load's hint telemetry: the reads hints made, those of an expert the
/// layer's route chose (their precision `P`), and their leads in layers,
/// summed over the reads (`ℓ`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HintCounts {
    pub reads: u64,
    pub used: u64,
    pub lead: u64,
}

impl HintCounts {
    /// The reads of `step`'s layers ([`LayerStep`]).
    pub fn of(step: &[LayerStep]) -> Self {
        step.iter().fold(Self::default(), |sum, layer| sum.plus(Self { reads: layer.hint_reads, used: layer.used, lead: layer.lead }))
    }

    pub fn plus(self, other: Self) -> Self {
        Self { reads: self.reads + other.reads, used: self.used + other.used, lead: self.lead + other.lead }
    }


    /// `P` and `ℓ`: no hint read wastes nothing and leads nothing.
    pub fn ahead(self) -> ReadAhead {
        match self.reads {
            0 => ReadAhead::NONE,
            reads => ReadAhead { precision: (self.used as f64 / reads as f64).max(f64::MIN_POSITIVE), lead: Lead::Layers(self.lead as f64 / reads as f64) },
        }
    }
}

/// A streamed decode step as the one cost function prices it
/// ([`super::streamed::StreamedLayers::step_seconds`]): `C`, what its
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

/// What a stored table's decode steps read (`stored_table::RowStack`): its
/// sampled rows by their depth in its LRU stack, in records of
/// `record_rows` rows (the last bucket deeper than every record), over
/// `steps` steps naming `rows` distinct rows in all, each read as stored
/// from `ranges`, one range of each of its tensors. A table of `r` records
/// misses exactly the rows `r` or more deep.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableCurve {
    pub steps: u64,
    pub rows: u64,
    pub sampled: Vec<u64>,
    pub record_rows: u64,
    pub ranges: Vec<TableRange>,
}

/// One stored tensor's range of every row of a stored table: row `r`'s
/// `bytes` from `first + r·bytes` of its file (`stored_table::RowRange`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableRange {
    pub first: u64,
    pub bytes: u64,
}

impl TableRange {
    /// The bytes a missed row's read of the range reads, over the table's
    /// rows: the whole pages the read engine reads it in
    /// (`seismic_io::ALIGNMENT`, `P`), each range a read of its own (a row's
    /// tensors lie whole tensors apart). Row `r` starts `s = (first + r·b)
    /// mod P` into its page; over the rows `s` takes each of the `n = P/g`
    /// values `c + j·g` (`g = gcd(b, P)`, `c = first mod g`) equally often,
    /// and with `b − 1 = q·P + m` the read spans `1 + q` pages, one more where
    /// `s ≥ P − m`.
    pub fn read_bytes(&self) -> f64 {
        let page = seismic_io::ALIGNMENT;
        let bytes = self.bytes.max(1);
        let g = gcd(bytes, page);
        let (n, c) = (page / g, self.first % g);
        let (q, m) = ((bytes - 1) / page, (bytes - 1) % page);
        // The starts below `P − m`, of the `n`: `ceil((P − m − c) / g)`.
        let short = (page - m).saturating_sub(c).div_ceil(g).min(n);
        page as f64 * ((1 + q) as f64 + (n - short) as f64 / n as f64)
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl TableCurve {
    /// The fewest records at which the table's misses stop falling: one past
    /// the deepest sampled row a record holds. A table of `r` records misses
    /// the sampled rows `r` or more deep ([`Self::step_seconds`]), and the
    /// last bucket, deeper than every record, misses at any size.
    pub fn saturation(&self) -> u64 {
        let held = &self.sampled[..self.sampled.len().saturating_sub(1)];
        held.iter().rposition(|&rows| rows > 0).map_or(0, |deepest| deepest as u64 + 1)
    }

    /// What a decode step's table read costs at `records` records, in
    /// seconds: its misses' one request along `line` (`ReadLine::seconds`),
    /// its latency taken when any of its rows misses and its transfer that of
    /// each read of each row expected to ([`TableRange::read_bytes`]), a
    /// row's miss share the sampled rows' at that depth, independent of the
    /// step's other rows (they are hashed rows of the table). None before a
    /// step counted a sampled row.
    pub fn step_seconds(&self, records: u64, line: &ReadLine) -> Option<f64> {
        let total = self.sampled.iter().sum::<u64>();
        if total == 0 || self.steps == 0 {
            return None;
        }
        let depth = (records as usize).min(self.sampled.len() - 1);
        let share = self.sampled[depth..].iter().sum::<u64>() as f64 / total as f64;
        let rows = self.rows as f64 / self.steps as f64;
        let misses = share * rows;
        let missing = 1. - (1. - share).powf(rows);
        let row = self.ranges.iter().map(TableRange::read_bytes).sum::<f64>();
        Some(missing * line.latency_seconds() + line.transfer_seconds(misses * row))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    const READ: ReadLine = match ReadLine::new(4e-4, 1. / 2e9) {
        Ok(line) => line,
        Err(_) => panic!("the tests' read line"),
    };

    /// The reference lines' own records, where they hold the figures the
    /// tests state.
    const RECORD: u64 = ExpertLines::REFERENCE_RECORD;

    fn lines() -> ExpertLines {
        ExpertLines::reference(RECORD)
    }

    /// A step routing 8 experts at each of four layers, one choice each.
    fn routed() -> LayerStep {
        LayerStep { routed: 8, choices: 8, ..LayerStep::default() }
    }

    /// A step's layer exposes its tier misses beyond its hits on the CPU
    /// and its file misses after their read.
    #[test]
    fn a_layer_step_exposes_its_misses_by_the_cost_function() {
        let lines = lines();
        assert_eq!(routed().exposed(&lines, &READ), 0.);
        // 6 hits (30 µs) beside one tier miss on the CPU (its handshake and
        // 40 µs), then the file miss once read, 40 µs more.
        let one = LayerStep { hosted: 1, misses: 1, ..routed() };
        let filed = READ.latency_seconds() + RECORD as f64 / 2e9;
        assert!((one.exposed(&lines, &READ) - (filed + 40e-6 - 30e-6)).abs() < 1e-12);
        // Two tier misses alone: 4 + 40 + 40 µs.
        let tiered = LayerStep { hosted: 2, ..routed() };
        assert!((tiered.exposed(&lines, &READ) - (84e-6 - 30e-6)).abs() < 1e-12);
    }

    /// A missed row costs a read of each of its ranges, each the whole pages
    /// it spans: a gridded E4M3 row (256 bytes of codes and 8 of scales, in
    /// two tensors) costs two pages, where the same 264 bytes as one range
    /// cost one, or two for the rows that cross a page.
    #[test]
    fn a_gridded_row_costs_a_read_of_each_of_its_ranges() {
        let page = seismic_io::ALIGNMENT;
        let range = |first, bytes| TableRange { first, bytes };
        // Rows of a page's divisors from a page boundary never cross one.
        assert_eq!(range(0, 256).read_bytes(), page as f64);
        assert_eq!(range(4096 * 7, 8).read_bytes(), page as f64);
        // 264-byte rows from a page boundary start at the 512 multiples of 8
        // in a page; the 32 from 3840 on cross into the next.
        assert_eq!(range(0, 264).read_bytes(), page as f64 * (1. + 32. / 512.));
        // A range wider than a page spans its pages and one more where it
        // crosses: odd starts of rows two bytes past a page, one in 2048
        // (from 4095).
        assert_eq!(range(0, 2 * page).read_bytes(), 2. * page as f64);
        assert_eq!(range(0, page + 2).read_bytes(), 2. * page as f64);
        assert_eq!(range(1, page + 2).read_bytes(), page as f64 * (2. + 1. / 2048.));
        let curve = |ranges| TableCurve { steps: 1, rows: 1, sampled: vec![0, 1], record_rows: 1, ranges };
        let gridded = curve(vec![range(0, 256), range(1 << 36, 8)]);
        let whole = curve(vec![range(0, 264)]);
        // Every row misses at no record: one request's latency, and the pages.
        let seconds = |curve: &TableCurve| curve.step_seconds(0, &READ).unwrap();
        assert_eq!(seconds(&gridded), READ.latency_seconds() + READ.transfer_seconds(2. * page as f64));
        assert!(seconds(&whole) < seconds(&gridded));
    }
}
