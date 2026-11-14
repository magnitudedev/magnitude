// A token source's reads under its pooled-key indexer (`unit_reads`, then
// `candidate_rows` over its units for the decode entry's Segments) against a
// host model of the rule (`Indexer`): a row at position t with c = (t + 1) / U
// complete units of U rows reads the rows of its picked units and its tail,
// the rows from c U on. Checked exactly:
// - the WorkList bitmap: bit u set when u is picked or u >= c;
// - the units: the picks, then c, then -1;
// - the Segments: the row's visible keys (positions 0..t, its fresh row apart)
//   whose unit is read, as maximal runs within each visible span, in
//   position order, [-1, -1] padded to K + 1 + R.
// On the portable body (small cases) and every backend this host opens.
// Cases: dense rows (c <= K, every unit picked), selecting rows of GLM's
// shape (U = 4, K = 512), a row before its first complete unit, and a young
// row whose tail crosses a span edge.

mod common;
mod indexer_common;
mod rounding_model;
mod versions_common;

use indexer_common::*;
use magnitude_kernels::{candidate_rows, unit_reads};
use seismic::{Device, Element, NativeSpecialization, Tensor};
use versions_common::*;

struct Case {
    rate: usize,
    k: usize,
    /// Bitmap words B.
    words: usize,
    /// Per row: its position, its picks (ascending unit ordinals) and its
    /// visible spans (the physical rows of positions 0..t, in order).
    rows: Vec<(usize, Vec<i32>, Vec<[i32; 2]>)>,
}

impl Case {
    fn new(rate: usize, k: usize, positions: &[usize], seed: u64) -> Self {
        let mut random = Random(seed);
        let rows = positions
            .iter()
            .enumerate()
            .map(|(row, &position)| {
                let complete = (position + 1) / rate;
                // The best min(K, c) units: a random subset, ascending.
                let mut units = (0..complete as i32).collect::<Vec<_>>();
                for at in (1..units.len()).rev() {
                    let other = ((random.next().abs() * 1e6) as usize) % (at + 1);
                    units.swap(at, other);
                }
                units.truncate(k.min(complete));
                units.sort_unstable();
                (position, units, scattered_spans(position, seed + row as u64))
            })
            .collect();
        let longest = positions.iter().max().copied().unwrap_or(0);
        Self { rate, k, words: longest / rate / 32 + 1, rows }
    }

    fn complete(&self) -> Vec<i32> {
        self.rows.iter().map(|(position, ..)| ((position + 1) / self.rate) as i32).collect()
    }

    fn picks(&self) -> Vec<i32> {
        self.rows.iter().flat_map(|(_, picks, _)| padded(picks, self.k)).collect()
    }

    fn read(&self, row: usize, unit: usize) -> bool {
        let (position, picks, _) = &self.rows[row];
        unit >= (position + 1) / self.rate || picks.contains(&(unit as i32))
    }

    fn bitmap(&self) -> Vec<i32> {
        (0..self.rows.len())
            .flat_map(|row| {
                (0..self.words).map(move |word| {
                    (0..32).filter(|bit| self.read(row, word * 32 + bit)).fold(0u32, |bits, bit| bits | 1 << bit) as i32
                })
            })
            .collect()
    }

    fn units(&self) -> Vec<i32> {
        self.rows
            .iter()
            .flat_map(|(position, picks, _)| {
                let mut units = picks.clone();
                units.push(((position + 1) / self.rate) as i32);
                padded(&units, self.k + 1)
            })
            .collect()
    }

    /// The Segments: each visible span's read keys as maximal runs.
    fn segments(&self, r: usize) -> Vec<i32> {
        let mut table = Vec::new();
        for (row, (_, _, spans)) in self.rows.iter().enumerate() {
            let mut runs: Vec<[i32; 2]> = Vec::new();
            let mut ordinal = 0;
            for [lo, hi] in spans {
                let mut open = false;
                for at in *lo..*hi {
                    if self.read(row, ordinal / self.rate) {
                        match (open, runs.last_mut()) {
                            (true, Some(run)) => run[1] = at + 1,
                            _ => runs.push([at, at + 1]),
                        }
                        open = true;
                    } else {
                        open = false;
                    }
                    ordinal += 1;
                }
            }
            runs.resize(self.k + 1 + r, [-1, -1]);
            table.extend(runs.into_iter().flatten());
        }
        table
    }

    fn spans(&self) -> (Vec<i32>, usize) {
        span_table(&self.rows.iter().map(|(_, _, spans)| spans.clone()).collect::<Vec<_>>())
    }
}

fn padded(values: &[i32], length: usize) -> Vec<i32> {
    values.iter().copied().chain(std::iter::repeat(-1)).take(length).collect()
}

fn words(tensor: &Tensor) -> Vec<i32> {
    tensor.read_to_host().unwrap().chunks_exact(4).map(|word| i32::from_le_bytes(word.try_into().unwrap())).collect()
}

/// (bitmap, units) through the portable body.
fn portable(case: &Case) -> (Vec<i32>, Vec<i32>) {
    let m = case.rows.len();
    let (results, _) = interpret(
        "unit_reads",
        false,
        vec![
            Input::I32(vec![m, case.k], case.picks()),
            Input::I32(vec![m], case.complete()),
            Input::U32(vec![0, case.words], Vec::new()),
        ],
    );
    // Bitmap words are u32 (read back as their bits); units are i32.
    let words = results[0].iter().map(|v| *v as u32 as i32).collect::<Vec<_>>();
    let units = results[1].iter().map(|v| *v as i32).collect::<Vec<_>>();
    (words, units)
}

/// An entry's kernel at its default specialization on `device`.
macro_rules! prepared {
    ($device:expr, $entry:expr) => {{
        let entry = $entry;
        let specialization =
            entry.offered($device).unwrap().unwrap().default_specialization(&NativeSpecialization::new()).unwrap();
        entry.prepare($device, &specialization).unwrap()
    }};
}

/// (bitmap, units, segments) on `device`.
fn native(device: &Device, case: &Case) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let m = case.rows.len() as u64;
    let (spans, r) = case.spans();
    let reads = prepared!(device, unit_reads::native_entry());
    let read = reads
        .call(unit_reads::Args {
            picks: &ints(device, &[m, case.k as u64], &case.picks()),
            complete: &ints(device, &[m], &case.complete()),
            bitmap_words: &Tensor::from_host(device, Element::u32(), &[0, case.words as u64], &[]).unwrap(),
        })
        .unwrap();
    let rows = prepared!(device, candidate_rows::native_entry());
    let host = |shape: &[u64]| Tensor::from_host(device, Element::i32(), shape, &[]).unwrap();
    let segments = rows
        .call(candidate_rows::Args {
            blocks: &read.r1,
            spans: &ints(device, &[m, r as u64, 2], &spans),
            form: &host(&[0, case.rate as u64, 0]),
        })
        .unwrap();
    (words(&read.r0), words(&read.r1), words(&segments.value))
}

fn check(case: &Case, label: &str) {
    let (_, r) = case.spans();
    let (bitmap, units, segments) = (case.bitmap(), case.units(), case.segments(r));
    if case.rows.len() * case.k <= 256 {
        assert_eq!(portable(case), (bitmap.clone(), units.clone()), "portable {label}");
    }
    for device in offering(&unit_reads::native_entry()) {
        let backend = device.backend().as_str();
        let (b, u, s) = native(&device, case);
        assert_eq!(b, bitmap, "{backend} {label}: bitmap");
        assert_eq!(u, units, "{backend} {label}: units");
        assert_eq!(s, segments, "{backend} {label}: segments");
    }
}

#[test]
fn unit_reads_read_the_picked_units_and_the_tail() {
    // Small, through the portable body: a row before its first unit, dense
    // rows, selecting rows.
    check(&Case::new(4, 3, &[2, 9, 13, 30, 61], 1), "small");
    // GLM's shape: groups of four, 512 picked units, rows past 2048 keys.
    check(&Case::new(4, 512, &[2047, 2048, 2051, 9000, 40_003], 2), "glm");
    // Qwen4exp-like: blocks of 64 rows, few picks.
    check(&Case::new(64, 16, &[1023, 1088, 5000], 3), "blocks of 64");
}
