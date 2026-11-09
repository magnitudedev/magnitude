// `rows_topk` bit-equal to a sort-based host reference, on the portable body
// and every backend this host opens: each row's picks are the key-plane rows
// of its K best scores (descending, ties to the lower ordinal), in ordinal
// order, -1 padded; a dense row (complete units at most K) reads the first K
// rows of its list. Cases: heavy ties (few distinct values, signed zeros,
// negative scores), lists shorter than K (narrowed or young rows), dense
// rows, and V4.1's K = 512 over 64k-row lists whose K-th score is tied.

mod common;
mod indexer_common;
mod rounding_model;
mod versions_common;

use indexer_common::*;
use magnitude_kernels::rows_topk;
use versions_common::*;

struct Case {
    n: usize,
    k: usize,
    scores: Vec<f32>,
    complete: Vec<i32>,
    lists: Vec<Vec<[i32; 2]>>,
}

impl Case {
    /// Rows with lists of `lengths` rows, `complete` units each, and scores
    /// drawn from `levels` distinct values (0: continuous).
    fn new(n: usize, k: usize, lengths: &[usize], complete: &[i32], levels: u32, seed: u64) -> Self {
        let mut random = Random(seed);
        let lists = lengths.iter().enumerate().map(|(row, length)| scattered_spans(*length, seed + row as u64)).collect::<Vec<_>>();
        let mut scores = vec![f32::NEG_INFINITY; lengths.len() * n];
        for (row, length) in lengths.iter().enumerate() {
            for ordinal in 0..(*length).min(n) {
                let x = random.next();
                scores[row * n + ordinal] = match levels {
                    0 => x * 100.0,
                    _ => {
                        let level = ((x + 1.0) * 0.5 * levels as f32) as i32 - levels as i32 / 2;
                        // Signed zeros at level zero.
                        if level == 0 && ordinal % 2 == 1 { -0.0 } else { level as f32 * 0.5 }
                    }
                };
            }
        }
        Self { n, k, scores, complete: complete.to_vec(), lists }
    }

    fn rows(&self) -> usize {
        self.lists.len()
    }

    /// The sort-based reference.
    fn expected(&self) -> Vec<i32> {
        let mut picks = Vec::new();
        for row in 0..self.rows() {
            let list = list_rows(&self.lists[row]).into_iter().take(self.n).collect::<Vec<_>>();
            let mut chosen = if i64::from(self.complete[row]) <= self.k as i64 {
                (0..list.len().min(self.k)).collect::<Vec<_>>()
            } else {
                let scores = &self.scores[row * self.n..][..self.n];
                let mut order = (0..list.len()).collect::<Vec<_>>();
                order.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap().then(a.cmp(b)));
                order.truncate(self.k);
                order.sort();
                order
            };
            chosen.resize(self.k, usize::MAX);
            picks.extend(chosen.into_iter().map(|ordinal| if ordinal == usize::MAX { -1 } else { list[ordinal] }));
        }
        picks
    }
}

fn portable(case: &Case) -> Vec<i32> {
    let m = case.rows();
    let (spans, r) = span_table(&case.lists);
    let (results, _) = interpret(
        "rows_topk",
        false,
        vec![
            Input::F32(vec![m, case.n], case.scores.clone()),
            Input::I32(vec![m], case.complete.clone()),
            Input::I32(vec![m, r, 2], spans),
            Input::I32(vec![0, case.k], Vec::new()),
        ],
    );
    results[0].iter().map(|v| *v as i32).collect()
}

fn native(device: &seismic::Device, case: &Case) -> Vec<i32> {
    let m = case.rows() as u64;
    let (spans, r) = span_table(&case.lists);
    let entry = rows_topk::native_entry();
    let specialization =
        entry.offered(device).unwrap().unwrap().default_specialization(&seismic::NativeSpecialization::new()).unwrap();
    let kernel = entry.prepare(device, &specialization).unwrap();
    let results = kernel
        .call(rows_topk::Args {
            scores: &tensor(device, seismic::Element::f32(), &[m, case.n as u64], &case.scores),
            complete: &ints(device, &[m], &case.complete),
            spans: &ints(device, &[m, r as u64, 2], &spans),
            top_slots: &ints(device, &[0, case.k as u64], &[]),
        })
        .unwrap();
    results.value.read_to_host().unwrap().chunks_exact(4).map(|word| i32::from_le_bytes(word.try_into().unwrap())).collect()
}

fn check(case: &Case, label: &str) {
    let expected = case.expected();
    if case.rows() * case.n <= 256 {
        assert_eq!(portable(case), expected, "portable {label}");
    }
    for device in offering(&rows_topk::native_entry()) {
        assert_eq!(native(&device, case), expected, "{} {label}", device.backend().as_str());
    }
}

#[test]
fn rows_topk_selects_the_reference_rows() {
    // Small, through the portable body too: ties, a short list, a dense row.
    check(&Case::new(40, 8, &[40, 5, 30], &[40, 20, 8], 5, 1), "small ties");
    check(&Case::new(40, 8, &[33, 40, 12], &[60, 40, 12], 0, 2), "small continuous");
    // V4.1's K: heavy ties over 2k rows, a list shorter than K (narrowed:
    // more complete units than rows), a young dense row.
    check(&Case::new(2048, 512, &[2048, 1500, 300, 400], &[2048, 1500, 900, 400], 7, 3), "ties at 512");
    // 64k-row lists, continuous and tied.
    check(&Case::new(65_536, 512, &[65_536, 60_000], &[65_536, 60_000], 0, 4), "64k continuous");
    check(&Case::new(65_536, 512, &[65_536], &[65_536], 3, 5), "64k three levels");
}
