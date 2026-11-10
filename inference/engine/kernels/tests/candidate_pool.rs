// The candidate pool (`candidate_blocks`, then `rows_topk` over each row's
// blocks, then `candidate_rows`) against a host model of the checkpoint's
// `select_candidate_blocks` and restricted top-k (`inference/model.py`): per
// row of c units, a block scores its units' largest Produce score, the
// newest block is pinned in, the B best blocks are the pool (ties to the
// lower block), and a restricted layer masks every unit outside the pool to
// -inf and takes its K best (ties to the lower unit). Checked exactly:
// - the pool's spans: the pool's units and, while they are fewer than
//   min(K, c), the lowest units outside it, as maximal runs within each list
//   span, in unit order, [-1, -1] padded;
// - `rows_topk` over the restricted scores of that list picks exactly the
//   masked top-k's rows.
// On the portable bodies (small cases) and every backend this host opens.
// Cases: V4.1's pool (2048 blocks of 8, K = 512) over long scattered lists,
// the small-pool definition (8 blocks of 2, K = 8), a pool whose newest
// block is partial and leaves it short of K (the fill), young rows with
// fewer blocks than B or units than K, an empty row, and tied scores.

mod common;
mod indexer_common;
mod rounding_model;
mod versions_common;

use indexer_common::*;
use magnitude_kernels::{candidate_blocks, candidate_rows, rows_topk};
use seismic::{Device, Element, Tensor};
use std::collections::HashMap;
use versions_common::*;

struct Case {
    /// Scores per row (N), the pool's blocks (B), units a block (BU) and the
    /// restricted picks (K).
    n: usize,
    b: usize,
    bu: usize,
    k: usize,
    /// Each row's list: its c units' plane rows.
    lists: Vec<Vec<[i32; 2]>>,
    /// The Produce and the restricted scores of each row's units, -inf past
    /// its c.
    produce: Vec<f32>,
    restrict: Vec<f32>,
}

impl Case {
    /// Rows with lists of `units` units each, and scores drawn from `levels`
    /// distinct values (0: continuous).
    #[allow(clippy::too_many_arguments)]
    fn new(n: usize, b: usize, bu: usize, k: usize, units: &[usize], levels: u32, seed: u64) -> Self {
        let mut random = Random(seed);
        let draw = |random: &mut Random| {
            let x = random.next();
            match levels {
                0 => x * 100.0,
                _ => ((x + 1.0) * 0.5 * levels as f32).floor() * 0.5,
            }
        };
        let lists = units.iter().enumerate().map(|(row, c)| scattered_spans(*c, seed + row as u64)).collect();
        let mut produce = vec![f32::NEG_INFINITY; units.len() * n];
        let mut restrict = produce.clone();
        for (row, c) in units.iter().enumerate() {
            for unit in 0..*c {
                produce[row * n + unit] = draw(&mut random);
                restrict[row * n + unit] = draw(&mut random);
            }
        }
        Self { n, b, bu, k, lists, produce, restrict }
    }

    fn rows(&self) -> usize {
        self.lists.len()
    }

    fn nb(&self) -> usize {
        self.n.div_ceil(self.bu)
    }

    fn units(&self, row: usize) -> usize {
        list_rows(&self.lists[row]).len()
    }

    /// The host model's pool of `row`: whether each unit is in it.
    fn pool(&self, row: usize) -> Vec<bool> {
        let c = self.units(row);
        let scores = &self.produce[row * self.n..][..c];
        let count = c.div_ceil(self.bu);
        let block = |b: usize| match b + 1 == count {
            true => f32::INFINITY,
            false => scores[b * self.bu..((b + 1) * self.bu).min(c)].iter().fold(f32::NEG_INFINITY, |m, s| m.max(*s)),
        };
        let mut order = (0..count).collect::<Vec<_>>();
        order.sort_by(|x, y| block(*y).partial_cmp(&block(*x)).unwrap().then(x.cmp(y)));
        order.truncate(self.b);
        (0..c).map(|unit| order.contains(&(unit / self.bu))).collect()
    }

    /// The host model's restricted picks of `row`, as plane rows ascending:
    /// every score outside the pool -inf, then the min(K, c) best, ties to
    /// the lower unit.
    fn picks(&self, row: usize) -> Vec<i32> {
        let rows = list_rows(&self.lists[row]);
        let pool = self.pool(row);
        let masked = |unit: usize| if pool[unit] { self.restrict[row * self.n + unit] } else { f32::NEG_INFINITY };
        let mut order = (0..rows.len()).collect::<Vec<_>>();
        order.sort_by(|x, y| masked(*y).partial_cmp(&masked(*x)).unwrap().then(x.cmp(y)));
        order.truncate(self.k.min(rows.len()));
        order.sort();
        order.into_iter().map(|unit| rows[unit]).collect()
    }

    /// The pool's expected spans of `row`: its units and the fill, as
    /// maximal runs within each list span.
    fn spans(&self, row: usize) -> Vec<[i32; 2]> {
        let pool = self.pool(row);
        let mut fill = self.k.min(pool.len()).saturating_sub(pool.iter().filter(|p| **p).count());
        let mut spans = Vec::new();
        let mut unit = 0;
        for &[lo, hi] in &self.lists[row] {
            let mut open = false;
            for at in lo..hi {
                let taken = pool[unit] || (fill > 0 && {
                    fill -= 1;
                    true
                });
                match (taken, open) {
                    (true, true) => spans.last_mut().map(|span: &mut [i32; 2]| span[1] = at + 1).unwrap(),
                    (true, false) => spans.push([at, at + 1]),
                    (false, _) => {}
                }
                open = taken;
                unit += 1;
            }
        }
        spans
    }

    fn complete(&self) -> Vec<i32> {
        (0..self.rows()).map(|row| self.units(row) as i32).collect()
    }

    /// The restricted scores in the order of each row's pool list.
    fn listed(&self, pool: &[i32], width: usize) -> Vec<f32> {
        let mut listed = vec![f32::NEG_INFINITY; self.rows() * self.n];
        for row in 0..self.rows() {
            let units = list_rows(&self.lists[row]).into_iter().enumerate().map(|(unit, at)| (at, unit)).collect::<HashMap<_, _>>();
            let spans = pool[row * width * 2..][..width * 2].chunks_exact(2).map(|s| [s[0], s[1]]).collect::<Vec<_>>();
            for (ordinal, at) in list_rows(&spans).into_iter().enumerate().take(self.n) {
                listed[row * self.n + ordinal] = self.restrict[row * self.n + units[&at]];
            }
        }
        listed
    }
}

/// The pool's spans `[M, B + R, 2]` and the restricted picks `[M, K]`.
struct Outcome {
    pool: Vec<i32>,
    picks: Vec<i32>,
}

fn portable(case: &Case) -> Outcome {
    let m = case.rows();
    let (spans, r) = span_table(&case.lists);
    let (blocks, _) = interpret(
        "candidate_blocks",
        false,
        vec![
            Input::F32(vec![m, case.n], case.produce.clone()),
            Input::I32(vec![m], case.complete()),
            Input::I32(vec![0, case.nb()], Vec::new()),
            Input::I32(vec![0, case.bu], Vec::new()),
        ],
    );
    let floats = |values: &[f64]| values.iter().map(|v| *v as f32).collect::<Vec<_>>();
    let ints = |values: &[f64]| values.iter().map(|v| *v as i32).collect::<Vec<_>>();
    let (picked, _) = interpret(
        "rows_topk",
        false,
        vec![
            Input::F32(vec![m, case.nb()], floats(&blocks[0])),
            Input::I32(vec![m], case.complete()),
            Input::I32(vec![m, 1, 2], ints(&blocks[1])),
            Input::I32(vec![0, case.b], Vec::new()),
        ],
    );
    let (pool, _) = interpret(
        "candidate_rows",
        false,
        vec![
            Input::I32(vec![m, case.b], ints(&picked[0])),
            Input::I32(vec![m, r, 2], spans),
            Input::I32(vec![0, case.bu, case.k], Vec::new()),
        ],
    );
    let pool = ints(&pool[0]);
    let width = case.b + r;
    let (picks, _) = interpret(
        "rows_topk",
        false,
        vec![
            Input::F32(vec![m, case.n], case.listed(&pool, width)),
            Input::I32(vec![m], case.complete()),
            Input::I32(vec![m, width, 2], pool.clone()),
            Input::I32(vec![0, case.k], Vec::new()),
        ],
    );
    Outcome { pool, picks: ints(&picks[0]) }
}

/// An entry's kernel at its default specialization on `device`.
macro_rules! prepared {
    ($device:expr, $entry:expr) => {{
        let entry = $entry;
        let specialization = entry
            .offered($device)
            .unwrap()
            .unwrap()
            .default_specialization(&seismic::NativeSpecialization::new())
            .unwrap();
        entry.prepare($device, &specialization).unwrap()
    }};
}

fn words(tensor: &Tensor) -> Vec<i32> {
    tensor.read_to_host().unwrap().chunks_exact(4).map(|word| i32::from_le_bytes(word.try_into().unwrap())).collect()
}

fn native(device: &Device, case: &Case) -> Outcome {
    let m = case.rows() as u64;
    let (spans, r) = span_table(&case.lists);
    let complete = ints(device, &[m], &case.complete());
    let host = |shape: &[u64]| Tensor::from_host(device, Element::i32(), shape, &[]).unwrap();
    let blocks = prepared!(device, candidate_blocks::native_entry())
        .call(candidate_blocks::Args {
            scores: &tensor(device, Element::f32(), &[m, case.n as u64], &case.produce),
            complete: &complete,
            block_slots: &ints(device, &[0, case.nb() as u64], &[]),
            form: &host(&[0, case.bu as u64]),
        })
        .unwrap();
    let topk = prepared!(device, rows_topk::native_entry());
    let picked = topk
        .call(rows_topk::Args {
            scores: &blocks.r0,
            complete: &complete,
            spans: &blocks.r1,
            top_slots: &ints(device, &[0, case.b as u64], &[]),
        })
        .unwrap();
    let pool = prepared!(device, candidate_rows::native_entry())
        .call(candidate_rows::Args {
            blocks: &picked.value,
            spans: &ints(device, &[m, r as u64, 2], &spans),
            form: &host(&[0, case.bu as u64, case.k as u64]),
        })
        .unwrap();
    let width = (case.b + r) as u64;
    let picks = topk
        .call(rows_topk::Args {
            scores: &tensor(device, Element::f32(), &[m, case.n as u64], &case.listed(&words(&pool.value), width as usize)),
            complete: &complete,
            spans: &pool.value,
            top_slots: &ints(device, &[0, case.k as u64], &[]),
        })
        .unwrap();
    Outcome { pool: words(&pool.value), picks: words(&picks.value) }
}

fn check(case: &Case, label: &str) {
    let (_, r) = span_table(&case.lists);
    let width = case.b + r;
    let mut pool = Vec::<i32>::new();
    let mut picks = Vec::<i32>::new();
    for row in 0..case.rows() {
        let spans = case.spans(row);
        assert!(spans.len() <= width, "{label}: row {row}'s pool takes {} spans", spans.len());
        pool.extend(spans.iter().chain(std::iter::repeat(&[-1, -1])).take(width).flatten());
        let row_picks = case.picks(row);
        picks.extend(row_picks.iter().copied().chain(std::iter::repeat(-1)).take(case.k));
    }
    if case.rows() * case.n <= 512 {
        let outcome = portable(case);
        assert_eq!(outcome.pool, pool, "portable {label}: pool");
        assert_eq!(outcome.picks, picks, "portable {label}: picks");
    }
    // The three entries share their backends (`build.rs`).
    for device in offering(&candidate_blocks::native_entry()) {
        let outcome = native(&device, case);
        let backend = device.backend().as_str();
        assert_eq!(outcome.pool, pool, "{backend} {label}: pool");
        assert_eq!(outcome.picks, picks, "{backend} {label}: picks");
    }
}

#[test]
fn candidate_pool_matches_the_masked_top_k() {
    // The small-pool definition (8 blocks of 2, K = 8): binding past 16
    // units, young rows below it, an empty row, ties.
    check(&Case::new(100, 8, 2, 8, &[100, 17, 9, 1, 0], 4, 1), "small pool ties");
    check(&Case::new(100, 8, 2, 8, &[97, 64, 16], 0, 2), "small pool continuous");
    // A pool of 2 blocks of 4 at K = 8 whose newest block holds one unit:
    // 5 pooled units, filled with the 3 lowest outside.
    check(&Case::new(64, 2, 4, 8, &[41, 33, 8, 5], 0, 3), "short pool filled");
    check(&Case::new(64, 2, 4, 8, &[41, 37, 29], 2, 4), "short pool filled, ties");
    // V4.1's pool (2048 blocks of 8, K = 512): long scattered lists, young
    // rows with fewer blocks than B, one with fewer units than K.
    check(&Case::new(65_536, 2048, 8, 512, &[65_536, 30_001, 16_385, 9_000, 400], 0, 5), "v4.1 continuous");
    check(&Case::new(20_000, 2048, 8, 512, &[20_000, 16_390], 3, 6), "v4.1 three levels");
}

/// The time of `candidate_rows` on Metal at DeepSeek V4.1 Flash's decode
/// shape as its producing layer serves it: one row (M = 1), its pool of
/// B = 2048 blocks of BU = 8 units filled to K = 512, over a list of one
/// span (R = 1). Two contexts: 2048 units, whose 256 blocks are all picked
/// (`rows_topk` takes every block of a row with at most B, then -1), and
/// 32768 units, 2048 of whose 4096 blocks are picked, every other one, so
/// each block is its own entry (the most entries B picks make). The entry
/// has no parameter: each context is timed as the default of its own
/// survey, the formed default its only program. It prints each one's
/// median device seconds a call.
#[test]
#[ignore = "times candidate_rows at V4.1's decode shape on Metal; run it by name under a timed hold"]
fn candidate_rows_at_deepseek_v41_decode_shape_times() {
    let Some(device) = offering(&candidate_rows::native_entry())
        .into_iter()
        .find(|device| device.backend() == seismic::BackendName::Metal)
    else {
        return;
    };
    let (b, bu, k) = (2048usize, 8usize, 512usize);
    for (units, stride) in [(2048usize, 1usize), (32768, 2)] {
        let picked = (0..b).map(|pick| pick * stride).take_while(|block| block * bu < units);
        let picks = picked.map(|block| block as i32).chain(std::iter::repeat(-1)).take(b).collect::<Vec<_>>();
        let blocks = ints(&device, &[1, b as u64], &picks);
        let spans = ints(&device, &[1, 1, 2], &[0, units as i32]);
        let form = Tensor::from_host(&device, Element::i32(), &[0, bu as u64, k as u64], &[]).unwrap();
        let point = seismic::TuningPoint {
            label: "m1".into(),
            weight: 1.0,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![candidate_rows::Args { blocks: &blocks, spans: &spans, form: &form }],
            initialize: None,
            written: Default::default(),
        };
        let result = candidate_rows::native_tune(
            &device,
            &seismic::NativeSpecialization::new(),
            vec![point],
            seismic::NumericalRequirement::BitExact,
            seismic::Strategy::Survey(seismic::SurveyPlan {
                samples: 15,
                min_sample_seconds: 0.001,
                domains: Default::default(),
            }),
        )
        .unwrap();
        let [record] = &result.configurations[..] else {
            panic!("candidate_rows c{units}: a fixed configuration surveys itself alone");
        };
        let seismic::Outcome::Measured { points, .. } = &record.outcome else {
            panic!("candidate_rows c{units}: the default is measured");
        };
        println!("candidate_rows v41 m1 c{units}: {:.1} us a call", points[0].median_seconds * 1e6);
    }
}
