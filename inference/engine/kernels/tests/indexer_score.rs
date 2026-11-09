// `indexer_score` against an F32 host model of DeepSeek V4.1's index scores
// (32 query heads of 128, rotated in adjacent trailing pairs at the row's
// position, rounded to E2M1 per 32 with UE8M0 scales; s = scale ·
// Σ_h relu(q_h · k) w_h, scale 1/√(128 · 32)) over row lists of up to 64k
// key-plane rows in runs across slab edges, the portable body and every
// backend this host opens. Scores past a row's list are -inf exactly; the
// others agree within F32 arithmetic order, plus a query value's code moving
// to its neighbour where its unrounded value lies at a code boundary (2e-3
// of the score's absolute terms covers both; a wrong head, row, weight,
// rotary or relu moves a score by O(1) of them). Also Qwen4's form: normed
// half-split queries, uniform head weights, no rounding.

mod common;
mod indexer_common;
mod rounding_model;
mod versions_common;

use indexer_common::*;
use magnitude_kernels::indexer_score;
use rounding_model::{INDEX, NONE, Rounding};
use seismic::{Element, NativeSpecialization, Tensor};
use versions_common::*;

struct Case {
    rows: usize,
    n: usize,
    layout: Layout,
    rounding: Rounding,
    /// Head weights per row (HW = H), or uniform.
    weighted: bool,
    /// A query norm (NQ = 1).
    normed: bool,
    query: Vec<f32>,
    query_norm: Vec<f32>,
    head_weights: Vec<f32>,
    positions: Vec<usize>,
    lists: Vec<Vec<[i32; 2]>>,
    plane_rows: usize,
    keys: Vec<f32>,
    scale: f32,
}

impl Case {
    fn new(rows: usize, n: usize, deepseek: bool, seed: u64) -> Self {
        let mut random = Random(seed);
        let (layout, rounding) = if deepseek { (DEEPSEEK, INDEX) } else { (HALF_SPLIT, NONE) };
        let query = (0..rows * H * W).map(|_| bf16_round(random.next() * 3.0)).collect();
        let query_norm = (0..W).map(|_| 1.0 + 0.5 * random.next()).collect();
        let head_weights = (0..rows * H).map(|_| bf16_round(random.next())).collect();
        // Row i lists n - 7 i rows (the last at most n), one in four rows a
        // decode row far into the context.
        let lists = (0..rows)
            .map(|row| scattered_spans(n.saturating_sub(7 * row).max(1), seed * 31 + row as u64))
            .collect::<Vec<_>>();
        let positions = (0..rows).map(|row| 2 * n + row).collect();
        let plane_rows = lists.iter().flatten().map(|span| span[1] as usize).max().unwrap() + 1;
        // Keys as `indexer_keys` stores them: rounded rows.
        let keys = (0..plane_rows)
            .flat_map(|_| {
                let row = (0..W).map(|_| random.next()).collect::<Vec<_>>();
                rounding_model::round(rounding, &row)
            })
            .collect();
        let scale = 1.0 / ((W * H) as f32).sqrt();
        Self {
            rows,
            n,
            layout,
            rounding,
            weighted: deepseek,
            normed: !deepseek,
            query,
            query_norm,
            head_weights,
            positions,
            lists,
            plane_rows,
            keys,
            scale,
        }
    }

    fn hw(&self) -> usize {
        if self.weighted { H } else { 0 }
    }

    fn nq(&self) -> usize {
        usize::from(self.normed)
    }

    fn coordinates(&self) -> Vec<i32> {
        self.positions.iter().flat_map(|position| coordinates(*position)).collect()
    }

    /// The model's scores and each score's absolute terms
    /// (scale · Σ_h |w_h| Σ_c |q_hc k_c|).
    fn expected(&self) -> (Vec<f32>, Vec<f32>) {
        let mut scores = vec![f32::NEG_INFINITY; self.rows * self.n];
        let mut terms = vec![0.0; self.rows * self.n];
        for row in 0..self.rows {
            let q = (0..H)
                .map(|h| {
                    let norm = self.normed.then_some(&self.query_norm[..]);
                    prepare(&self.query[(row * H + h) * W..][..W], norm, self.positions[row], self.layout, self.rounding)
                })
                .collect::<Vec<_>>();
            for (ordinal, at) in list_rows(&self.lists[row]).into_iter().take(self.n).enumerate() {
                let k = &self.keys[at as usize * W..][..W];
                let (mut score, mut term) = (0.0f32, 0.0f32);
                for (h, qh) in q.iter().enumerate() {
                    let w = if self.weighted { self.head_weights[row * H + h] } else { 1.0 };
                    let dot = qh.iter().zip(k).fold(0.0f32, |dot, (a, b)| a.mul_add(*b, dot));
                    score += dot.max(0.0) * w;
                    term += w.abs() * qh.iter().zip(k).map(|(a, b)| (a * b).abs()).sum::<f32>();
                }
                scores[row * self.n + ordinal] = self.scale * score;
                terms[row * self.n + ordinal] = self.scale * term;
            }
        }
        (scores, terms)
    }
}

fn portable(case: &Case) -> Vec<f32> {
    let m = case.rows;
    let (spans, r) = span_table(&case.lists);
    let [block, element, scale, floor] = rounding_inputs(case.rounding);
    let (results, _) = interpret(
        "indexer_score",
        true,
        vec![
            Input::Bf16(vec![m, H, W], case.query.clone()),
            Input::F32(vec![case.nq(), W], case.query_norm[..case.nq() * W].to_vec()),
            Input::Bf16(vec![m, case.hw()], if case.weighted { case.head_weights.clone() } else { Vec::new() }),
            Input::I32(vec![P], vec![0; P]),
            Input::F32(vec![P], frequencies()),
            Input::F32(vec![P], vec![1.0; P]),
            Input::I32(vec![m, 4], case.coordinates()),
            Input::F32(vec![case.plane_rows, W], case.keys.clone()),
            Input::I32(vec![m, r, 2], spans),
            Input::I32(vec![0, case.n], Vec::new()),
            Input::I32(vec![0, case.layout.ra, case.layout.rt], Vec::new()),
            Input::ScalarF32(case.scale),
            Input::ScalarF32(EPSILON),
            block,
            element,
            scale,
            floor,
            Input::ScalarU32(SLAB_ROWS as u32),
        ],
    );
    results[0].iter().map(|v| *v as f32).collect()
}

fn native(device: &seismic::Device, case: &Case) -> Vec<f32> {
    let m = case.rows as u64;
    let (spans, r) = span_table(&case.lists);
    let statics = [
        ("H", H as u64),
        ("P", P as u64),
        ("S", S as u64),
        ("NQ", case.nq() as u64),
        ("HW", case.hw() as u64),
        ("RA", case.layout.ra as u64),
        ("RT", case.layout.rt as u64),
    ];
    let specialization = statics.iter().fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(*name, *value));
    let entry = indexer_score::native_entry_with(indexer_score::Elements { A: Element::bf16() });
    let specialization = entry.offered(device).unwrap().unwrap().default_specialization(&specialization).unwrap();
    let kernel = entry.prepare(device, &specialization).unwrap();
    let (_slabs, keys) = plane(device, Element::f32(), case.plane_rows, &case.keys);
    let results = kernel
        .call(indexer_score::Args {
            query: &tensor(device, Element::bf16(), &[m, H as u64, W as u64], &case.query),
            query_norm: &tensor(device, Element::f32(), &[case.nq() as u64, W as u64], &case.query_norm[..case.nq() * W]),
            head_weights: &tensor(
                device,
                Element::bf16(),
                &[m, case.hw() as u64],
                if case.weighted { &case.head_weights } else { &[] },
            ),
            rotary_components: &ints(device, &[P as u64], &[0; P]),
            rotary_frequencies: &tensor(device, Element::f32(), &[P as u64], &frequencies()),
            rotary_amplitudes: &tensor(device, Element::f32(), &[P as u64], &[1.0; P]),
            coordinates: &ints(device, &[m, 4], &case.coordinates()),
            keys: &keys,
            spans: &ints(device, &[m, r as u64, 2], &spans),
            score_slots: &ints(device, &[0, case.n as u64], &[]),
            form: &Tensor::from_host(device, Element::i32(), &[0, case.layout.ra as u64, case.layout.rt as u64], &[]).unwrap(),
            scale: case.scale,
            epsilon: EPSILON,
            block: case.rounding.block,
            element: case.rounding.element,
            rounding_scale: case.rounding.scale,
            amax_floor: case.rounding.amax_floor,
            slab_rows: SLAB_ROWS as u32,
        })
        .unwrap();
    read(&results.value)
}

fn check(case: &Case, label: &str) {
    let (expected, terms) = case.expected();
    let compare = |actual: &[f32], who: &str| {
        assert_eq!(actual.len(), expected.len());
        for (index, ((a, e), t)) in actual.iter().zip(&expected).zip(&terms).enumerate() {
            if *e == f32::NEG_INFINITY {
                assert!(*a == f32::NEG_INFINITY, "{who} {label}: score {index} past the list is {a}");
            } else {
                assert!((a - e).abs() <= 2e-3 * t + 1e-30, "{who} {label}: score {index} {a} vs model {e} (terms {t})");
            }
        }
    };
    if case.rows * case.n <= 256 {
        compare(&portable(case), "portable");
    }
    for device in offering(&indexer_score::native_entry_with(indexer_score::Elements { A: Element::bf16() })) {
        compare(&native(&device, case), device.backend().as_str());
    }
}

#[test]
fn indexer_scores_equal_the_reference_scores() {
    check(&Case::new(3, 40, true, 1), "deepseek small");
    check(&Case::new(2, 30, false, 2), "qwen form small");
    check(&Case::new(16, 2048, true, 3), "deepseek prefill");
    check(&Case::new(1, 65_536, true, 4), "deepseek decode 64k");
}
