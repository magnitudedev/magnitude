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
        // Row i lists n - 7 i rows (the last at most n), one in four rows a
        // decode row far into the context.
        Self::listing(rows, n, deepseek, seed, |row| {
            scattered_spans(n.saturating_sub(7 * row).max(1), seed * 31 + row as u64)
        })
    }

    /// Rows listing a candidate pool's shape: hundreds of short spans,
    /// empty ones among them, so a simdgroup's ordinals start inside a span
    /// and cover several, and a row's spans fill many 32-span chunks.
    fn pooled(rows: usize, n: usize, seed: u64) -> Self {
        Self::listing(rows, n, true, seed, |row| pool_spans(n.saturating_sub(5 * row).max(1), seed * 17 + row as u64))
    }

    fn listing(rows: usize, n: usize, deepseek: bool, seed: u64, list: impl Fn(usize) -> Vec<[i32; 2]>) -> Self {
        let mut random = Random(seed);
        let (layout, rounding) = if deepseek { (DEEPSEEK, INDEX) } else { (HALF_SPLIT, NONE) };
        let query = (0..rows * H * W).map(|_| bf16_round(random.next() * 3.0)).collect();
        let query_norm = (0..W).map(|_| 1.0 + 0.5 * random.next()).collect();
        let head_weights = (0..rows * H).map(|_| bf16_round(random.next())).collect();
        let lists = (0..rows).map(list).collect::<Vec<_>>();
        let positions = (0..rows).map(|row| 2 * n + row).collect();
        let plane_rows = lists.iter().flatten().map(|span| span[1]).max().unwrap() as usize + 1;
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

/// `count` key-plane rows as a candidate pool lists them: blocks of 8 rows
/// (some cut short) apart from each other, with empty spans ([-1, -1] and
/// [lo, lo]) between them. No span crosses a slab edge.
fn pool_spans(count: usize, seed: u64) -> Vec<[i32; 2]> {
    let mut random = Random(seed);
    let mut spans = Vec::new();
    let mut at = (SLAB_ROWS - 13) as i32;
    let mut left = count as i32;
    while left > 0 {
        let draw = random.next().abs();
        if draw < 0.1 {
            spans.push([-1, -1]);
        } else if draw < 0.15 {
            spans.push([at, at]);
        } else {
            let length = if draw < 0.3 { (draw * 20.0) as i32 % 8 + 1 } else { 8 }.min(left);
            let length = length.min(SLAB_ROWS as i32 - at % SLAB_ROWS as i32);
            spans.push([at, at + length]);
            left -= length;
            at += length;
        }
        at += (random.next().abs() * 12.0) as i32;
    }
    spans
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

/// Rows listing a candidate pool's shape (hundreds of short spans, empty
/// ones among them) score as the reference: each simdgroup's scan of the
/// spans 32 at a time finds every ordinal's row however the spans fall.
#[test]
fn indexer_scores_over_a_candidate_pool_list_equal_the_reference_scores() {
    check(&Case::pooled(3, 2048, 5), "deepseek candidate pool");
}

/// The time of `indexer_score` on Metal at DeepSeek V4.1 Flash's prefill
/// shape: a 2048-token prompt's rows (M = 2048, positions 0..2048) scoring
/// at a 16384-row served context (N = 16384, as a ratio-1 indexer lists its
/// rows, and the candidate pool's cap of 2048 blocks of 8), its 32 heads of
/// 128 with V4.1's head weights and rotary. Two lists: a Reindex layer's
/// (24+), the candidate pool's 2048 block spans of 8 rows (row t's blocks
/// covering its rows [0, t], the rest [-1, -1]) plus its window span; and an
/// own layer's (20), its one span [0, t + 1). The entry has no parameters, so
/// each list is timed as the default of its own survey, the survey's domain
/// that one configuration. It prints the median device seconds a call and
/// the rate of the scores' useful arithmetic (32 · 128 · 2 FLOP a listed
/// row).
#[test]
#[ignore = "times indexer_score at V4.1's prefill shape on Metal; run it by name under a timed hold"]
fn indexer_score_at_deepseek_v41_prefill_shape_times() {
    let Some(device) = devices().into_iter().find(|device| device.backend() == seismic::BackendName::Metal) else {
        return;
    };
    let (m, n, blocks, block) = (2048usize, 16384usize, 2048usize, 8usize);
    let pool = |t: usize| {
        let covered = t / block + 1;
        let mut spans = (0..blocks)
            .map(|b| if b < covered { [(b * block) as i32, ((b + 1) * block).min(t + 1) as i32] } else { [-1, -1] })
            .collect::<Vec<_>>();
        spans.push([-1, -1]);
        spans
    };
    let own = |t: usize| vec![[0, t as i32 + 1]];
    let lists: [(&str, &dyn Fn(usize) -> Vec<[i32; 2]>); 2] = [("reindex pool", &pool), ("own span", &own)];
    let mut random = Random(41);
    let query = (0..m * H * W).map(|_| bf16_round(random.next() * 3.0)).collect::<Vec<_>>();
    let weights = (0..m * H).map(|_| bf16_round(random.next())).collect::<Vec<_>>();
    let keys = (0..m * W).map(|_| random.next()).collect::<Vec<_>>();
    let coordinates = (0..m).flat_map(coordinates).collect::<Vec<_>>();
    let (_slabs, plane) = plane(&device, Element::f32(), m, &keys);
    let statics = [("H", H as u64), ("P", P as u64), ("S", S as u64), ("NQ", 0), ("HW", H as u64), ("RA", 1), ("RT", 1)]
        .iter()
        .fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(*name, *value));
    let query = tensor(&device, Element::bf16(), &[m as u64, H as u64, W as u64], &query);
    let query_norm = tensor(&device, Element::f32(), &[0, W as u64], &[]);
    let head_weights = tensor(&device, Element::bf16(), &[m as u64, H as u64], &weights);
    let components = ints(&device, &[P as u64], &[0; P]);
    let frequencies = tensor(&device, Element::f32(), &[P as u64], &frequencies());
    let amplitudes = tensor(&device, Element::f32(), &[P as u64], &[1.0; P]);
    let coordinates = ints(&device, &[m as u64, 4], &coordinates);
    let score_slots = ints(&device, &[0, n as u64], &[]);
    let form = Tensor::from_host(&device, Element::i32(), &[0, 1, 1], &[]).unwrap();
    for (name, list) in lists {
        let listed = (0..m).map(list).collect::<Vec<_>>();
        let (spans, r) = span_table(&listed);
        let spans = ints(&device, &[m as u64, r as u64, 2], &spans);
        let point = seismic::TuningPoint {
            label: "m2048".into(),
            weight: 1.0,
            class: None,
            cost: 1.0,
            workload: seismic::WorkloadRotation::All,
            rotation: vec![indexer_score::Args {
                query: &query,
                query_norm: &query_norm,
                head_weights: &head_weights,
                rotary_components: &components,
                rotary_frequencies: &frequencies,
                rotary_amplitudes: &amplitudes,
                coordinates: &coordinates,
                keys: &plane,
                spans: &spans,
                score_slots: &score_slots,
                form: &form,
                scale: 1.0 / ((W * H) as f32).sqrt(),
                epsilon: EPSILON,
                block: INDEX.block,
                element: INDEX.element,
                rounding_scale: INDEX.scale,
                amax_floor: INDEX.amax_floor,
                slab_rows: SLAB_ROWS as u32,
            }],
            initialize: None,
            written: Default::default(),
        };
        let result = indexer_score::native_tune_with(
            &device,
            indexer_score::Elements { A: Element::bf16() },
            &statics,
            vec![point],
            seismic::NumericalRequirement::RoundingSteps { alpha: 8., activation: Some(seismic::DType::BF16) },
            seismic::Strategy::Survey(seismic::SurveyPlan {
                samples: 15,
                min_sample_seconds: 0.001,
                domains: Default::default(),
            }),
        )
        .unwrap();
        let [record] = &result.configurations[..] else {
            panic!("{name}: the entry's one configuration surveys itself alone");
        };
        let seismic::Outcome::Measured { points, .. } = &record.outcome else {
            panic!("{name}: the default is measured");
        };
        let median = points[0].median_seconds;
        let flops = (m * (m + 1) / 2 * H * W * 2) as f64;
        println!(
            "indexer_score v41 m2048 n{n} {name} (R {r}): {:.3} ms a call; scores at {:.2} TFLOPS",
            median * 1e3,
            flops / median / 1e12
        );
    }
}
