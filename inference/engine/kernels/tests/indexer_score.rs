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
    h: usize,
    p: usize,
    s: usize,
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

    fn with_shape(rows: usize, n: usize, h: usize, p: usize, s: usize, deepseek: bool, seed: u64) -> Self {
        Self::listing_with_shape(rows, n, h, p, s, deepseek, seed, |row| {
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
        Self::listing_with_shape(rows, n, H, P, S, deepseek, seed, list)
    }

    fn listing_with_shape(
        rows: usize,
        n: usize,
        h: usize,
        p: usize,
        s: usize,
        deepseek: bool,
        seed: u64,
        list: impl Fn(usize) -> Vec<[i32; 2]>,
    ) -> Self {
        let w = 2 * p + s;
        let mut random = Random(seed);
        let (layout, rounding) = if deepseek { (DEEPSEEK, INDEX) } else { (HALF_SPLIT, NONE) };
        let query = (0..rows * h * w).map(|_| bf16_round(random.next() * 3.0)).collect();
        let query_norm = (0..w).map(|_| 1.0 + 0.5 * random.next()).collect();
        let head_weights = (0..rows * h).map(|_| bf16_round(random.next())).collect();
        let lists = (0..rows).map(list).collect::<Vec<_>>();
        let positions = (0..rows).map(|row| 2 * n + row).collect();
        let plane_rows = lists.iter().flatten().map(|span| span[1]).max().unwrap() as usize + 1;
        // Keys as `indexer_keys` stores them: rounded rows.
        let keys = (0..plane_rows)
            .flat_map(|_| {
                let row = (0..w).map(|_| random.next()).collect::<Vec<_>>();
                rounding_model::round(rounding, &row)
            })
            .collect();
        let scale = 1.0 / ((w * h) as f32).sqrt();
        Self {
            rows,
            n,
            h,
            p,
            s,
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

    fn w(&self) -> usize {
        2 * self.p + self.s
    }

    fn hw(&self) -> usize {
        if self.weighted { self.h } else { 0 }
    }

    fn nq(&self) -> usize {
        usize::from(self.normed)
    }

    fn frequencies(&self) -> Vec<f32> {
        (0..self.p).map(|pair| 160_000f32.powf(-(2.0 * pair as f32) / (2.0 * self.p as f32))).collect()
    }

    fn prepare_head(&self, x: &[f32], position: usize, freqs: &[f32]) -> Vec<f32> {
        let hold = |y: &mut [f32]| {
            if self.rounding.block != 0 {
                y.iter_mut().for_each(|v| *v = bf16_round(*v));
            }
        };
        let mut y = x.to_vec();
        if self.normed {
            let squares = y.iter().map(|v| v * v).sum::<f32>();
            let inverse = 1.0 / (squares / self.w() as f32 + EPSILON).sqrt();
            for (v, n) in y.iter_mut().zip(&self.query_norm) {
                *v = *v * inverse * n;
            }
            hold(&mut y);
        }
        let first = if self.layout.rt == 1 { self.s } else { 0 };
        for (pair, frequency) in freqs.iter().enumerate() {
            let (lo, hi) = if self.layout.ra == 1 {
                (first + 2 * pair, first + 2 * pair + 1)
            } else {
                (first + pair, first + pair + self.p)
            };
            let (sin, cos) = (position as f32 * frequency).sin_cos();
            let (low, high) = (y[lo], y[hi]);
            y[lo] = low * cos - high * sin;
            y[hi] = high * cos + low * sin;
        }
        hold(&mut y);
        rounding_model::round(self.rounding, &y)
    }

    fn coordinates(&self) -> Vec<i32> {
        self.positions.iter().flat_map(|position| coordinates(*position)).collect()
    }

    /// The model's scores and each score's absolute terms
    /// (scale · Σ_h |w_h| Σ_c |q_hc k_c|).
    fn expected(&self) -> (Vec<f32>, Vec<f32>) {
        let w = self.w();
        let freqs = self.frequencies();
        let mut scores = vec![f32::NEG_INFINITY; self.rows * self.n];
        let mut terms = vec![0.0; self.rows * self.n];
        for row in 0..self.rows {
            let q = (0..self.h)
                .map(|h| self.prepare_head(&self.query[(row * self.h + h) * w..][..w], self.positions[row], &freqs))
                .collect::<Vec<_>>();
            for (ordinal, at) in list_rows(&self.lists[row]).into_iter().take(self.n).enumerate() {
                let k = &self.keys[at as usize * w..][..w];
                let (mut score, mut term) = (0.0f32, 0.0f32);
                for (h, qh) in q.iter().enumerate() {
                    let wt = if self.weighted { self.head_weights[row * self.h + h] } else { 1.0 };
                    let dot = qh.iter().zip(k).fold(0.0f32, |dot, (a, b)| a.mul_add(*b, dot));
                    score += dot.max(0.0) * wt;
                    term += wt.abs() * qh.iter().zip(k).map(|(a, b)| (a * b).abs()).sum::<f32>();
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
    let w = case.w();
    let (spans, r) = span_table(&case.lists);
    let [block, element, scale, floor] = rounding_inputs(case.rounding);
    let (results, _) = interpret(
        "indexer_score",
        true,
        vec![
            Input::Bf16(vec![m, case.h, w], case.query.clone()),
            Input::F32(vec![case.nq(), w], case.query_norm[..case.nq() * w].to_vec()),
            Input::Bf16(vec![m, case.hw()], if case.weighted { case.head_weights.clone() } else { Vec::new() }),
            Input::I32(vec![case.p], vec![0; case.p]),
            Input::F32(vec![case.p], case.frequencies()),
            Input::F32(vec![case.p], vec![1.0; case.p]),
            Input::I32(vec![m, 4], case.coordinates()),
            Input::F32(vec![case.plane_rows, w], case.keys.clone()),
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

fn native_if_admitted(device: &seismic::Device, case: &Case, required: bool) -> Option<Vec<f32>> {
    let charges = seismic::Charges::tally(device);
    let m = case.rows as u64;
    let w = case.w();
    let (spans, r) = span_table(&case.lists);
    let statics = [
        ("H", case.h as u64),
        ("P", case.p as u64),
        ("S", case.s as u64),
        ("NQ", case.nq() as u64),
        ("HW", case.hw() as u64),
        ("RA", case.layout.ra as u64),
        ("RT", case.layout.rt as u64),
    ];
    let specialization = statics.iter().fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(*name, *value));
    let entry = indexer_score::native_entry_with(indexer_score::Elements { A: Element::bf16() });
    let offer = entry.offered(device).unwrap().unwrap();
    let specialization = match offer.default_specialization(&specialization) {
        Ok(spec) => spec,
        Err(err) if required => panic!("{} rejected required shape (H={}, W={}): {err}", device.backend().as_str(), case.h, w),
        Err(_) => return None,
    };
    let kernel = entry.prepare(device, &specialization).unwrap();
    let (_slabs, logical) = slab_arenas(
        device,
        case.plane_rows,
        SLAB_ROWS as u64,
        &[Arena { element: Element::f32(), bank_shape: vec![w as u64], values: &case.keys }],
        false,
    );
    let keys = logical.into_iter().next().unwrap();
    let results = kernel
        .call(&charges, indexer_score::Args {
            query: &tensor(device, Element::bf16(), &[m, case.h as u64, w as u64], &case.query),
            query_norm: &tensor(device, Element::f32(), &[case.nq() as u64, w as u64], &case.query_norm[..case.nq() * w]),
            head_weights: &tensor(
                device,
                Element::bf16(),
                &[m, case.hw() as u64],
                if case.weighted { &case.head_weights } else { &[] },
            ),
            rotary_components: &ints(device, &[case.p as u64], &vec![0; case.p]),
            rotary_frequencies: &tensor(device, Element::f32(), &[case.p as u64], &case.frequencies()),
            rotary_amplitudes: &tensor(device, Element::f32(), &[case.p as u64], &vec![1.0; case.p]),
            coordinates: &ints(device, &[m, 4], &case.coordinates()),
            keys: &keys,
            spans: &ints(device, &[m, r as u64, 2], &spans),
            score_slots: &ints(device, &[0, case.n as u64], &[]),
            form: &Tensor::from_host(&charges, Element::i32(), &[0, case.layout.ra as u64, case.layout.rt as u64], &[]).unwrap(),
            scale: case.scale,
            epsilon: EPSILON,
            block: case.rounding.block,
            element: case.rounding.element,
            rounding_scale: case.rounding.scale,
            amax_floor: case.rounding.amax_floor,
            slab_rows: SLAB_ROWS as u32,
        })
        .unwrap();
    Some(read(&results.value))
}

fn check_with(case: &Case, label: &str, must_admit: impl Fn(seismic::BackendName) -> bool) {
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
    if case.rows * case.n <= 256 && case.w() <= 128 {
        compare(&portable(case), "portable");
    }
    for device in offering(&indexer_score::native_entry_with(indexer_score::Elements { A: Element::bf16() })) {
        if let Some(actual) = native_if_admitted(&device, case, must_admit(device.backend())) {
            compare(&actual, device.backend().as_str());
        }
    }
}

fn check(case: &Case, label: &str) {
    check_with(case, label, |_| true);
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

/// Wide-width shapes (`W = 512`) that exceed Metal's 32 KiB threadgroup budget
/// when holding all H prepared heads at once: Metal's head-tiled path admits
/// them (including partial head tiles and `H = 64`) and matches the F32 model.
#[test]
fn indexer_scores_head_tiled_wide_shape_equal_the_reference_scores() {
    use seismic::BackendName;
    check_with(
        &Case::with_shape(3, 300, 32, 128, 256, true, 10),
        "wide w512 h32 deepseek",
        |b| matches!(b, BackendName::Metal | BackendName::Cuda | BackendName::Cpu),
    );
    check_with(
        &Case::with_shape(2, 300, 12, 128, 256, false, 11),
        "wide w512 h12 partial tile qwen",
        |b| matches!(b, BackendName::Metal | BackendName::Cuda | BackendName::Cpu),
    );
    check_with(
        &Case::with_shape(2, 64, 64, 128, 256, true, 12),
        "wide w512 h64 deepseek",
        |b| matches!(b, BackendName::Metal | BackendName::Cuda | BackendName::Cpu),
    );
}

fn check_cuda_wide_shape(
    device: &seismic::Device,
    h: usize,
    p: usize,
    s: usize,
    rows: usize,
    n: usize,
    deepseek: bool,
    pooled: bool,
    seed: u64,
    label: &str,
) {
    let w = 2 * p + s;
    let mut random = Random(seed);
    let (layout, rounding) = if deepseek { (DEEPSEEK, INDEX) } else { (HALF_SPLIT, NONE) };
    let weighted = deepseek;
    let normed = !deepseek;
    let hw = if weighted { h } else { 0 };
    let nq = usize::from(normed);
    let freqs: Vec<f32> = (0..p).map(|pair| 160_000f32.powf(-(2.0 * pair as f32) / (2.0 * p as f32))).collect();
    let query: Vec<f32> = (0..rows * h * w).map(|_| bf16_round(random.next() * 3.0)).collect();
    let query_norm: Vec<f32> = (0..w).map(|_| 1.0 + 0.5 * random.next()).collect();
    let head_weights: Vec<f32> = (0..rows * h).map(|_| bf16_round(random.next())).collect();
    let lists: Vec<Vec<[i32; 2]>> = (0..rows)
        .map(|row| {
            let count = n.saturating_sub(7 * row).max(1);
            if pooled { pool_spans(count, seed * 17 + row as u64) } else { scattered_spans(count, seed * 31 + row as u64) }
        })
        .collect();
    let positions: Vec<usize> = (0..rows).map(|row| 2 * n + row).collect();
    let plane_rows = lists.iter().flatten().map(|span| span[1]).max().unwrap() as usize + 1;
    let keys: Vec<f32> = (0..plane_rows)
        .flat_map(|_| {
            let row = (0..w).map(|_| random.next()).collect::<Vec<_>>();
            rounding_model::round(rounding, &row)
        })
        .collect();
    let scale = 1.0 / ((w * h) as f32).sqrt();

    let mut expected = vec![f32::NEG_INFINITY; rows * n];
    let mut terms = vec![0.0f32; rows * n];
    for row in 0..rows {
        let q: Vec<Vec<f32>> = (0..h)
            .map(|head| {
                let mut y = query[(row * h + head) * w..][..w].to_vec();
                let hold = |vals: &mut [f32]| {
                    if rounding.block != 0 {
                        vals.iter_mut().for_each(|v| *v = bf16_round(*v));
                    }
                };
                if normed {
                    let squares = y.iter().map(|v| v * v).sum::<f32>();
                    let inverse = 1.0 / (squares / w as f32 + EPSILON).sqrt();
                    for (v, qn) in y.iter_mut().zip(&query_norm) {
                        *v = *v * inverse * qn;
                    }
                    hold(&mut y);
                }
                let first = if layout.rt == 1 { s } else { 0 };
                for (pair, frequency) in freqs.iter().enumerate() {
                    let (lo, hi) = if layout.ra == 1 { (first + 2 * pair, first + 2 * pair + 1) } else { (first + pair, first + pair + p) };
                    let (sin, cos) = (positions[row] as f32 * frequency).sin_cos();
                    let (low, high) = (y[lo], y[hi]);
                    y[lo] = low * cos - high * sin;
                    y[hi] = high * cos + low * sin;
                }
                hold(&mut y);
                rounding_model::round(rounding, &y)
            })
            .collect();
        for (ordinal, at) in list_rows(&lists[row]).into_iter().take(n).enumerate() {
            let k = &keys[at as usize * w..][..w];
            let (mut score, mut term) = (0.0f32, 0.0f32);
            for (head, qh) in q.iter().enumerate() {
                let weight = if weighted { head_weights[row * h + head] } else { 1.0 };
                let dot = qh.iter().zip(k).fold(0.0f32, |acc, (a, b)| a.mul_add(*b, acc));
                score += dot.max(0.0) * weight;
                term += weight.abs() * qh.iter().zip(k).map(|(a, b)| (a * b).abs()).sum::<f32>();
            }
            expected[row * n + ordinal] = scale * score;
            terms[row * n + ordinal] = scale * term;
        }
    }

    let charges = seismic::Charges::tally(device);
    let m = rows as u64;
    let (spans, r) = span_table(&lists);
    let statics = [
        ("H", h as u64),
        ("P", p as u64),
        ("S", s as u64),
        ("NQ", nq as u64),
        ("HW", hw as u64),
        ("RA", layout.ra as u64),
        ("RT", layout.rt as u64),
    ];
    let specialization = statics.iter().fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(*name, *value));
    let entry = indexer_score::native_entry_with(indexer_score::Elements { A: Element::bf16() });
    let specialization = entry.offered(device).unwrap().unwrap().default_specialization(&specialization).unwrap();
    let kernel = entry.prepare(device, &specialization).unwrap();
    let (_slabs, planes) =
        slab_arenas(device, plane_rows, SLAB_ROWS as u64, &[Arena { element: Element::f32(), bank_shape: vec![w as u64], values: &keys }], false);
    let keys_plane = planes.into_iter().next().unwrap();
    let coords: Vec<i32> = positions.iter().flat_map(|position| coordinates(*position)).collect();
    let results = kernel
        .call(&charges, indexer_score::Args {
            query: &tensor(device, Element::bf16(), &[m, h as u64, w as u64], &query),
            query_norm: &tensor(device, Element::f32(), &[nq as u64, w as u64], &query_norm[..nq * w]),
            head_weights: &tensor(device, Element::bf16(), &[m, hw as u64], if weighted { &head_weights } else { &[] }),
            rotary_components: &ints(device, &[p as u64], &vec![0i32; p]),
            rotary_frequencies: &tensor(device, Element::f32(), &[p as u64], &freqs),
            rotary_amplitudes: &tensor(device, Element::f32(), &[p as u64], &vec![1.0f32; p]),
            coordinates: &ints(device, &[m, 4], &coords),
            keys: &keys_plane,
            spans: &ints(device, &[m, r as u64, 2], &spans),
            score_slots: &ints(device, &[0, n as u64], &[]),
            form: &Tensor::from_host(&charges, Element::i32(), &[0, layout.ra as u64, layout.rt as u64], &[]).unwrap(),
            scale,
            epsilon: EPSILON,
            block: rounding.block,
            element: rounding.element,
            rounding_scale: rounding.scale,
            amax_floor: rounding.amax_floor,
            slab_rows: SLAB_ROWS as u32,
        })
        .unwrap();
    let actual = read(&results.value);
    assert_eq!(actual.len(), expected.len());
    for (index, ((a, e), t)) in actual.iter().zip(&expected).zip(&terms).enumerate() {
        if *e == f32::NEG_INFINITY {
            assert!(*a == f32::NEG_INFINITY, "cuda {label}: score {index} past the list is {a}");
        } else {
            assert!((a - e).abs() <= 2e-3 * t + 1e-30, "cuda {label}: score {index} {a} vs model {e} (terms {t})");
        }
    }
}

/// Wide shapes (`H = 64, W = 512` and `H = 32, W = 512`) exercise CUDA's
/// head-tiled path (fitting within 48 KiB shared memory with `HEAD_TILE = 8`).
#[test]
fn indexer_scores_cuda_wide_shapes_equal_the_reference_scores() {
    let Some(device) = devices().into_iter().find(|device| device.backend() == seismic::BackendName::Cuda) else {
        return;
    };
    check_cuda_wide_shape(&device, 64, 128, 256, 3, 290, true, true, 101, "H=64 W=512 pool");
    check_cuda_wide_shape(&device, 32, 128, 256, 2, 70, false, false, 102, "H=32 W=512 qwen");
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
    let charges = seismic::Charges::tally(&device);
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
    let form = Tensor::from_host(&charges, Element::i32(), &[0, 1, 1], &[]).unwrap();
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
            &charges,
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
