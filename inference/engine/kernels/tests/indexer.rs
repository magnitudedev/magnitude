// A selected read over token rows (`graph::select`) end to end on every
// backend this host opens: its keys pooled into F32 unit keys (`pool_rows`),
// scored (`indexer_score`), its units picked over each row's unit list
// (`rows_topk`), and its reads (`unit_reads`: the WorkList bitmap; then
// `candidate_rows`, no fill: a decode row's Segments), against an
// independent host model of the contract's indexer (`Indexer`): each
// complete unit's key rotated at its first row, s = scale · Σ_h relu(q_h ·
// k) w_h, the K best complete units with ties to the lower, every unit from
// the row's tail on. Checked exactly: the bitmap, the Segments and the
// pending keys appended (bit for bit, but under a per-token norm, within F32
// arithmetic order); the unit keys within F32 arithmetic order.
//
// The chain runs through its entries' portable bodies too, where that costs
// little; each entry's own test also holds its portable body to this model.
//
// Two forms. Qwen4exp's: mean-pooled raw keys under a pooled RMS norm,
// normed queries, uniform heads at scale f32(1/√W); its host model also
// selects exactly what the dividing `Σ/√W` order selects on every case (the
// one arithmetic change). GLM-5.3's:
// softmax-gated keys with a per-place term under a per-token LayerNorm with
// bias, unnormed queries, projected head weights, scale (H·W)^-1/2, at its
// shapes (H 32, W 128, rate 4, top 512) past 2048 keys.

mod common;
mod indexer_common;
mod rounding_model;
mod versions_common;

use common::bound::{self, Float, Library, assert_within, gamma};
use magnitude_kernels::{candidate_rows, indexer_score, pool_rows, rows_topk, unit_reads};
use seismic::{BackendName, Device, Element, NativeSpecialization, Tensor};
use versions_common::*;

const EPSILON: f32 = 1e-6;
const P: usize = 32;
const S: usize = 64;
const W: usize = 2 * P + S;
const SB: usize = 4;
/// Physical rows of a token-history slab: spans break at its edges.
const SLAB_ROWS: usize = 64;
const WINDOW_ROWS: usize = 64;

/// The indexer's form.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    /// Qwen4exp: mean pool, pooled RMS norm, normed queries, uniform heads.
    Qwen,
    /// GLM-5.3: softmax gate with a place term, per-token LayerNorm and bias,
    /// unnormed queries, projected head weights.
    Glm,
}

/// One launch over rows at positions `first..first + rows` of one sequence.
struct Case {
    form: Form,
    /// The element the queries and head weights arrive as.
    element: Element,
    heads: usize,
    rows: usize,
    first: usize,
    /// Picked units per row.
    k: usize,
    /// Unit capacity (scores, bitmap extent): 32 a word.
    nb: usize,
    words: usize,
    /// Whether the class reads Segments (decode).
    decode: bool,
    scale: f32,
    query: Vec<f32>,
    query_norm: Vec<f32>,
    head_weights: Vec<f32>,
    /// The launch's tokens' projections `u` (in A) and gate scores `g`.
    key: Vec<f32>,
    gate: Vec<f32>,
    norm: Vec<f32>,
    bias: Vec<f32>,
    places: Vec<f32>,
    components: Vec<i32>,
    frequencies: Vec<f32>,
    /// Physical token-history row of each history position.
    history_rows: Vec<i32>,
    /// Window rows of every position before the launch (only the last
    /// SB - 1 are read) and of the launch's rows.
    window_of: Vec<i32>,
    window: Vec<f32>,
    /// Block row of every unit finished before the launch, and of the
    /// launch's own finished units.
    block_of: Vec<i32>,
    block_rows: usize,
    blocks: Vec<f32>,
}

fn coordinates(position: usize) -> [i32; 4] {
    // Distinct multi-axis coordinates, as an image row's are.
    [position as i32, position as i32 / 2 + 3, position as i32 % 7, 0]
}

impl Case {
    fn qwen(rows: usize, first: usize, k: usize, nb: usize, decode: bool, seed: u64) -> Self {
        Self::of(Form::Qwen, Element::bf16(), 4, rows, first, k, nb, decode, seed)
    }

    fn glm(rows: usize, first: usize, decode: bool, seed: u64) -> Self {
        Self::of(Form::Glm, Element::bf16(), 32, rows, first, 512, first / SB + rows, decode, seed)
    }

    #[allow(clippy::too_many_arguments)]
    fn of(form: Form, element: Element, heads: usize, rows: usize, first: usize, k: usize, nb: usize, decode: bool, seed: u64) -> Self {
        let mut random = Random(seed);
        let mut values = |count: usize, scale: f32, rounded: bool| {
            (0..count).map(|_| if rounded { bf16_round(random.next() * scale) } else { random.next() * scale }).collect::<Vec<_>>()
        };
        let gated = form == Form::Glm;
        // Queries and head weights arrive as `element`: F32 values are not
        // BF16-representable.
        let query = values(rows * heads * W, 2.0, element == Element::bf16());
        let key = values(rows * W, 1.5, true);
        let gate = if gated { values(rows * W, 2.0, true) } else { Vec::new() };
        let head_weights = if gated { values(rows * heads, 1.0, element == Element::bf16()) } else { Vec::new() };
        let query_norm = if gated { Vec::new() } else { values(W, 1.0, true).iter().map(|x| 1.0 + 0.5 * x).collect() };
        let norm = values(W, 1.0, true).iter().map(|x| 1.0 + 0.5 * x).collect();
        let bias = if gated { values(W, 0.5, true) } else { Vec::new() };
        let places = if gated { values(SB * W, 1.0, true) } else { Vec::new() };
        let components = (0..P).map(|pair| [0, 1, 2][pair % 3]).collect();
        let frequencies = (0..P).map(|pair| 10_000f32.powf(-(2.0 * pair as f32) / (2.0 * P as f32))).collect();
        // History positions in two physical runs, the first crossing a slab
        // edge.
        let history_rows = (0..first)
            .map(|position| if position < 20 { 50 + position as i32 } else { 200 + position as i32 })
            .collect();
        // Scattered window rows: the pending tokens before the launch in
        // [0, 24), the launch's in [24, 64), each set distinct.
        let window_of = (0..first + rows)
            .map(|position| match position >= first {
                false => ((position * 7) % 24) as i32,
                true => ((position * 7) % 40) as i32 + 24,
            })
            .collect::<Vec<_>>();
        // The pending tokens before the launch, as the window holds them: a
        // gated form's normed key and its gate scores beside it.
        let window_width = if gated { 2 * W } else { W };
        let mut window = vec![SENTINEL; WINDOW_ROWS * window_width];
        for position in first.saturating_sub(SB - 1)..first {
            let at = window_of[position] as usize;
            window[at * window_width..][..window_width].copy_from_slice(&values(window_width, 1.5, true));
        }
        let finished = (first + rows) / SB;
        let block_of = (0..finished)
            .map(|block| if block < 5 { block as i32 } else { 10 + block as i32 })
            .collect::<Vec<_>>();
        let block_rows = finished + 16;
        let mut blocks = vec![SENTINEL; block_rows * W];
        for block in 0..first / SB {
            let at = block_of[block] as usize;
            blocks[at * W..][..W].copy_from_slice(&values(W, 1.0, true));
        }
        let words = (nb + 1).div_ceil(32);
        let scale = if gated { 1.0 / ((heads * W) as f32).sqrt() } else { 1.0 / (W as f32).sqrt() };
        Self {
            form,
            element,
            heads,
            rows,
            first,
            k,
            nb: 32 * words,
            words,
            decode,
            scale,
            query,
            query_norm,
            head_weights,
            key,
            gate,
            norm,
            bias,
            places,
            components,
            frequencies,
            history_rows,
            window_of,
            window,
            block_of,
            block_rows,
            blocks,
        }
    }

    fn gated(&self) -> bool {
        self.form == Form::Glm
    }

    fn window_width(&self) -> usize {
        if self.gated() { 2 * W } else { W }
    }

    fn position(&self, row: usize) -> usize {
        self.first + row
    }

    /// The token history's physical runs (split at slab edges).
    fn token_runs(&self) -> Vec<[i32; 2]> {
        let mut runs: Vec<[i32; 2]> = Vec::new();
        for &at in &self.history_rows {
            match runs.last_mut() {
                Some(run) if run[1] == at && at as usize % SLAB_ROWS != 0 => run[1] += 1,
                _ => runs.push([at, at + 1]),
            }
        }
        runs
    }

    /// Per row, its Window runs (the window rows of the SB - 1 positions
    /// before it) and its Block runs (the units complete through it).
    fn read_runs(&self) -> (Vec<Vec<[i32; 2]>>, Vec<Vec<[i32; 2]>>) {
        let runs = |rows: &[i32]| {
            let mut runs: Vec<[i32; 2]> = Vec::new();
            for &at in rows {
                match runs.last_mut() {
                    Some(run) if run[1] == at => run[1] += 1,
                    _ => runs.push([at, at + 1]),
                }
            }
            runs
        };
        let positions = (0..self.rows).map(|row| self.position(row));
        (
            positions
                .clone()
                .map(|position| runs(&self.window_of[position.saturating_sub(SB - 1)..position]))
                .collect(),
            positions.map(|position| runs(&self.block_of[..(position + 1) / SB])).collect(),
        )
    }

    /// `R`: the most spans any read of a row has, as one class pads them.
    fn spans(&self) -> usize {
        let (window, block) = self.read_runs();
        [self.token_runs().len()]
            .into_iter()
            .chain(window.iter().chain(&block).map(Vec::len))
            .max()
            .unwrap_or(1)
            .max(1)
    }

    fn pad(&self, per_row: &[Vec<[i32; 2]>]) -> Vec<i32> {
        let spans = self.spans();
        per_row
            .iter()
            .flat_map(|runs| (0..spans).flat_map(|index| runs.get(index).copied().unwrap_or([0, 0])))
            .collect()
    }

    fn complete(&self) -> Vec<i32> {
        (0..self.rows).map(|row| ((self.position(row) + 1) / SB) as i32).collect()
    }
}

// ---- host model ---------------------------------------------------------------------------------

/// `x` under the RMS norm `norm` (none), rotated at `coordinates` in
/// half-split leading pairs.
fn head_rotary(x: &[f32], norm: Option<&[f32]>, coordinates: [i32; 4], case: &Case) -> Vec<f32> {
    let normed = match norm {
        Some(norm) => {
            let squares: f32 = x.iter().map(|v| v * v).sum();
            let inverse = 1.0 / (squares / W as f32 + EPSILON).sqrt();
            x.iter().zip(norm).map(|(v, w)| v * inverse * w).collect::<Vec<_>>()
        }
        None => x.to_vec(),
    };
    let mut rotated = normed.clone();
    for pair in 0..P {
        let angle = coordinates[case.components[pair] as usize] as f32 * case.frequencies[pair];
        let (s, c) = angle.sin_cos();
        rotated[pair] = normed[pair] * c - normed[pair + P] * s;
        rotated[pair + P] = normed[pair + P] * c + normed[pair] * s;
    }
    rotated
}

/// A token's key as the window holds it: GLM's under its LayerNorm and bias.
fn token_key(case: &Case, u: &[f32]) -> Vec<f32> {
    if !case.gated() {
        return u.to_vec();
    }
    let mean = u.iter().sum::<f32>() / W as f32;
    let centered = u.iter().map(|v| v - mean).collect::<Vec<_>>();
    let inverse = 1.0 / (centered.iter().map(|v| v * v).sum::<f32>() / W as f32 + EPSILON).sqrt();
    centered.iter().zip(&case.norm).zip(&case.bias).map(|((v, w), b)| v * inverse * w + b).collect()
}

struct Model {
    /// Finished unit keys (unit -> key) and the window rows the launch's
    /// tokens append.
    unit_keys: Vec<(usize, Vec<f32>)>,
    /// Each finished unit's pooled rows (a token's key, then its gate scores
    /// when gated), by the unit's first position.
    pools: Vec<(usize, Vec<Vec<f32>>)>,
    pending: Vec<Vec<f32>>,
    scores: Vec<Vec<f32>>,
    /// Each score's order spread: the most two F32 summation orders of the
    /// same terms (the dot's `W` products, then the `heads` terms) can move
    /// it, `2 (γ_W + γ_heads)` times the terms' magnitudes through `reduce`.
    spreads: Vec<Vec<f64>>,
    segments: Vec<Vec<[i32; 2]>>,
    bitmap: Vec<u32>,
}

/// The host model, its scores reduced by `reduce` (the summed relu terms to
/// the score): the contract's `scale · Σ`, or Qwen4exp's dividing `Σ / √W`.
fn model_with(case: &Case, reduce: impl Fn(f32) -> f32) -> Model {
    model_rounding(case, reduce, |v| v)
}

/// The host model with every prepared query and every unit key rounded by
/// `round` (the identity: the contract's F32).
fn model_rounding(case: &Case, reduce: impl Fn(f32) -> f32, round: impl Fn(f32) -> f32) -> Model {
    let width = case.window_width();
    // Each position's pending row (key, then gate scores) by position.
    let mut pending = vec![Vec::new(); case.first + case.rows];
    for position in case.first.saturating_sub(SB - 1)..case.first {
        let at = case.window_of[position] as usize;
        pending[position] = case.window[at * width..][..width].to_vec();
    }
    let mut appended = Vec::new();
    for row in 0..case.rows {
        let mut row_values = token_key(case, &case.key[row * W..][..W]);
        if case.gated() {
            row_values.extend_from_slice(&case.gate[row * W..][..W]);
        }
        pending[case.position(row)] = row_values.clone();
        appended.push(row_values);
    }
    let mut keys = (0..case.first / SB)
        .map(|unit| case.blocks[case.block_of[unit] as usize * W..][..W].to_vec())
        .collect::<Vec<_>>();
    let mut unit_keys = Vec::new();
    let mut pools = Vec::new();
    for row in 0..case.rows {
        let position = case.position(row);
        if (position + 1) % SB != 0 {
            continue;
        }
        let start = position / SB * SB;
        let members = &pending[start..start + SB];
        let pooled = (0..W)
            .map(|c| match case.gated() {
                false => members.iter().map(|m| m[c]).sum::<f32>() / SB as f32,
                true => {
                    let gates = (0..SB).map(|i| members[i][W + c] + case.places[i * W + c]).collect::<Vec<_>>();
                    let top = gates.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let total = gates.iter().map(|g| (g - top).exp()).sum::<f32>();
                    (0..SB).map(|i| (gates[i] - top).exp() / total * members[i][c]).sum::<f32>()
                }
            })
            .collect::<Vec<_>>();
        let norm = (!case.gated()).then_some(case.norm.as_slice());
        let key = head_rotary(&pooled, norm, coordinates(start), case).into_iter().map(&round).collect::<Vec<_>>();
        pools.push((start, members.to_vec()));
        keys.push(key.clone());
        unit_keys.push((position / SB, key));
    }
    let runs = case.token_runs();
    let mut all_scores = Vec::new();
    let mut all_spreads = Vec::new();
    let mut segments = Vec::new();
    let mut bitmap = vec![0u32; case.rows * case.words];
    for row in 0..case.rows {
        let position = case.position(row);
        let complete = (position + 1) / SB;
        let norm = (!case.gated()).then_some(case.query_norm.as_slice());
        let q = (0..case.heads)
            .map(|h| {
                let q = head_rotary(&case.query[(row * case.heads + h) * W..][..W], norm, coordinates(position), case);
                q.into_iter().map(&round).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let scores = (0..complete)
            .map(|unit| {
                reduce(
                    (0..case.heads)
                        .map(|h| {
                            let dot = q[h].iter().zip(&keys[unit]).map(|(a, b)| a * b).sum::<f32>().max(0.0);
                            if case.gated() { dot * case.head_weights[row * case.heads + h] } else { dot }
                        })
                        .sum::<f32>(),
                )
            })
            .collect::<Vec<_>>();
        let order_spread = 2.0 * (gamma(W) + gamma(case.heads));
        let spreads = (0..complete)
            .map(|unit| {
                let magnitudes = (0..case.heads)
                    .map(|h| {
                        let weight = if case.gated() { case.head_weights[row * case.heads + h].abs() } else { 1.0 };
                        weight * q[h].iter().zip(&keys[unit]).map(|(a, b)| (a * b).abs()).sum::<f32>()
                    })
                    .sum::<f32>();
                order_spread * f64::from(reduce(magnitudes))
            })
            .collect::<Vec<_>>();
        let mut read = vec![true; case.nb + 1];
        if complete > case.k {
            let mut order = (0..complete).collect::<Vec<_>>();
            order.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap().then(a.cmp(b)));
            read[..complete].fill(false);
            for &unit in &order[..case.k] {
                read[unit] = true;
            }
        }
        for word in 0..case.words {
            bitmap[row * case.words + word] =
                (0..32).filter(|bit| read[(word * 32 + bit).min(case.nb)]).fold(0, |w, bit| w | 1 << bit);
        }
        // The read keys of the visible spans, as maximal runs within each.
        let mut row_runs: Vec<[i32; 2]> = Vec::new();
        let mut key = 0;
        for &[lo, hi] in &runs {
            let mut open = false;
            for stored in lo..hi {
                if read[(key / SB).min(case.nb)] {
                    match row_runs.last_mut() {
                        Some(run) if open => run[1] = stored + 1,
                        _ => row_runs.push([stored, stored + 1]),
                    }
                    open = true;
                } else {
                    open = false;
                }
                key += 1;
            }
        }
        segments.push(row_runs);
        all_scores.push(scores);
        all_spreads.push(spreads);
    }
    Model { unit_keys, pools, pending: appended, scores: all_scores, spreads: all_spreads, segments, bitmap }
}

/// The contract's scores: the summed terms times f32 `scale`.
fn model(case: &Case) -> Model {
    model_with(case, |sum| case.scale * sum)
}

// ---- portable bodies -----------------------------------------------------------------------------

/// The chain through its entries' portable bodies, each fed the one before's
/// results: the same launch as [`native`]'s.
fn portable(case: &Case) -> Outcome {
    use indexer_common::{Input, after, interpret, interpret_with};
    use seismic_lang::{entry::ElementBindings, registry, types::DType};
    let m = case.rows;
    let gt = usize::from(case.gated());
    let (pr, tn) = if case.gated() { (SB, 1) } else { (0, 0) };
    let spans = case.spans();
    let (window_runs, block_runs) = case.read_runs();
    let positions = (0..m).map(|row| case.position(row)).collect::<Vec<_>>();
    let coordinate_rows = positions.iter().flat_map(|&p| coordinates(p)).collect::<Vec<_>>();
    let mut coordinate_words = vec![-7; WINDOW_ROWS * 4];
    for position in case.first.saturating_sub(SB - 1)..case.first {
        coordinate_words[case.window_of[position] as usize * 4..][..4].copy_from_slice(&coordinates(position));
    }
    let amplitudes = vec![1.0; P];
    let bf16_bindings = ElementBindings::new().bind("A", registry::dense(DType::BF16)).bind("Y", registry::dense(DType::F32));
    let (_, planes) = interpret_with(
        "pool_rows",
        &bf16_bindings,
        vec![
            Input::F32(vec![m, W], case.key.clone()),
            Input::F32(vec![m, gt * W], case.gate.clone()),
            Input::F32(vec![pr, W], case.places.clone()),
            Input::F32(vec![1, W], case.norm.clone()),
            Input::F32(vec![tn, W], case.bias.clone()),
            Input::I32(vec![P], case.components.clone()),
            Input::F32(vec![P], case.frequencies.clone()),
            Input::F32(vec![P], amplitudes.clone()),
            Input::I32(vec![m, 4], coordinate_rows.clone()),
            Input::I32(vec![m, spans, 2], case.pad(&window_runs)),
            Input::I32(vec![m], positions.iter().map(|&p| case.window_of[p]).collect()),
            Input::I32(
                vec![m],
                positions.iter().map(|&p| if (p + 1) % SB == 0 { case.block_of[p / SB] } else { -1 }).collect(),
            ),
            Input::F32(vec![WINDOW_ROWS, case.window_width()], case.window.clone()),
            Input::I32(vec![WINDOW_ROWS, 4], coordinate_words),
            Input::F32(vec![case.block_rows, W], case.blocks.clone()),
            Input::I32(vec![0, 0, 0, SB, gt], Vec::new()),
            Input::ScalarF32(EPSILON),
            Input::ScalarU32(WINDOW_ROWS as u32),
            Input::ScalarU32(case.block_rows as u32),
            Input::ScalarI32(0),
            Input::ScalarI32(0),
            Input::ScalarI32(0),
            Input::ScalarF32(0.0),
        ],
    );
    let window = after(&planes, 12).iter().map(|v| *v as f32).collect::<Vec<_>>();
    let blocks = after(&planes, 14).iter().map(|v| *v as f32).collect::<Vec<_>>();
    let (nq, hw) = if case.gated() { (0, case.heads) } else { (1, 0) };
    let (scores, _) = interpret(
        "indexer_score",
        true,
        vec![
            Input::Bf16(vec![m, case.heads, W], case.query.clone()),
            Input::F32(vec![nq, W], case.query_norm.clone()),
            Input::Bf16(vec![m, hw], case.head_weights.clone()),
            Input::I32(vec![P], case.components.clone()),
            Input::F32(vec![P], case.frequencies.clone()),
            Input::F32(vec![P], amplitudes),
            Input::I32(vec![m, 4], coordinate_rows),
            Input::F32(vec![case.block_rows, W], blocks.clone()),
            Input::I32(vec![m, spans, 2], case.pad(&block_runs)),
            Input::I32(vec![0, case.nb], Vec::new()),
            Input::I32(vec![0, 0, 0], Vec::new()),
            Input::ScalarF32(case.scale),
            Input::ScalarF32(EPSILON),
            Input::ScalarI32(0),
            Input::ScalarI32(0),
            Input::ScalarI32(0),
            Input::ScalarF32(0.0),
            Input::ScalarU32(case.block_rows as u32),
        ],
    );
    let scores = scores[0].iter().map(|v| *v as f32).collect::<Vec<_>>();
    let complete = case.complete();
    let (picks, _) = interpret(
        "rows_topk",
        false,
        vec![
            Input::F32(vec![m, case.nb], scores),
            Input::I32(vec![m], complete.clone()),
            Input::I32(vec![m, 1, 2], complete.iter().flat_map(|&c| [0, c]).collect()),
            Input::I32(vec![0, case.k], Vec::new()),
        ],
    );
    let picks = picks[0].iter().map(|v| *v as i32).collect::<Vec<_>>();
    let (reads, _) = interpret(
        "unit_reads",
        false,
        vec![
            Input::I32(vec![m, case.k], picks),
            Input::I32(vec![m], complete),
            Input::U32(vec![0, case.words], Vec::new()),
        ],
    );
    let bitmap = reads[0].iter().map(|v| *v as u32).collect();
    let segments = match case.decode {
        false => Vec::new(),
        true => {
            let token = case.token_runs();
            let visible = (0..m).map(|_| token.clone()).collect::<Vec<_>>();
            let (table, _) = interpret(
                "candidate_rows",
                false,
                vec![
                    Input::I32(vec![m, case.k + 1], reads[1].iter().map(|v| *v as i32).collect()),
                    Input::I32(vec![m, spans, 2], case.pad(&visible)),
                    Input::I32(vec![0, SB, 0], Vec::new()),
                ],
            );
            table[0]
                .chunks_exact((case.k + 1 + spans) * 2)
                .map(|row| {
                    row.chunks_exact(2).map(|run| [run[0] as i32, run[1] as i32]).take_while(|run| run[0] >= 0).collect()
                })
                .collect()
        }
    };
    Outcome { bitmap, segments, window, blocks }
}

// ---- natives -------------------------------------------------------------------------------------

struct Outcome {
    bitmap: Vec<u32>,
    segments: Vec<Vec<[i32; 2]>>,
    window: Vec<f32>,
    blocks: Vec<f32>,
}

/// An entry's kernel at `statics` on `device`.
macro_rules! kernel {
    ($device:expr, $entry:expr, $statics:expr $(,)?) => {{
        let entry = $entry;
        let statics: &[(&'static str, u64)] = $statics;
        let specialization =
            statics.iter().fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(*name, *value));
        let specialization = entry.offered($device).unwrap().unwrap().default_specialization(&specialization).unwrap();
        entry.prepare($device, &specialization).unwrap()
    }};
}

fn words(tensor: &Tensor) -> Vec<i32> {
    tensor.read_to_host().unwrap().chunks_exact(4).map(|word| i32::from_le_bytes(word.try_into().unwrap())).collect()
}

/// The rotary statics every entry of the chain takes (half-split leading).
const ROTARY: [(&str, u64); 4] = [("P", P as u64), ("S", S as u64), ("RA", 0), ("RT", 0)];

/// `pool_rows`' entry and statics for `case`: GLM's keys normed per token
/// (`TN`) and gated by place (`PR`).
fn pool_entry(case: &Case) -> (seismic::BoundEntry<pool_rows::Entry>, Vec<(&'static str, u64)>) {
    let (gt, pr, tn) = if case.gated() { (1, SB as u64, 1) } else { (0, 0, 0) };
    (
        pool_rows::native_entry_with(pool_rows::Elements { A: Element::bf16(), Y: Element::f32() }),
        [&ROTARY[..], &[("GT", gt), ("PR", pr), ("TN", tn), ("RATE", SB as u64)]].concat(),
    )
}

fn native(device: &Device, case: &Case) -> Outcome {
    let m = case.rows as u64;
    let (h, w) = (case.heads as u64, W as u64);
    let gt = u64::from(case.gated());
    let (pr, tn) = if case.gated() { (SB as u64, 1) } else { (0, 0) };
    let spans = case.spans();
    let r = spans as u64;
    let (window_runs, block_runs) = case.read_runs();
    let positions = (0..case.rows).map(|row| case.position(row)).collect::<Vec<_>>();
    let coordinates = ints(device, &[m, 4], &positions.iter().flat_map(|&p| coordinates(p)).collect::<Vec<_>>());
    let plane = |rows: usize, element: Element, width: usize, values: &[f32]| {
        let (slabs, logical) = slab_arenas(
            device,
            rows,
            rows as u64,
            &[Arena { element, bank_shape: vec![width as u64], values }],
            false,
        );
        (slabs, logical.into_iter().next().unwrap())
    };
    let (_window_slabs, mut window) = plane(WINDOW_ROWS, Element::f32(), case.window_width(), &case.window);
    let mut coordinate_words = vec![-7; WINDOW_ROWS * 4];
    for position in case.first.saturating_sub(SB - 1)..case.first {
        coordinate_words[case.window_of[position] as usize * 4..][..4].copy_from_slice(&self::coordinates(position));
    }
    // 32-bit words travel as the bits of f32 values (`bytes`).
    let coordinate_bits = coordinate_words.iter().map(|value| f32::from_bits(*value as u32)).collect::<Vec<_>>();
    let (_coordinate_slabs, mut window_coordinates) = plane(WINDOW_ROWS, Element::i32(), 4, &coordinate_bits);
    let (_block_slabs, mut blocks) = plane(case.block_rows, Element::f32(), W, &case.blocks);
    let (pool_entry, pool_statics) = pool_entry(case);
    let pool = kernel!(device, pool_entry, &pool_statics);
    let components = ints(device, &[P as u64], &case.components);
    let frequencies = tensor(device, Element::f32(), &[P as u64], &case.frequencies);
    let amplitudes = tensor(device, Element::f32(), &[P as u64], &vec![1.0; P]);
    pool.call(pool_rows::Args {
        u: &tensor(device, Element::f32(), &[m, w], &case.key),
        g: &tensor(device, Element::f32(), &[m, gt * w], &case.gate),
        position: &tensor(device, Element::f32(), &[pr, w], &case.places),
        norm: &tensor(device, Element::f32(), &[1, w], &case.norm),
        bias: &tensor(device, Element::f32(), &[tn, w], &case.bias),
        rotary_components: &components,
        rotary_frequencies: &frequencies,
        rotary_amplitudes: &amplitudes,
        coordinates: &coordinates,
        window_visible: &ints(device, &[m, r, 2], &case.pad(&window_runs)),
        window_destinations: &ints(device, &[m], &positions.iter().map(|&p| case.window_of[p]).collect::<Vec<_>>()),
        block_destinations: &ints(
            device,
            &[m],
            &positions.iter().map(|&p| if (p + 1) % SB == 0 { case.block_of[p / SB] } else { -1 }).collect::<Vec<_>>(),
        ),
        window_rows: &mut window,
        window_coordinates: &mut window_coordinates,
        block_rows: &mut blocks,
        form: &ints(device, &[0, 0, 0, SB as u64, gt], &[]),
        epsilon: EPSILON,
        window_slab_rows: WINDOW_ROWS as u32,
        block_slab_rows: case.block_rows as u32,
        block: 0,
        element: 0,
        scale: 0,
        amax_floor: 0.0,
    })
    .unwrap();
    let (nq, hw) = if case.gated() { (0, h) } else { (1, 0) };
    let score = kernel!(
        device,
        indexer_score::native_entry_with(indexer_score::Elements { A: case.element }),
        &[&ROTARY[..], &[("H", h), ("NQ", nq), ("HW", hw)]].concat(),
    );
    let n = case.nb as u64;
    let scores = score
        .call(indexer_score::Args {
            query: &tensor(device, case.element, &[m, h, w], &case.query),
            query_norm: &tensor(device, Element::f32(), &[nq, w], &case.query_norm),
            head_weights: &tensor(device, case.element, &[m, hw], &case.head_weights),
            rotary_components: &components,
            rotary_frequencies: &frequencies,
            rotary_amplitudes: &amplitudes,
            coordinates: &coordinates,
            keys: &blocks,
            spans: &ints(device, &[m, r, 2], &case.pad(&block_runs)),
            score_slots: &ints(device, &[0, n], &[]),
            form: &ints(device, &[0, 0, 0], &[]),
            scale: case.scale,
            epsilon: EPSILON,
            block: 0,
            element: 0,
            rounding_scale: 0,
            amax_floor: 0.0,
            slab_rows: case.block_rows as u32,
        })
        .unwrap()
        .value;
    let complete = ints(device, &[m], &case.complete());
    let units = ints(device, &[m, 1, 2], &case.complete().iter().flat_map(|&c| [0, c]).collect::<Vec<_>>());
    let picks = kernel!(device, rows_topk::native_entry(), &[])
        .call(rows_topk::Args {
            scores: &scores,
            complete: &complete,
            spans: &units,
            top_slots: &ints(device, &[0, case.k as u64], &[]),
        })
        .unwrap()
        .value;
    let reads = kernel!(device, unit_reads::native_entry(), &[])
        .call(unit_reads::Args {
            picks: &picks,
            complete: &complete,
            bitmap_words: &Tensor::from_host(device, Element::u32(), &[0, case.words as u64], &[]).unwrap(),
        })
        .unwrap();
    let segments = match case.decode {
        false => Vec::new(),
        true => {
            let token = case.token_runs();
            let visible = (0..case.rows).map(|_| token.clone()).collect::<Vec<_>>();
            let table = kernel!(device, candidate_rows::native_entry(), &[])
                .call(candidate_rows::Args {
                    blocks: &reads.r1,
                    spans: &ints(device, &[m, r, 2], &case.pad(&visible)),
                    form: &ints(device, &[0, SB as u64, 0], &[]),
                })
                .unwrap()
                .value;
            words(&table)
                .chunks_exact((case.k + 1 + spans) * 2)
                .map(|row| row.chunks_exact(2).map(|run| [run[0], run[1]]).take_while(|run| run[0] >= 0).collect())
                .collect()
        }
    };
    Outcome {
        bitmap: words(&reads.r0).into_iter().map(|word| word as u32).collect(),
        segments,
        window: read(&window),
        blocks: read(&blocks),
    }
}

// ---- bounds --------------------------------------------------------------------------------------
//
// How far a backend's F32 arithmetic may be from the host model's, element by
// element: each is a rounding analysis of the operations both perform
// (`common::bound`), the host's own library being `Library::of(Cpu)`.

/// The absolute error of a backend's sine or cosine of an F32 angle, values
/// in [-1, 1]. CPU and CUDA evaluate the library's `sinf` and `cosf` (1 and 2
/// ulp). Metal's `fast::sin` and Vulkan's `sin` document no absolute error
/// tighter than Vulkan's 2^-11 on [-pi, pi], where both are evaluated after
/// the kernel's exact angle reduction.
fn trig(backend: BackendName) -> f64 {
    let ulp = 2.0 * Float::F32.unit();
    match backend {
        BackendName::Cpu => ulp,
        BackendName::Cuda => 2.0 * ulp,
        BackendName::Metal | BackendName::Vulkan => 2f64.powi(-11),
    }
}

/// Native versus host for a row under `token_norm`
/// ([`bound::layer_norm_between`]).
fn token_norm_bound(case: &Case, u: &[f32], backend: BackendName) -> Vec<f64> {
    bound::layer_norm_between((&Library::of(backend), &Library::of(BackendName::Cpu)), u, &case.norm, &case.bias, EPSILON)
}

/// Native versus host for a unit's finished key: its `SB` pooled rows
/// (`members`, each with its own error `errors`), pooled by the softmax gate
/// (`exp`, the division and the sum) or the mean, normed by `head_norm` when
/// not gated, and rotated at the unit's first coordinates.
fn unit_key_bound(
    case: &Case,
    members: &[Vec<f32>],
    errors: &[Vec<f64>],
    backend: BackendName,
) -> Vec<f64> {
    let (native, host) = (Library::of(backend), Library::of(BackendName::Cpu));
    let (unit, sums) = (Float::F32.unit(), gamma(2 * SB));
    let pooled = (0..W)
        .map(|c| {
            let values = members.iter().map(|m| f64::from(m[c])).collect::<Vec<_>>();
            if case.gated() {
                // Both add the same F32 place terms and subtract the same top,
                // so the exponentials' arguments are identical.
                let gates = (0..SB).map(|i| f64::from(members[i][W + c] + case.places[i * W + c])).collect::<Vec<_>>();
                let top = gates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps = gates.iter().map(|g| (g - top).exp()).collect::<Vec<_>>();
                let total = exps.iter().sum::<f64>();
                let exp = native.exp + host.exp;
                let weight = (1.0 + exp).powi(2) * (1.0 + sums) * (1.0 + native.divide + host.divide) - 1.0;
                let products = (0..SB).map(|i| exps[i] / total * values[i].abs()).sum::<f64>();
                let inherited = (0..SB).map(|i| exps[i] / total * errors[i][c]).sum::<f64>();
                (values.iter().zip(&exps).map(|(v, e)| e / total * v).sum::<f64>(), (weight + 2.0 * unit + sums) * products + inherited)
            } else {
                let mean = values.iter().sum::<f64>() / SB as f64;
                let magnitude = values.iter().map(|v| v.abs()).sum::<f64>() / SB as f64;
                (mean, sums * magnitude + (native.divide + host.divide) * mean.abs())
            }
        })
        .collect::<Vec<_>>();
    let normed = if case.gated() {
        pooled.clone()
    } else {
        let squares = pooled.iter().map(|(p, _)| p * p).sum::<f64>();
        let inverse = 1.0 / (squares / W as f64 + f64::from(EPSILON)).sqrt();
        let shift = pooled.iter().map(|(p, e)| p.abs() * e).sum::<f64>() / squares;
        let relative = native.rms(W) + host.rms(W) + shift;
        (0..W)
            .map(|c| {
                let weight = f64::from(case.norm[c]).abs();
                (pooled[c].0 * inverse * f64::from(case.norm[c]), pooled[c].0.abs() * inverse * weight * relative + pooled[c].1 * inverse * weight)
            })
            .collect()
    };
    let tolerance = trig(backend) + trig(BackendName::Cpu);
    let rotated = |lo: usize, hi: usize| {
        let reach = normed[lo].0.abs() + normed[hi].0.abs();
        normed[lo].1 + normed[hi].1 + reach * (tolerance + 6.0 * unit)
    };
    // The rotated pairs are the leading half-split 2P columns; the rest pass through.
    (0..W)
        .map(|c| match c {
            c if c < P => rotated(c, c + P),
            c if c < 2 * P => rotated(c - P, c),
            c => normed[c].1,
        })
        .collect()
}

fn check(case: &Case, label: &str) {
    let expected = model(case);
    // The interpreted chain costs rows × units × heads × W multiplications.
    if case.element == Element::bf16() && case.rows * case.complete().iter().max().copied().unwrap_or(0) as usize * case.heads * W <= 2_000_000 {
        compare(case, &expected, &portable(case), "portable", BackendName::Cpu, label);
    }
    for device in devices() {
        compare(case, &expected, &native(&device, case), device.backend().as_str(), device.backend(), label);
    }
}

/// `outcome` of `who` (running on `backend`'s arithmetic) against the host
/// model `expected`: the picks exactly, the keys within their bounds.
fn compare(case: &Case, expected: &Model, outcome: &Outcome, who: &str, backend: BackendName, label: &str) {
    let width = case.window_width();
    assert_eq!(outcome.bitmap, expected.bitmap, "{who} {label}: bitmap");
    if case.decode {
        assert_eq!(outcome.segments, expected.segments, "{who} {label}: segments");
    }
    for (row, appended) in expected.pending.iter().enumerate() {
        let at = case.window_of[case.position(row)] as usize;
        let stored = &outcome.window[at * width..][..width];
        // A per-token norm is computed in F32 arithmetic order; the rest
        // are copies.
        let mut bound = vec![0.0; width];
        if case.gated() {
            bound[..W].copy_from_slice(&token_norm_bound(case, &case.key[row * W..][..W], backend));
        }
        assert_within(&format!("{who} {label}: row {row}'s pending key"), stored, appended, &bound);
    }
    for ((unit, key), (start, members)) in expected.unit_keys.iter().zip(&expected.pools) {
        let at = case.block_of[*unit] as usize;
        let stored = &outcome.blocks[at * W..][..W];
        let errors = (*start..start + SB)
            .map(|position| match case.gated() && position >= case.first {
                true => token_norm_bound(case, &case.key[(position - case.first) * W..][..W], backend),
                false => vec![0.0; W],
            })
            .collect::<Vec<_>>();
        let bound = unit_key_bound(case, members, &errors, backend);
        assert_within(&format!("{who} {label}: unit {unit}'s key"), stored, key, &bound);
    }
}

/// Whether every selecting row's K-th and (K + 1)-th scores lie farther
/// apart than the backends' F32 summation orders move them: more than the
/// two scores' order spreads together.
fn robust(case: &Case) -> bool {
    let model = model(case);
    model.scores.iter().zip(&model.spreads).all(|(scores, spreads)| {
        let mut order = (0..scores.len()).collect::<Vec<_>>();
        order.sort_by(|a, b| scores[*b].partial_cmp(&scores[*a]).unwrap());
        order.len() <= case.k
            || f64::from(scores[order[case.k - 1]] - scores[order[case.k]]) > spreads[order[case.k - 1]] + spreads[order[case.k]]
    })
}

/// The first of 64 seeds from `seed` whose case is robust.
fn first_robust(make: impl Fn(u64) -> Case, seed: u64) -> Case {
    (seed..seed + 64).map(make).find(robust).expect("a robust case")
}

/// A Qwen4exp case through the chain (its first robust seed's): its picks
/// equal the dividing order's.
fn qwen(make: impl Fn(u64) -> Case, seed: u64, label: &str) {
    let case = first_robust(make, seed);
    let dividing = model_with(&case, |sum| sum / (W as f32).sqrt());
    assert_eq!(model(&case).bitmap, dividing.bitmap, "{label}: scale · Σ and Σ / √W select alike");
    check(&case, label);
}

#[test]
fn decode_rows_select_segments() {
    // Two rows past the dense limit, the second finishing unit 10.
    qwen(|seed| Case::qwen(2, 42, 4, 16, true, seed), 1, "decode");
    // One row whose unit is complete at its own position.
    qwen(|seed| Case::qwen(1, 47, 4, 16, true, seed), 2, "decode own unit");
    // A long row: its keys, bitmap words and runs spread over every thread.
    qwen(|seed| Case::qwen(1, 1500, 64, 400, true, seed), 9, "decode long");
}

#[test]
fn dense_rows_read_every_visible_row() {
    qwen(|seed| Case::qwen(2, 42, 16, 16, true, seed), 3, "dense decode");
    qwen(|seed| Case::qwen(16, 20, 16, 16, false, seed), 4, "dense prefill");
}

#[test]
fn prefill_rows_select_a_bitmap() {
    // Sixteen rows finishing four units, mixing dense and selecting rows.
    qwen(|seed| Case::qwen(16, 12, 5, 16, false, seed), 5, "prefill mixed");
    qwen(|seed| Case::qwen(16, 36, 4, 16, false, seed), 6, "prefill");
}

#[test]
fn glm_rows_select_units_past_2048_keys() {
    // GLM-5.3's indexer at its shapes: decode rows and a prefill launch
    // past 2048 keys (over 512 complete units), the first robust seed's.
    check(&first_robust(|seed| Case::glm(3, 2150, true, seed), 40), "glm decode");
    check(&first_robust(|seed| Case::glm(16, 2100, false, seed), 140), "glm prefill");
}

#[test]
fn f32_indexer_selects_the_reference_blocks_of_near_ties() {
    // A case whose near-tie scores an indexer rounding its prepared queries
    // and unit keys to BF16 would order differently, while every selecting
    // row's K-th and (K + 1)-th F32 scores lie farther apart than F32
    // summation order moves them: the native chain's picks are the F32 ones.
    let rounded = |v: f32| bf16_round(v);
    let case = (0..400)
        .map(|seed| Case::of(Form::Qwen, Element::bf16(), 8, 40, 3, 4, 16, false, 100 + seed))
        .find(|case| {
            let reference = model(case);
            let rounding = model_rounding(case, |sum| case.scale * sum, rounded);
            reference.bitmap != rounding.bitmap && robust(case)
        })
        .expect("a case whose BF16 rounding reorders its near ties");
    check(&case, "near ties");
}

#[test]
fn f32_inputs_select_as_the_model_does() {
    // Queries and head weights that BF16 cannot hold, through both forms.
    check(&first_robust(|seed| Case::of(Form::Qwen, Element::f32(), 4, 2, 42, 4, 16, true, seed), 8), "f32 qwen decode");
    check(&first_robust(|seed| Case::of(Form::Glm, Element::f32(), 32, 3, 2150, 512, 2150 / SB + 3, true, seed), 80), "f32 glm decode");
}

#[test]
fn glm_rows_select_through_the_portable_chain() {
    // GLM's form at a size the interpreted chain affords: decode and prefill
    // rows selecting 4 of 10 units.
    check(&first_robust(|seed| Case::of(Form::Glm, Element::bf16(), 4, 2, 41, 4, 16, true, seed), 3), "glm small decode");
    check(&first_robust(|seed| Case::of(Form::Glm, Element::bf16(), 4, 8, 37, 4, 16, false, seed), 11), "glm small prefill");
}
