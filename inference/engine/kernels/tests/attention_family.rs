// Conformance of the `attention_*` family's natives with its portable body
// over every axis of the family (attention.seismic): interleaved, separate
// per-head and per-column gates (sigmoid and softplus) and ungated heads;
// q/k norms and none; value norms; rotary tables with amplitudes, partial
// rotation and none (P = 0); layers without fresh rows; non-causal fresh
// blocks without appends; sliding windows smaller than a batch; visible spans
// split at slab edges; head widths to 512 with groups to 16; decode and
// prefill row classes.
//
// A host model of the portable body is checked against the reference
// interpreter at small shapes and stands for it at the others. Natives run on
// the backends named by ATTENTION_BACKENDS (a comma list of metal, cuda,
// vulkan and cpu; default: every backend the host opens).
#![allow(dead_code)]

mod common;
mod rounding_model;

use magnitude_kernels::{attention_append_dense, attention_decode, attention_prefill};
use rounding_model::Rounding;
use seismic::{
    BackendName, Device, Element, NativeSpecialization, SlabRegion, SlabTensor,
    Tensor,
};

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn bf16(value: f32) -> f32 {
    f32::from_bits(u32::from(bf16_bits(value)) << 16)
}

/// Deterministic values in [-1, 1).
struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// The structural axes of one attention layer.
#[derive(Clone, Copy, Debug)]
struct Form {
    kv: usize,
    g: usize,
    p: usize,
    s: usize,
    /// Gates interleaved after each query head's W columns.
    interleaved: bool,
    /// Separate gates per query head (0, 1 or W).
    separate: usize,
    softplus: bool,
    fresh: bool,
    /// A query norm (`NQ = 1`).
    query_norm: bool,
    /// A key norm (`N = 1`; a layer without fresh rows takes the query's).
    norm: bool,
    value_norm: bool,
    rotary: Rotary,
    /// Values are the stored key row (`VK = 1`): no value input, value norm
    /// or value plane.
    stored_key: bool,
}

/// The rotary layout and output rotary (`RA`, `RT`, `RO`).
#[derive(Clone, Copy, Debug)]
struct Rotary {
    adjacent: bool,
    trailing: bool,
    inverse_output: bool,
}

impl Rotary {
    const HALF_SPLIT_LEADING: Self = Self {
        adjacent: false,
        trailing: false,
        inverse_output: false,
    };
    /// DeepSeek V4.1's: adjacent pairs over the trailing columns, the
    /// attended row turned back.
    const ADJACENT_TRAILING_INVERSE: Self = Self {
        adjacent: true,
        trailing: true,
        inverse_output: true,
    };

    /// The (lo, hi) columns of pair `pair` of `p` in a `w`-column row.
    fn columns(self, pair: usize, p: usize, w: usize) -> (usize, usize) {
        let first = if self.trailing { w - 2 * p } else { 0 };
        if self.adjacent {
            (first + 2 * pair, first + 2 * pair + 1)
        } else {
            (first + pair, first + pair + p)
        }
    }
}

impl Form {
    fn w(&self) -> usize {
        2 * self.p + self.s
    }
    fn heads(&self) -> usize {
        self.kv * self.g
    }
    fn i(&self) -> usize {
        if self.interleaved { self.w() } else { 0 }
    }
    /// The gate values of one query head.
    fn gates(&self) -> usize {
        self.i().max(self.separate)
    }
    /// The extents of the entries' empty `form` table: 0, RA, RT, RO, VK.
    fn extents(&self) -> [usize; 5] {
        [
            0,
            usize::from(self.rotary.adjacent),
            usize::from(self.rotary.trailing),
            usize::from(self.rotary.inverse_output),
            usize::from(self.stored_key),
        ]
    }
    /// The layers of the entries' value input: none for stored-key values.
    fn value_layers(&self) -> usize {
        usize::from(self.fresh && !self.stored_key)
    }
    /// The value plane's rows of a `rows`-row history: none for stored-key
    /// values.
    fn value_rows(&self, rows: usize) -> usize {
        if self.stored_key { 0 } else { rows }
    }
}

/// A row's controls: visible history spans, its fresh span over the batch
/// rows, its destination and position.
#[derive(Clone)]
struct Row {
    spans: Vec<(i32, i32)>,
    fresh: (i32, i32),
    destination: i32,
    position: i32,
}

/// One invocation: rows, their controls, projections, weights and history.
#[derive(Clone)]
struct Case {
    form: Form,
    rows: usize,
    history_rows: usize,
    spans: usize,
    slab_rows: usize,
    query: Vec<f32>,
    gate: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    value_norm: Vec<f32>,
    /// One sink logit per query head (`SK = 1`), or none (`SK = 0`).
    sinks: Vec<f32>,
    components: Vec<i32>,
    frequencies: Vec<f32>,
    amplitudes: Vec<f32>,
    coordinates: Vec<i32>,
    visible: Vec<i32>,
    fresh: Vec<i32>,
    destinations: Vec<i32>,
    history_key: Vec<f32>,
    history_value: Vec<f32>,
    epsilon: f32,
    scale: f32,
    /// The index-list source (`KS` picks a row, -1 none) over its plane of
    /// `selected_rows_total` rows (`TS`), slabs of `selected_slab_rows`.
    picks: usize,
    selected_rows: Vec<i32>,
    selected: Vec<f32>,
    selected_slab_rows: usize,
    /// The trained rounding of the prepared key (`KR = 1`), or none.
    key_rounding: Option<Rounding>,
}

/// Splits each span at slab edges: a history span never crosses one.
fn split_at_slabs(rows: Vec<Row>, slab_rows: usize) -> Vec<Row> {
    let slab = slab_rows as i32;
    rows.into_iter()
        .map(|row| Row {
            spans: row
                .spans
                .iter()
                .flat_map(|&(lo, hi)| {
                    let mut pieces = Vec::new();
                    let mut at = lo;
                    while at < hi {
                        let end = hi.min((at / slab + 1) * slab);
                        pieces.push((at, end));
                        at = end;
                    }
                    pieces
                })
                .collect(),
            ..row
        })
        .collect()
}

impl Case {
    #[allow(clippy::too_many_arguments)]
    fn new(
        form: Form,
        history_rows: usize,
        rows: Vec<Row>,
        slab_rows: usize,
        amplitude: f32,
        scale: f32,
        seed: u64,
    ) -> Self {
        let rows = split_at_slabs(rows, slab_rows);
        let spans = rows.iter().map(|row| row.spans.len()).max().unwrap().max(1);
        let history_rows = history_rows.div_ceil(slab_rows) * slab_rows;
        let mut noise = Noise(seed);
        let (w, heads, m) = (form.w(), form.heads(), rows.len());
        let mut values = |count: usize, scale: f32| {
            (0..count)
                .map(|_| bf16(noise.next() * scale))
                .collect::<Vec<_>>()
        };
        let query = values(m * heads * (w + form.i()), 2.0);
        let gate = values(m * heads * form.separate, 2.0);
        let key = values(if form.fresh { m * form.kv * w } else { 0 }, 2.0);
        let value = values(form.value_layers() * m * form.kv * w, 1.0);
        let history_key = values(history_rows * form.kv * w, 3.0);
        let history_value = values(form.value_rows(history_rows) * form.kv * w, 1.0);
        let visible = rows
            .iter()
            .flat_map(|row| {
                row.spans
                    .iter()
                    .copied()
                    .chain(std::iter::repeat((0, 0)))
                    .take(spans)
                    .flat_map(|(lo, hi)| [lo, hi])
                    .collect::<Vec<_>>()
            })
            .collect();
        Self {
            form,
            rows: m,
            history_rows,
            spans,
            slab_rows,
            query,
            gate,
            key,
            value,
            query_norm: (0..w).map(|i| 0.6 + (i % 7) as f32 * 0.1).collect(),
            key_norm: (0..w).map(|i| 1.3 - (i % 5) as f32 * 0.1).collect(),
            value_norm: vec![1.0; w],
            sinks: Vec::new(),
            // Three M-RoPE axes over the pairs (axis p % 3), a rotary base of
            // 1e6, and the amplitude on every pair.
            components: (0..form.p).map(|pair| (pair % 3) as i32).collect(),
            frequencies: (0..form.p)
                .map(|pair| 1.0e6f64.powf(-((2 * pair) as f64) / (2 * form.p.max(1)) as f64) as f32)
                .collect(),
            amplitudes: vec![amplitude; form.p],
            coordinates: rows
                .iter()
                .flat_map(|row| [row.position, row.position + 3, row.position / 2, 0])
                .collect(),
            visible,
            fresh: rows
                .iter()
                .flat_map(|row| [row.fresh.0, row.fresh.1])
                .collect(),
            destinations: rows.iter().map(|row| row.destination).collect(),
            history_key,
            history_value,
            epsilon: 1.0e-6,
            scale,
            picks: 0,
            selected_rows: Vec::new(),
            selected: Vec::new(),
            selected_slab_rows: 1,
            key_rounding: None,
        }
    }

    /// This case with an index-list source: `picks` picks a row over a
    /// plane of `rows` rows in slabs of `slab_rows`. Every seventh pick is -1
    /// (none), and the picks of a row repeat no plane row (17 is coprime to
    /// `rows`).
    fn with_picks(mut self, picks: usize, rows: usize, slab_rows: usize, seed: u64) -> Self {
        assert!(picks <= rows && rows % 17 != 0);
        let mut noise = Noise(seed);
        let width = self.form.kv * self.form.w();
        self.selected = (0..rows * width).map(|_| bf16(noise.next() * 3.0)).collect();
        self.selected_rows = (0..self.rows)
            .flat_map(|row| {
                (0..picks).map(move |pick| {
                    if (row + pick) % 7 == 3 {
                        -1
                    } else {
                        ((row * 131 + pick * 17) % rows) as i32
                    }
                })
            })
            .collect();
        self.picks = picks;
        self.selected_slab_rows = slab_rows;
        self
    }

    /// This case with its prepared keys rounded by `rounding` (`KR = 1`).
    fn with_key_rounding(mut self, rounding: Rounding) -> Self {
        assert!(self.form.stored_key, "key rounding is stored-key only");
        self.key_rounding = Some(rounding);
        self
    }

    /// The index-list plane's rows (`TS`).
    fn selected_plane_rows(&self) -> usize {
        self.selected.len() / (self.form.kv * self.form.w())
    }

    /// The extents of the entries' empty `form` table: the layer form's,
    /// then KR.
    fn extents(&self) -> [usize; 6] {
        let [zero, ra, rt, ro, vk] = self.form.extents();
        [zero, ra, rt, ro, vk, usize::from(self.key_rounding.is_some())]
    }

    /// The key-rounding scalars (block 0 rounds nothing).
    fn rounding_scalars(&self) -> Rounding {
        self.key_rounding.unwrap_or(rounding_model::NONE)
    }

    /// The picks of `row` that name a plane row, each as that row's
    /// `kv_head` head.
    fn picked(&self, row: usize, kv_head: usize) -> impl Iterator<Item = &[f32]> {
        let (kv, w) = (self.form.kv, self.form.w());
        self.selected_rows[row * self.picks..][..self.picks]
            .iter()
            .filter(|pick| **pick >= 0)
            .map(move |pick| &self.selected[(*pick as usize * kv + kv_head) * w..][..w])
    }

    /// This case with one sink logit per query head: from well below to well
    /// above the heads' scores, so some sinks dominate their softmax.
    fn with_sinks(mut self, seed: u64) -> Self {
        let mut noise = Noise(seed);
        self.sinks = (0..self.form.heads())
            .map(|_| noise.next() * 6.0 + 1.0)
            .collect();
        self
    }

    /// The sink rows (`SK`).
    fn sink_rows(&self) -> usize {
        usize::from(!self.sinks.is_empty())
    }

    /// The portable `head_rotary` (or `head_norm` with `rotate` false) of one
    /// head row, rounded to bf16.
    fn prepare(&self, raw: &[f32], norm: Option<&[f32]>, rotate: bool, row: usize) -> Vec<f32> {
        self.prepare_rounded(raw, norm, rotate, row, rounding_model::NONE)
    }

    /// [`Self::prepare`] with the rotated F32 row rounded by `rounding`
    /// before it is published (the portable `rounded_head_rotary` of the
    /// key): a rounded row is held in BF16 after its norm and after its
    /// rotation, as the trained model holds it before its rounding.
    fn prepare_rounded(&self, raw: &[f32], norm: Option<&[f32]>, rotate: bool, row: usize, rounding: Rounding) -> Vec<f32> {
        self.prepare_held(raw, norm, rotate, row, rounding, rounding.block > 0)
    }

    /// [`Self::prepare_rounded`] with the holds `hold` says, whether or not
    /// the row rounds (unheld rounded keys: what the holds move).
    fn prepare_held(&self, raw: &[f32], norm: Option<&[f32]>, rotate: bool, row: usize, rounding: Rounding, hold: bool) -> Vec<f32> {
        let (p, w) = (if rotate { self.form.p } else { 0 }, self.form.w());
        let held = |x: Vec<f32>| if hold { x.into_iter().map(bf16).collect() } else { x };
        let normalized = match norm {
            Some(norm) => {
                let squares = raw.iter().fold(0.0f32, |sum, x| x.mul_add(*x, sum));
                let inverse = 1.0 / (squares / w as f32 + self.epsilon).sqrt();
                raw.iter()
                    .zip(norm)
                    .map(|(x, n)| x * inverse * n)
                    .collect::<Vec<_>>()
            }
            None => raw.to_vec(),
        };
        let rotated = held(self.turn(held(normalized), row, p, 1.0));
        rounding_model::round(rounding, &rotated).into_iter().map(bf16).collect()
    }

    /// `row`'s first `p` rotary pairs of `x` turned by their angles times
    /// `sign` (the portable `rotate_pairs`), in F32.
    fn turn(&self, mut x: Vec<f32>, row: usize, p: usize, sign: f32) -> Vec<f32> {
        let w = x.len();
        for pair in 0..p {
            let (lo, hi) = self.form.rotary.columns(pair, p, w);
            let coordinate = self.coordinates[row * 4 + self.components[pair] as usize];
            let angle = coordinate as f32 * self.frequencies[pair];
            let (c, s) = (
                angle.cos() * self.amplitudes[pair],
                sign * angle.sin() * self.amplitudes[pair],
            );
            let (low, high) = (x[lo], x[hi]);
            x[lo] = low * c - high * s;
            x[hi] = high * c + low * s;
        }
        x
    }

    /// The portable body's result and final histories, computed on the host.
    fn expected(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.expected_with(None)
    }

    /// [`Self::expected`] with the rows' prepared keys `keys` (`[M, KV * W]`)
    /// in place of the host model's: a native's own rounded keys, which an
    /// F32 input on the other side of a rounding boundary may move by one
    /// code.
    fn expected_with(&self, keys: Option<&[f32]>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let form = self.form;
        let (kv, g, w, i) = (form.kv, form.g, form.w(), form.i());
        fn norm(form: Form, weights: &[f32]) -> Option<&[f32]> {
            form.norm.then_some(weights)
        }
        let norm = |weights| norm(form, weights);
        let fresh_keys = (0..if form.fresh { self.rows } else { 0 })
            .map(|row| {
                (0..kv)
                    .map(|head| match keys {
                        Some(keys) => keys[(row * kv + head) * w..][..w].to_vec(),
                        None => self.prepare_rounded(
                            &self.key[(row * kv + head) * w..][..w],
                            norm(&self.key_norm),
                            true,
                            row,
                            self.rounding_scalars(),
                        ),
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // A stored-key value is the prepared key.
        let fresh_values = if form.stored_key {
            fresh_keys.clone()
        } else {
            (0..if form.fresh { self.rows } else { 0 })
                .map(|row| {
                    (0..kv)
                        .map(|head| {
                            self.prepare(
                                &self.value[(row * kv + head) * w..][..w],
                                form.value_norm.then_some(&self.value_norm[..]),
                                false,
                                row,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        // History values are read from the key plane for stored keys.
        let history_values = if form.stored_key {
            &self.history_key
        } else {
            &self.history_value
        };
        let mut result = vec![0.0; self.rows * kv * g * w];
        for row in 0..self.rows {
            for head in 0..kv * g {
                let kv_head = head / g;
                let query = self.prepare(
                    &self.query[(row * kv * g + head) * (w + i)..][..w],
                    form.query_norm.then_some(&self.query_norm[..]),
                    true,
                    row,
                );
                let mut entries: Vec<(&[f32], &[f32])> = Vec::new();
                for span in 0..self.spans {
                    let lo = self.visible[(row * self.spans + span) * 2];
                    let hi = self.visible[(row * self.spans + span) * 2 + 1];
                    for token in lo.max(0)..hi {
                        let at = (token as usize * kv + kv_head) * w;
                        entries.push((&self.history_key[at..][..w], &history_values[at..][..w]));
                    }
                }
                if form.fresh {
                    for token in self.fresh[row * 2].max(0)..self.fresh[row * 2 + 1] {
                        entries.push((
                            &fresh_keys[token as usize][kv_head],
                            &fresh_values[token as usize][kv_head],
                        ));
                    }
                }
                // The index-list source's rows, each read as key and value.
                entries.extend(self.picked(row, kv_head).map(|picked| (picked, picked)));
                let scores = entries
                    .iter()
                    .map(|(key, _)| {
                        query
                            .iter()
                            .zip(key.iter())
                            .map(|(q, k)| f64::from(*q) * f64::from(*k))
                            .sum::<f64>()
                            * f64::from(self.scale)
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - maximum).exp())
                    .collect::<Vec<_>>();
                // A sink's logit weighs in the denominator only.
                let sink = self
                    .sinks
                    .get(head)
                    .map_or(0.0, |sink| (f64::from(*sink) - maximum).exp());
                let denominator = (weights.iter().sum::<f64>() + sink).max(1e-30);
                for column in 0..w {
                    let attended = entries
                        .iter()
                        .zip(&weights)
                        .map(|((_, value), weight)| weight * f64::from(value[column]))
                        .sum::<f64>()
                        / denominator;
                    let gate = if form.interleaved {
                        Some(self.query[(row * kv * g + head) * (w + i) + w + column])
                    } else if form.separate > 0 {
                        Some(
                            self.gate
                                [(row * kv * g + head) * form.separate + column % form.separate],
                        )
                    } else {
                        None
                    };
                    let gated = match gate.map(f64::from) {
                        None => attended,
                        Some(gate) if form.softplus => {
                            attended * (gate.max(0.0) + (-gate.abs()).exp().ln_1p())
                        }
                        Some(gate) => attended / (1.0 + (-gate).exp()),
                    };
                    result[(row * kv * g + head) * w + column] = bf16(gated as f32);
                }
                // The output rotary (ungated forms only): the published row
                // turned back and published again.
                if form.rotary.inverse_output {
                    let out = &mut result[(row * kv * g + head) * w..][..w];
                    let turned = self.turn(out.to_vec(), row, form.p, -1.0);
                    for (target, value) in out.iter_mut().zip(turned) {
                        *target = bf16(value);
                    }
                }
            }
        }
        let mut history_key = self.history_key.clone();
        let mut history_value = self.history_value.clone();
        for row in 0..if form.fresh { self.rows } else { 0 } {
            let destination = self.destinations[row];
            if destination < 0 {
                continue;
            }
            for head in 0..kv {
                let at = (destination as usize * kv + head) * w;
                history_key[at..at + w].copy_from_slice(&fresh_keys[row][head]);
                if !form.stored_key {
                    history_value[at..at + w].copy_from_slice(&fresh_values[row][head]);
                }
            }
        }
        (result, history_key, history_value)
    }
}

// ---------------------------------------------------------------------------
// The reference interpreter.

/// Checks the host model against the portable body of `entry`.
fn check_host_model(entry: &str, case: &Case) {
    let form = case.form;
    let (result, history_key, history_value) = portable(entry, case);
    let expected = case.expected();
    for (index, (a, e)) in result.iter().zip(&expected.0).enumerate() {
        assert!(
            (a - e).abs() <= 1.0e-3 + 8.0e-3 * e.abs(),
            "{entry} {form:?}: portable result[{index}] {a}, host model {e}"
        );
    }
    for (actual, expected, name) in [
        (&history_key, &expected.1, "history_key"),
        (&history_value, &expected.2, "history_value"),
    ] {
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 8.0e-3 * e.abs().max(1.0),
                "{entry} {form:?}: portable {name}[{index}] {a}, host model {e}"
            );
        }
    }
}

/// The portable body of `entry` on `case`: its result and final histories.
fn portable(entry: &str, case: &Case) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    use seismic_lang::{
        checked::{SourceFile},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        reference_math::ReferenceScalar,
        registry,
        types::DType,
    };
    let mut sources = seismic_std::sources();
    sources.push(SourceFile {
        path: "attention.seismic".into(),
        text: include_str!("../kernels/attention.seismic").into(),
    });
    sources.push(SourceFile {
        path: "functions.seismic".into(),
        text: include_str!("../kernels/functions.seismic").into(),
    });
    sources.push(SourceFile {
        path: "rounding.seismic".into(),
        text: include_str!("../kernels/rounding.seismic").into(),
    });
    let module = crate::common::check(sources);
    let logical = module
        .entry(
            module.entry_named(entry).unwrap(),
            &ElementBindings::new().bind("A", registry::dense(DType::BF16)),
        )
        .unwrap();
    let form = case.form;
    let (m, w, t, kv, heads) = (
        case.rows,
        form.w(),
        case.history_rows,
        form.kv,
        form.heads(),
    );
    let f = usize::from(form.fresh);
    let nq = usize::from(form.query_norm);
    let n = usize::from(form.norm);
    let nv = usize::from(form.value_norm);
    let mut interpreter = Interpreter::new(&logical);
    let mut tensor = |dtype, shape: Vec<usize>, values: Vec<f64>| {
        Arg::Tensor(interpreter.add_tensor(TensorData::dense(dtype, shape, values)))
    };
    let floats = |values: &[f32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let ints = |values: &[i32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let rows = |values: &[f32], count: usize| {
        floats(
            &values
                .iter()
                .copied()
                .cycle()
                .take(count * values.len())
                .collect::<Vec<_>>(),
        )
    };
    let mut args = vec![
        tensor(
            DType::BF16,
            vec![m, heads, w + form.i()],
            floats(&case.query),
        ),
        tensor(
            DType::BF16,
            vec![m, heads, form.separate],
            floats(&case.gate),
        ),
        tensor(DType::BF16, vec![f, m, kv * w], floats(&case.key)),
        tensor(DType::BF16, vec![form.value_layers(), m, kv * w], floats(&case.value)),
        tensor(DType::F32, vec![nq, w], rows(&case.query_norm, nq)),
        tensor(DType::F32, vec![n, w], rows(&case.key_norm, n)),
        tensor(DType::F32, vec![nv, w], rows(&case.value_norm, nv)),
        tensor(DType::F32, vec![case.sink_rows(), heads], floats(&case.sinks)),
        tensor(DType::I32, vec![form.p], ints(&case.components)),
        tensor(DType::F32, vec![form.p], floats(&case.frequencies)),
        tensor(DType::F32, vec![form.p], floats(&case.amplitudes)),
        tensor(DType::I32, vec![m, 4], ints(&case.coordinates)),
        tensor(DType::I32, vec![m, case.spans, 2], ints(&case.visible)),
        tensor(DType::I32, vec![m, 2], ints(&case.fresh)),
        tensor(DType::I32, vec![m], ints(&case.destinations)),
        tensor(DType::I32, vec![m, case.picks], ints(&case.selected_rows)),
        tensor(DType::BF16, vec![case.selected_plane_rows(), kv, w], floats(&case.selected)),
        tensor(DType::BF16, vec![t, kv, w], floats(&case.history_key)),
        tensor(DType::BF16, vec![form.value_rows(t), kv, w], floats(&case.history_value)),
        tensor(DType::I32, case.extents().to_vec(), Vec::new()),
        Arg::Scalar(ReferenceScalar::F32(case.epsilon.to_bits())),
        Arg::Scalar(ReferenceScalar::F32(case.scale.to_bits())),
        Arg::Scalar(ReferenceScalar::I32(i32::from(form.softplus))),
        Arg::Scalar(ReferenceScalar::U32(t as u32)),
        Arg::Scalar(ReferenceScalar::U32(case.selected_slab_rows as u32)),
        Arg::Scalar(ReferenceScalar::I32(case.rounding_scalars().block)),
        Arg::Scalar(ReferenceScalar::I32(case.rounding_scalars().element)),
        Arg::Scalar(ReferenceScalar::I32(case.rounding_scalars().scale)),
        Arg::Scalar(ReferenceScalar::F32(case.rounding_scalars().amax_floor.to_bits())),
    ];
    // The prefill takes its WorkList bitmap (none: B = 0) and block keys
    // (BR = 1) after the destinations.
    let prefill = entry == "attention_prefill";
    if prefill {
        args.insert(15, tensor(DType::I32, vec![0, 1], Vec::new()));
        args.insert(15, tensor(DType::U32, vec![m, 0], Vec::new()));
    }
    // The history planes' ordinals, after the index-list source.
    let planes = if prefill { 19 } else { 17 };
    let outcome = interpreter.run(&args).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("{entry} portable body failed: {failure}");
    }
    let result = outcome.results().next().unwrap();
    let OutcomeValue::Tensor(tensor) = result.value() else {
        panic!("{entry} returns a tensor")
    };
    let result = (0..tensor.element_count()).map(|index| tensor.read(index).unwrap() as f32).collect();
    let inputs = outcome.inputs().collect::<Vec<_>>();
    let plane = |ordinal: usize| {
        let tensor = inputs.iter().find(|input| input.ordinal() == ordinal).unwrap().tensor();
        (0..tensor.element_count()).map(|index| tensor.read(index).unwrap() as f32).collect::<Vec<_>>()
    };
    (result, plane(planes), plane(planes + 1))
}

// ---------------------------------------------------------------------------
// Devices.

fn devices() -> Vec<Device> {
    let catalog = seismic::Devices::discover(&[]).unwrap();
    let names =
        std::env::var("ATTENTION_BACKENDS").unwrap_or_else(|_| "metal,cuda,vulkan,cpu".into());
    names
        .split(',')
        .filter_map(|name| {
            let backend = match name.trim() {
                "metal" => BackendName::Metal,
                "cuda" => BackendName::Cuda,
                "vulkan" => BackendName::Vulkan,
                "cpu" => BackendName::Cpu,
                other => panic!("ATTENTION_BACKENDS: unknown backend {other}"),
            };
            catalog.open_backend(backend).ok()
        })
        .collect()
}

fn tensor(device: &Device, element: Element, shape: &[usize], bytes: Vec<u8>) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    Tensor::from_host(device, element, &shape, &bytes).unwrap()
}

fn bf16_tensor(device: &Device, shape: &[usize], values: &[f32]) -> Tensor {
    tensor(
        device,
        Element::bf16(),
        shape,
        values
            .iter()
            .flat_map(|v| bf16_bits(*v).to_le_bytes())
            .collect(),
    )
}

fn f32_tensor(device: &Device, shape: &[usize], values: &[f32]) -> Tensor {
    tensor(
        device,
        Element::f32(),
        shape,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

fn i32_tensor(device: &Device, shape: &[usize], values: &[i32]) -> Tensor {
    tensor(
        device,
        Element::i32(),
        shape,
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

fn bf16_values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(2)
        .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16))
        .collect()
}

/// Device tensors of one case.
struct Bound {
    query: Tensor,
    gate: Tensor,
    key: Tensor,
    value: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
    value_norm: Tensor,
    sinks: Tensor,
    components: Tensor,
    frequencies: Tensor,
    amplitudes: Tensor,
    coordinates: Tensor,
    visible: Tensor,
    fresh: Tensor,
    destinations: Tensor,
    /// The prefill's absent WorkList bitmap ([M, 0]).
    selection: Tensor,
    /// Its blocks' keys (`[0, 1]`: every key read).
    block_slots: Tensor,
    /// The index-list source: its picks and its slab-stored plane.
    selected_rows: Tensor,
    selected: Tensor,
    _selected_slabs: Option<SlabTensor>,
    form: Tensor,
    _slabs: SlabTensor,
    history_key: Tensor,
    history_value: Tensor,
}

impl Bound {
    fn new(device: &Device, case: &Case) -> Self {
        let form = case.form;
        let (m, w, t, kv, heads) = (
            case.rows,
            form.w(),
            case.history_rows,
            form.kv,
            form.heads(),
        );
        let (f, nq, n, nv) = (
            usize::from(form.fresh),
            usize::from(form.query_norm),
            usize::from(form.norm),
            usize::from(form.value_norm),
        );
        let slab_rows = case.slab_rows as u64;
        let region = || SlabRegion {
            element: Element::bf16(),
            row_shape: vec![kv as u64, w as u64],
        };
        // A stored-key history holds the key plane alone; its entries bind
        // zero rows of a value plane.
        let stored = [&case.history_key, &case.history_value];
        let stored = &stored[..if form.stored_key { 1 } else { 2 }];
        let mut slabs =
            SlabTensor::new(device, slab_rows, t as u64, vec![region(); stored.len()]).unwrap();
        let row_bytes = kv * w * 2;
        let planes = stored.iter().map(|values| {
            values
                .iter()
                .flat_map(|v| bf16_bits(*v).to_le_bytes())
                .collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        for index in 0..(t as u64).div_ceil(slab_rows) {
            slabs.add_slab().unwrap();
            let start = index * slab_rows;
            let count = slab_rows.min(t as u64 - start);
            for (plane, bytes) in planes.iter().enumerate() {
                slabs
                    .region_rows(plane, start, count)
                    .unwrap()
                    .write_from_host(
                        &bytes[start as usize * row_bytes..(start + count) as usize * row_bytes],
                    )
                    .unwrap();
            }
        }
        let norm = |values: &[f32], rows: usize| {
            f32_tensor(
                device,
                &[rows, w],
                &values
                    .iter()
                    .copied()
                    .cycle()
                    .take(rows * w)
                    .collect::<Vec<_>>(),
            )
        };
        // The index-list plane in slabs of its own, which its picks cross.
        let selected_rows_total = case.selected_plane_rows() as u64;
        let (selected_slabs, selected) = if selected_rows_total == 0 {
            (None, bf16_tensor(device, &[0, kv, w], &[]))
        } else {
            let per = case.selected_slab_rows as u64;
            let mut slabs = SlabTensor::new(device, per, selected_rows_total, vec![region()]).unwrap();
            let bytes = case.selected.iter().flat_map(|v| bf16_bits(*v).to_le_bytes()).collect::<Vec<_>>();
            for index in 0..selected_rows_total.div_ceil(per) {
                slabs.add_slab().unwrap();
                let start = index * per;
                let count = per.min(selected_rows_total - start);
                slabs
                    .region_rows(0, start, count)
                    .unwrap()
                    .write_from_host(&bytes[start as usize * row_bytes..(start + count) as usize * row_bytes])
                    .unwrap();
            }
            let logical = slabs.logical_region(0).unwrap();
            (Some(slabs), logical)
        };
        Self {
            query: bf16_tensor(device, &[m, heads, w + form.i()], &case.query),
            gate: bf16_tensor(device, &[m, heads, form.separate], &case.gate),
            key: bf16_tensor(device, &[f, m, kv * w], &case.key),
            value: bf16_tensor(device, &[form.value_layers(), m, kv * w], &case.value),
            query_norm: norm(&case.query_norm, nq),
            key_norm: norm(&case.key_norm, n),
            value_norm: norm(&case.value_norm, nv),
            sinks: f32_tensor(device, &[case.sink_rows(), heads], &case.sinks),
            components: i32_tensor(device, &[form.p], &case.components),
            frequencies: f32_tensor(device, &[form.p], &case.frequencies),
            amplitudes: f32_tensor(device, &[form.p], &case.amplitudes),
            coordinates: i32_tensor(device, &[m, 4], &case.coordinates),
            visible: i32_tensor(device, &[m, case.spans, 2], &case.visible),
            fresh: i32_tensor(device, &[m, 2], &case.fresh),
            destinations: i32_tensor(device, &[m], &case.destinations),
            selection: Tensor::from_host(device, Element::u32(), &[m as u64, 0], &[]).unwrap(),
            block_slots: Tensor::from_host(device, Element::i32(), &[0, 1], &[]).unwrap(),
            selected_rows: i32_tensor(device, &[m, case.picks], &case.selected_rows),
            selected,
            _selected_slabs: selected_slabs,
            form: Tensor::from_host(
                device,
                Element::i32(),
                &case.extents().map(|extent| extent as u64),
                &[],
            )
            .unwrap(),
            history_key: slabs.logical_region(0).unwrap(),
            history_value: if form.stored_key {
                bf16_tensor(device, &[0, kv, w], &[])
            } else {
                slabs.logical_region(1).unwrap()
            },
            _slabs: slabs,
        }
    }
}

/// The statics of `form` with `sinks` sink rows on `device` (CPU fixes only
/// its declared ones).
fn statics(device: &Device, form: Form, sinks: usize) -> NativeSpecialization {
    let common = [
        ("P", form.p),
        ("S", form.s),
        ("I", form.i()),
        ("U", form.separate),
        ("F", usize::from(form.fresh)),
        ("NQ", usize::from(form.query_norm)),
        ("N", usize::from(form.norm)),
        ("NV", usize::from(form.value_norm)),
        ("SK", sinks),
        ("RA", usize::from(form.rotary.adjacent)),
        ("RT", usize::from(form.rotary.trailing)),
        ("RO", usize::from(form.rotary.inverse_output)),
        ("VK", usize::from(form.stored_key)),
    ];
    let gpu = [("KV", form.kv), ("G", form.g)];
    let base = common
        .into_iter()
        .fold(NativeSpecialization::new(), |s, (name, value)| {
            s.with_static(name, value as u64)
        });
    if device.backend() == BackendName::Cpu {
        base
    } else {
        gpu.into_iter()
            .fold(base, |s, (name, value)| s.with_static(name, value as u64))
    }
}

/// Vulkan grouped-query matrix decode configurations (PARTS, SIMDS) whose
/// SIMDS * 16 matrix rows hold the query group, within the declaration's
/// shared bytes.
fn vulkan_matrix_configurations(
    form: Form,
    configs: &[(u64, u64)],
) -> Vec<Vec<(&'static str, u64)>> {
    let w = form.w() as u64;
    configs
        .iter()
        .filter(|(_, simds)| {
            form.g as u64 <= simds * 16
                && (w <= 64 || w % 64 == 0)
                && (w <= 256 || w % 256 == 0)
                && 64 * (w.min(256) + 8) + simds * 2048 + simds * 16 * (w + 8) * 2 <= 32768
        })
        .map(|&(parts, simds)| {
            vec![
                ("SPAN", 32),
                ("PARTS", parts),
                ("SIMDS", simds),
                ("SLICES", 1),
                ("MATRIX", 1),
            ]
        })
        .collect()
}

/// CUDA grouped-query matrix decode configurations (PARTS, WARPS) whose
/// WARPS * 16 matrix rows hold the query group, within the declaration's
/// shared bytes.
fn cuda_matrix_configurations(
    form: Form,
    configs: &[(u64, u64, u64, u64)],
) -> Vec<Vec<(&'static str, u64)>> {
    configs
        .iter()
        .filter(|(_, warps, stages, columns)| {
            // The declaration's shared bound: the query tile, STAGES K-piece +
            // V-window stages (16-key tiles above W = 256), the exchange.
            let w = form.w() as u64;
            let keys = if w > 256 { 16 } else { 32 };
            form.g as u64 <= warps * 16
                && (w <= 256 || w % 256 == 0)
                && w.min(256 * columns) % (16 * columns) == 0
                && (warps * 16 * w + stages * keys * (w.min(256) + w.min(256 * columns))) * 2
                    + warps * columns * w * 4
                    <= 98304
        })
        .map(|&(parts, warps, stages, columns)| {
            vec![
                ("PARTS", parts),
                ("WARPS", warps),
                ("SLICES", 1),
                ("MATRIX", 1),
                ("STAGES", stages),
                ("COLUMNS", columns),
                ("QREG", 0),
            ]
        })
        .collect()
}

/// Metal's grouped-query matrix decode configurations admissible at `form`,
/// from (span, parts, simds, keys): a simdgroup's outputs cover at most 128
/// columns of every 8-row block, teams of whole column slices, within the
/// declaration's threadgroup bytes.
fn metal_matrix_configurations(
    form: Form,
    configs: &[(u64, u64, u64, u64)],
) -> Vec<Vec<(&'static str, u64)>> {
    let w = (2 * form.p + form.s) as u64;
    let mut admitted: Vec<Vec<(&'static str, u64)>> = Vec::new();
    // One row and four rows per threadgroup (verification rows).
    for tokens in [1u64, 4] {
        let rows = (tokens * form.g as u64).div_ceil(8) * 8;
        let cols = (rows / 8 * w / 128).max(1);
        let wc = w / cols;
        for &(span, parts, simds, keys) in configs {
            let simds = simds.max(cols);
            let exchange = if cols > 1 {
                2 * simds * rows * keys * 4
            } else {
                0
            };
            let bytes = rows * w * 2
                + (simds * keys * (wc + 8) * 2 + exchange).max(rows * w * 4)
                + simds / cols * rows * 8;
            let config = vec![
                ("SPAN", span),
                ("PARTS", parts),
                ("SIMDS", simds),
                ("SLICES", 1),
                ("MATRIX", 1),
                ("KEYS", keys),
                ("TOKENS", tokens),
            ];
            if w % cols == 0
                && wc % 32 == 0
                && simds % cols == 0
                && bytes <= 32768
                && !admitted.contains(&config)
            {
                admitted.push(config);
            }
        }
    }
    admitted
}

/// Tuning configurations of the decode or prefill entry on `device`.
fn configurations(device: &Device, decode: bool, form: Form) -> Vec<Vec<(&'static str, u64)>> {
    match (device.backend(), decode) {
        (BackendName::Cpu, true) => vec![vec![("PARTS", 8)], vec![("PARTS", 1)]],
        (BackendName::Cpu, false) => vec![vec![]],
        (BackendName::Metal, true) => {
            let slices = [8u64, 4, 2, 1]
                .into_iter()
                .find(|s| form.g as u64 % s == 0)
                .unwrap();
            let mut configurations = vec![
                vec![
                    ("SPAN", 32),
                    ("PARTS", 16),
                    ("SIMDS", 4),
                    ("SLICES", 1),
                    ("MATRIX", 0),
                    ("KEYS", 16),
                    ("TOKENS", 1),
                ],
                vec![
                    ("SPAN", 64),
                    ("PARTS", 8),
                    ("SIMDS", 8),
                    ("SLICES", slices),
                    ("MATRIX", 0),
                    ("KEYS", 16),
                    ("TOKENS", 1),
                ],
            ];
            configurations.extend(metal_matrix_configurations(
                form,
                &[(32, 16, 4, 16), (64, 8, 2, 8)],
            ));
            configurations
        }
        (BackendName::Metal, false) => {
            // The smallest declared HEADS holding all of a kv head's query heads.
            let heads = (form.g.next_power_of_two() as u64).min(16);
            vec![
                vec![
                    ("QT", 16),
                    ("HEADS", heads),
                    ("SPLIT_GROUPS", 256),
                    ("DIRECT", 0),
                ],
                vec![
                    ("QT", 8),
                    ("HEADS", heads),
                    ("SPLIT_GROUPS", 1),
                    ("DIRECT", 0),
                ],
                vec![
                    ("QT", 16),
                    ("HEADS", heads),
                    ("SPLIT_GROUPS", 256),
                    ("DIRECT", 1),
                ],
                vec![
                    ("QT", 16),
                    ("HEADS", heads),
                    ("SPLIT_GROUPS", 1),
                    ("DIRECT", 1),
                ],
            ]
        }
        (BackendName::Cuda, true) => {
            let slices = [8u64, 4, 2, 1]
                .into_iter()
                .find(|s| form.g as u64 % s == 0)
                .unwrap();
            let mut configurations = vec![
                vec![
                    ("PARTS", 12),
                    ("WARPS", 4),
                    ("SLICES", 1),
                    ("MATRIX", 0),
                    ("STAGES", 2),
                    ("COLUMNS", 1),
                    ("QREG", 0),
                ],
                vec![
                    ("PARTS", 48),
                    ("WARPS", 8),
                    ("SLICES", slices),
                    ("MATRIX", 0),
                    ("STAGES", 2),
                    ("COLUMNS", 1),
                    ("QREG", 0),
                ],
            ];
            configurations.extend(cuda_matrix_configurations(
                form,
                &[(12, 1, 2, 4), (48, 2, 3, 2), (24, 4, 4, 1)],
            ));
            configurations
        }
        (BackendName::Cuda, false) => {
            // The split configuration's block: two warps of two column
            // groups, or for a query group wider than their 32 rows
            // (G <= WARPS * 16) four warps of one.
            let (warps, columns) = if form.g <= 32 { (2, 2) } else { (4, 1) };
            let mut configurations = vec![
                vec![
                    ("WARPS", 4),
                    ("SPLIT_GROUPS", 1),
                    ("STAGES", 2),
                    ("COLUMNS", 1),
                    ("QREG", 0),
                ],
                vec![
                    ("WARPS", warps),
                    ("SPLIT_GROUPS", 256),
                    ("STAGES", 2),
                    ("COLUMNS", columns),
                    ("QREG", 0),
                ],
            ];
            // Register queries hold heads up to 256 columns.
            if form.w() <= 256 {
                configurations.push(vec![
                    ("WARPS", 8),
                    ("SPLIT_GROUPS", 256),
                    ("STAGES", 2),
                    ("COLUMNS", 1),
                    ("QREG", 1),
                ]);
            }
            configurations
        }
        (BackendName::Vulkan, true) => {
            let slices = [8u64, 4, 2, 1]
                .into_iter()
                .find(|s| form.g as u64 % s == 0)
                .unwrap();
            let mut configurations = vec![
                vec![
                    ("SPAN", 32),
                    ("PARTS", 16),
                    ("SIMDS", 4),
                    ("SLICES", 1),
                    ("MATRIX", 0),
                ],
                vec![
                    ("SPAN", 64),
                    ("PARTS", 8),
                    ("SIMDS", 8),
                    ("SLICES", slices),
                    ("MATRIX", 0),
                ],
            ];
            configurations.extend(vulkan_matrix_configurations(
                form,
                &[(16, 1), (64, 2), (32, 4)],
            ));
            configurations
        }
        (BackendName::Vulkan, false) => vec![
            vec![("ROWS", 64), ("SPLIT_GROUPS", 256)],
            vec![("ROWS", 64), ("SPLIT_GROUPS", 1)],
        ],
        (other, _) => panic!("no configurations for {other:?}"),
    }
}

/// [`configurations`] that can hold `case`: an index-list source runs in
/// Metal's fragment prefill form alone (its declaration refuses DIRECT).
fn case_configurations(device: &Device, decode: bool, case: &Case) -> Vec<Vec<(&'static str, u64)>> {
    configurations(device, decode, case.form)
        .into_iter()
        .filter(|config| case.picks == 0 || !config.contains(&("DIRECT", 1)))
        .collect()
}

/// Runs the decode or prefill entry of `case` with `specialization`; the
/// result and final histories, or the preparation error.
fn run(
    device: &Device,
    decode: bool,
    specialization: &NativeSpecialization,
    case: &Case,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let mut bound = Bound::new(device, case);
    // The prefill in its Visible form: no WorkList bitmap (B = 0, a static
    // of the GPU natives).
    let specialization = &if decode || device.backend() == BackendName::Cpu {
        specialization.clone()
    } else {
        specialization.clone().with_static("B", 0)
    }
    // The case's key rounding and index-list source.
    .with_static("KR", u64::from(case.key_rounding.is_some()))
    .with_static("KS", case.picks as u64);
    let rounding = case.rounding_scalars();
    macro_rules! call {
        ($module:ident $(, $extra:ident: $value:expr)*) => {{
            let kernel = $module::native_for_device_with(
                device,
                $module::Elements { A: Element::bf16() },
                specialization,
            )
            .map_err(|error| error.to_string())?;
            kernel
                .call($module::Args {
                    $($extra: $value,)*
                    query: &bound.query,
                    gate: &bound.gate,
                    key: &bound.key,
                    value: &bound.value,
                    query_norm: &bound.query_norm,
                    key_norm: &bound.key_norm,
                    value_norm: &bound.value_norm,
                    sinks: &bound.sinks,
                    rotary_components: &bound.components,
                    rotary_frequencies: &bound.frequencies,
                    rotary_amplitudes: &bound.amplitudes,
                    coordinates: &bound.coordinates,
                    visible: &bound.visible,
                    fresh: &bound.fresh,
                    destinations: &bound.destinations,
                    selected_rows: &bound.selected_rows,
                    selected: &bound.selected,
                    history_key: &mut bound.history_key,
                    history_value: &mut bound.history_value,
                    form: &bound.form,
                    epsilon: case.epsilon,
                    scale: case.scale,
                    gate_function: i32::from(case.form.softplus),
                    slab_rows: case.slab_rows as u32,
                    selected_slab_rows: case.selected_slab_rows as u32,
                    key_block: rounding.block,
                    key_element: rounding.element,
                    key_scale: rounding.scale,
                    key_amax_floor: rounding.amax_floor,
                })
                .map_err(|error| format!("{error:?}"))?
                .value
        }};
    }
    let result = if decode {
        call!(attention_decode)
    } else {
        call!(attention_prefill, selection: &bound.selection, block_slots: &bound.block_slots)
    };
    Ok((
        bf16_values(&result),
        bf16_values(&bound.history_key),
        bf16_values(&bound.history_value),
    ))
}

/// An attended row's bar: bf16 publication plus reduced-precision operands,
/// `b(x) = 4e-3 + 1.6e-2 |x|` per component.
fn attended_bar(x: f32) -> f32 {
    4.0e-3 + 1.6e-2 * x.abs()
}

/// Each result's bound. An unturned result is an attended row, held to
/// [`attended_bar`]. A result under the output rotary (RO = 1) is
/// `bf16(T(P))`: the published attended row `P` turned back by `T`, then
/// published again. The bound follows its roundings:
/// - Before the turn, `P` takes the roundings of an attended row and no
///   others: the stored rows it reads, rounded once on the device and in the
///   host model (`fresh_key`, attention.seismic:692; `prepare`,
///   attention_family.rs:309) and twice in V4.1's reference (the norm's
///   output, then the rotated row, in `v41_reference`), which reach `P`
///   through its scores and values; and `P`'s own publication
///   (attention.seismic:640; `bf16(gated)`, attention_family.rs:451; the
///   reference's `bf16` of `o`). An attended row's bar holds them, at most one bf16 step
///   (2^-7 |x|) of it the publication's, so `|ΔP_c| <= b(P_c)` for each of the
///   pair's components, at their own scale, not the turned result's.
/// - The turn is a rotation scaled by the pair's amplitude `a`: it carries the
///   pair's error vector unchanged in norm but for `a`, however far it cancels
///   the pair's components. So `|ΔT| <= a |(b(P_lo), b(P_hi))| <= a (√2 · 4e-3)
///   + 1.6e-2 |T(P)|` (Minkowski, with `a |P| = |T(P)|`).
/// - The turn's own F32 arithmetic (two products and a sum per component,
///   and its cosine and sine) adds under 2^-20 |T(P)|.
/// - The second publication rounds each side once, half a step (2^-8 |x|)
///   each: `|bf16(x) - bf16(y)| <= (1 + 2^-8) |x - y| + 2^-7 (1 + 2^-8) |x|`.
///   It also makes `|T(P)| <= (1 + 2^-8)` times the published pair's norm `r`.
///
/// With `u = 1 + 2^-8`, a turned component `e` is held to
/// `u (√2 · 4e-3 a + (1.6e-2 + 2^-20) u r) + 2^-7 u |e|`.
///
/// Each bound is paired with whether its result is turned.
fn result_bounds(case: &Case, expected: &[f32]) -> Vec<(f32, bool)> {
    let form = case.form;
    let (w, p) = (form.w(), form.p);
    let mut bounds: Vec<(f32, bool)> = expected.iter().map(|e| (attended_bar(*e), false)).collect();
    if form.rotary.inverse_output {
        let u = 1.0 + 2f32.powi(-8);
        for (row, published) in bounds.chunks_mut(w).zip(expected.chunks(w)) {
            for pair in 0..p {
                let (lo, hi) = form.rotary.columns(pair, p, w);
                let r = published[lo].hypot(published[hi]);
                let turned = 2f32.sqrt() * 4.0e-3 * case.amplitudes[pair].abs()
                    + (1.6e-2 + 2f32.powi(-20)) * u * r;
                for column in [lo, hi] {
                    row[column] = (u * turned + 2f32.powi(-7) * u * published[column].abs(), true);
                }
            }
        }
    }
    bounds
}

/// Results within [`result_bounds`]; histories exactly but keys, which may
/// differ by one bf16 step.
fn check(
    label: &str,
    actual: &(Vec<f32>, Vec<f32>, Vec<f32>),
    expected: &(Vec<f32>, Vec<f32>, Vec<f32>),
    case: &Case,
) {
    let w = case.form.w();
    let bounds = result_bounds(case, &expected.0);
    let mut worst = 0.0f32;
    // The turned results' largest error and largest share of their bound.
    let (mut turned_worst, mut turned_used) = (0.0f32, 0.0f32);
    assert_eq!(actual.0.len(), expected.0.len(), "{label}: result shape");
    let outside: Vec<(usize, f32, f32)> = actual
        .0
        .iter()
        .zip(&expected.0)
        .enumerate()
        .inspect(|(index, (a, e))| {
            let (error, (bound, turned)) = ((*a - *e).abs(), bounds[*index]);
            worst = worst.max(error);
            if turned {
                turned_worst = turned_worst.max(error);
                turned_used = turned_used.max(error / bound);
            }
        })
        .filter(|(index, (a, e))| (*a - *e).abs() > bounds[*index].0)
        .map(|(index, (a, e))| (index, *a, *e))
        .collect();
    if !outside.is_empty() {
        let listed: Vec<String> = outside
            .iter()
            .take(12)
            .map(|&(index, a, e)| {
                format!("(row·head {}, column {}) {a} vs {e}", index / w, index % w)
            })
            .collect();
        panic!(
            "{label}: {} results outside the bound (max |error| {worst:.2e}): {}",
            outside.len(),
            listed.join(", ")
        );
    }
    for (index, (a, e)) in actual.1.iter().zip(&expected.1).enumerate() {
        assert!(
            (a - e).abs() <= 8.0e-3 * e.abs().max(1.0),
            "{label}: history_key[{index}] device {a} expected {e}"
        );
    }
    for (index, (a, e)) in actual.2.iter().zip(&expected.2).enumerate() {
        assert!(
            (a - e).abs() <= 8.0e-3 * e.abs().max(1.0),
            "{label}: history_value[{index}] device {a} expected {e}"
        );
    }
    eprintln!(
        "{label}: max |result error| {worst:.2e}; turned: max |error| {turned_worst:.2e}, \
         at most {turned_used:.2} of its bound"
    );
}

/// Every device's natives of the decode or prefill entry against the host
/// model. A form a declaration's `where` rejects is reported, not run: it is
/// the typed plan-time refusal, allowed only on the backends `rejections`
/// lists (for some or all of their configurations).
fn check_natives(name: &str, decode: bool, case: &Case, rejections: &[BackendName]) {
    check_natives_against(name, decode, case, rejections, false, &case.expected());
}

/// [`check_natives`] against `expected`; with `required`, every
/// configuration of a backend `rejections` lists must be refused.
fn check_natives_against(
    name: &str,
    decode: bool,
    case: &Case,
    rejections: &[BackendName],
    required: bool,
    expected: &(Vec<f32>, Vec<f32>, Vec<f32>),
) {
    let mut failures = Vec::new();
    for device in devices() {
        for config in case_configurations(&device, decode, case) {
            let specialization = config
                .iter()
                .fold(statics(&device, case.form, case.sink_rows()), |s, (n, v)| {
                    s.with_param(*n, *v)
                });
            let label = format!("{name} {:?} {config:?}", device.backend());
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match run(&device, decode, &specialization, case) {
                    Ok(_) if required && rejections.contains(&device.backend()) => {
                        panic!("{label}: admitted a form its declaration must refuse")
                    }
                    Ok(actual) => check(&label, &actual, expected, case),
                    Err(error) if error.contains("violates the native `where` condition") => {
                        assert!(rejections.contains(&device.backend()), "{label}: {error}");
                        eprintln!("{label}: rejected by the declaration's `where`");
                    }
                    Err(error) => panic!("{label}: {error}"),
                }
            }));
            if outcome.is_err() {
                failures.push(label);
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}

// ---------------------------------------------------------------------------
// Rows.

/// Decode rows of two sequences: multi-span history, speculative rows seeing
/// earlier fresh rows, a row without a destination and a padding row.
fn decode_rows(base: i32) -> Vec<Row> {
    vec![
        Row {
            spans: vec![(0, base), (base + 7, base + 19)],
            fresh: (0, 1),
            destination: base + 40,
            position: base + 12,
        },
        Row {
            spans: vec![(0, base), (base + 7, base + 19)],
            fresh: (0, 2),
            destination: base + 41,
            position: base + 13,
        },
        Row {
            spans: vec![(base + 20, base + 33)],
            fresh: (2, 3),
            destination: -1,
            position: 13,
        },
        Row {
            spans: vec![],
            fresh: (0, 0),
            destination: -1,
            position: 0,
        },
    ]
}

/// Causal prefill rows of one sequence after `history` rows, under a sliding
/// window of `window` keys: row r sees the history rows and fresh rows
/// within its window, so windows smaller than the batch clip both.
fn window_rows(rows: usize, history: i32, window: i32) -> Vec<Row> {
    (0..rows as i32)
        .map(|r| {
            let position = history + r;
            let first = position - window + 1;
            Row {
                spans: if first < history {
                    vec![(first.max(0), history)]
                } else {
                    vec![]
                },
                fresh: ((first - history).max(0), r + 1),
                destination: history + 64 + r,
                position,
            }
        })
        .collect()
}

/// A non-causal block after `history` rows: every row sees the whole
/// history and the whole block, and nothing is appended (a DFlash/DSpark
/// draft block).
fn block_rows(rows: usize, history: i32) -> Vec<Row> {
    (0..rows as i32)
        .map(|r| Row {
            spans: vec![(0, history)],
            fresh: (0, rows as i32),
            destination: -1,
            position: history + r,
        })
        .collect()
}

/// A layer reading a source layer's history: the batch rows the source
/// appended (after `history` rows, at history + 64 + r) are visible history
/// spans, causally; there are no fresh rows.
fn shared_rows(rows: usize, history: i32) -> Vec<Row> {
    (0..rows as i32)
        .map(|r| Row {
            spans: vec![(0, history), (history + 64, history + 64 + r + 1)],
            fresh: (0, 0),
            destination: -1,
            position: history + r,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Forms.

const QWEN_SMALL: Form = Form {
    kv: 2,
    g: 2,
    p: 4,
    s: 24,
    interleaved: true,
    separate: 0,
    softplus: false,
    fresh: true,
    query_norm: true,
    norm: true,
    value_norm: false,
    rotary: Rotary::HALF_SPLIT_LEADING,
    stored_key: false,
};

/// Nemotron / MiniCPM-style: ungated, no q/k norm, no rotation.
const PLAIN_NOPE: Form = Form {
    kv: 2,
    g: 4,
    p: 0,
    s: 64,
    interleaved: false,
    separate: 0,
    softplus: false,
    fresh: true,
    query_norm: false,
    norm: false,
    value_norm: false,
    rotary: Rotary::HALF_SPLIT_LEADING,
    stored_key: false,
};

/// Laguna-style: a softplus gate per head, partial YaRN rotation with amplitude.
const HEAD_SOFTPLUS: Form = Form {
    kv: 2,
    g: 3,
    p: 16,
    s: 32,
    interleaved: false,
    separate: 1,
    softplus: true,
    fresh: true,
    query_norm: true,
    norm: true,
    value_norm: false,
    rotary: Rotary::HALF_SPLIT_LEADING,
    stored_key: false,
};

/// Muse-style: a separate sigmoid gate per column, G = 16.
const COLUMN_SIGMOID: Form = Form {
    kv: 1,
    g: 16,
    p: 16,
    s: 0,
    interleaved: false,
    separate: 32,
    softplus: false,
    fresh: true,
    query_norm: true,
    norm: true,
    value_norm: false,
    rotary: Rotary::HALF_SPLIT_LEADING,
    stored_key: false,
};

/// Gemma-style: ungated, value norm, scale 1.
const VALUE_NORM: Form = Form {
    kv: 2,
    g: 2,
    p: 8,
    s: 48,
    interleaved: false,
    separate: 0,
    softplus: false,
    fresh: true,
    query_norm: true,
    norm: true,
    value_norm: true,
    rotary: Rotary::HALF_SPLIT_LEADING,
    stored_key: false,
};

/// A KV-shared layer: no fresh rows.
const SHARED: Form = Form {
    kv: 2,
    g: 2,
    p: 8,
    s: 16,
    interleaved: false,
    separate: 0,
    softplus: false,
    fresh: false,
    query_norm: true,
    norm: true,
    value_norm: false,
    rotary: Rotary::HALF_SPLIT_LEADING,
    stored_key: false,
};

/// A draft block layer: q/k norms, full rotation, ungated.
const DRAFT: Form = Form {
    kv: 2,
    g: 2,
    p: 16,
    s: 0,
    interleaved: false,
    separate: 0,
    softplus: false,
    fresh: true,
    query_norm: true,
    norm: true,
    value_norm: false,
    rotary: Rotary::HALF_SPLIT_LEADING,
    stored_key: false,
};

/// (name, decode rows, case, backends whose declarations reject the form).
fn portable_cases() -> Vec<(&'static str, bool, Case, &'static [BackendName])> {
    vec![
        (
            "qwen decode",
            true,
            Case::new(
                QWEN_SMALL,
                64,
                decode_rows(5),
                16,
                1.0,
                1.0 / (32f32).sqrt(),
                11,
            ),
            &[],
        ),
        (
            "plain nope decode",
            true,
            Case::new(PLAIN_NOPE, 48, decode_rows(5), 16, 1.0, 0.125, 12),
            &[],
        ),
        (
            "head softplus decode",
            true,
            Case::new(HEAD_SOFTPLUS, 48, decode_rows(5), 16, 1.3466, 0.125, 13),
            &[],
        ),
        (
            "column sigmoid decode",
            true,
            Case::new(COLUMN_SIGMOID, 48, decode_rows(5), 16, 1.0, 0.1767767, 14),
            &[],
        ),
        (
            "value norm decode",
            true,
            Case::new(VALUE_NORM, 48, decode_rows(5), 16, 1.0, 1.0, 15),
            &[],
        ),
        (
            "shared decode",
            true,
            Case::new(SHARED, 96, shared_rows(3, 9), 32, 1.0, 0.2, 16),
            &[],
        ),
        (
            "draft block decode",
            true,
            Case::new(DRAFT, 48, block_rows(4, 21), 16, 1.0, 0.1767767, 17),
            &[],
        ),
        (
            "window prefill",
            false,
            Case::new(
                HEAD_SOFTPLUS,
                160,
                window_rows(20, 30, 8),
                16,
                1.3466,
                0.125,
                18,
            ),
            &[],
        ),
        (
            "draft block prefill",
            false,
            Case::new(DRAFT, 64, block_rows(16, 23), 16, 1.0, 0.1767767, 19),
            &[],
        ),
        // Sinks: a row without keys (the padding decode row) still attends
        // to zero.
        (
            "qwen sinks decode",
            true,
            Case::new(
                QWEN_SMALL,
                64,
                decode_rows(5),
                16,
                1.0,
                1.0 / (32f32).sqrt(),
                20,
            )
            .with_sinks(40),
            &[],
        ),
        (
            "draft sinks decode",
            true,
            Case::new(DRAFT, 48, block_rows(4, 21), 16, 1.0, 0.1767767, 21).with_sinks(41),
            &[],
        ),
        (
            "draft sinks prefill",
            false,
            Case::new(DRAFT, 64, block_rows(16, 23), 16, 1.0, 0.1767767, 22).with_sinks(42),
            &[],
        ),
        (
            "window sinks prefill",
            false,
            Case::new(
                HEAD_SOFTPLUS,
                160,
                window_rows(20, 30, 8),
                16,
                1.3466,
                0.125,
                23,
            )
            .with_sinks(43),
            &[],
        ),
    ]
}

#[test]
fn host_model_matches_portable_body() {
    for (name, decode, case, _) in portable_cases() {
        let entry = if decode {
            "attention_decode"
        } else {
            "attention_prefill"
        };
        check_host_model(entry, &case);
        eprintln!("{name}: host model agrees with the portable body");
    }
}

#[test]
fn decode_natives_match_portable_body() {
    for (name, decode, case, rejections) in portable_cases() {
        if decode {
            check_natives(name, true, &case, rejections);
        }
    }
}

/// Prefill cases checked against the host model only (the interpreter is too
/// slow at these shapes): every form over windowed and shared rows.
fn prefill_cases() -> Vec<(&'static str, Case, &'static [BackendName])> {
    vec![
        (
            "qwen prefill",
            Case::new(
                QWEN_SMALL,
                400,
                window_rows(40, 300, 1000),
                64,
                1.0,
                1.0 / (32f32).sqrt(),
                30,
            ),
            &[],
        ),
        (
            "plain nope prefill",
            Case::new(
                PLAIN_NOPE,
                160,
                window_rows(24, 60, 1000),
                32,
                1.0,
                0.125,
                31,
            ),
            &[],
        ),
        (
            "column sigmoid window prefill",
            Case::new(
                COLUMN_SIGMOID,
                160,
                window_rows(20, 30, 8),
                16,
                1.0,
                0.1767767,
                32,
            ),
            &[],
        ),
        (
            "value norm window prefill",
            Case::new(VALUE_NORM, 256, window_rows(32, 100, 40), 32, 1.0, 1.0, 33),
            &[],
        ),
        (
            "shared prefill",
            Case::new(SHARED, 256, shared_rows(18, 70), 32, 1.0, 0.2, 34),
            &[],
        ),
    ]
}

/// Gemma-style full layer: W = 512 (64 rotated pairs, 384 pass-through
/// columns), ungated, value norm, scale 1.
const WIDE_G8: Form = Form {
    kv: 2,
    g: 8,
    p: 64,
    s: 384,
    interleaved: false,
    separate: 0,
    softplus: false,
    fresh: true,
    query_norm: true,
    norm: true,
    value_norm: true,
    rotary: Rotary::HALF_SPLIT_LEADING,
    stored_key: false,
};
const WIDE_G16: Form = Form {
    g: 16,
    kv: 1,
    ..WIDE_G8
};
/// Muse-style: W = 128, G = 16, a separate sigmoid gate per column.
const MUSE: Form = Form {
    kv: 2,
    g: 16,
    p: 64,
    s: 0,
    interleaved: false,
    separate: 128,
    softplus: false,
    fresh: true,
    query_norm: true,
    norm: true,
    value_norm: false,
    rotary: Rotary::HALF_SPLIT_LEADING,
    stored_key: false,
};
/// MiniCPM5-style: W = 128, G = 8, no q/k norm, ungated.
const PLAIN_G8: Form = Form {
    kv: 2,
    g: 8,
    p: 64,
    s: 0,
    interleaved: false,
    separate: 0,
    softplus: false,
    fresh: true,
    query_norm: false,
    norm: false,
    value_norm: false,
    rotary: Rotary::HALF_SPLIT_LEADING,
    stored_key: false,
};

/// Wide heads and large groups at decode and prefill rows (host model only).
fn wide_cases() -> Vec<(&'static str, bool, Case, &'static [BackendName])> {
    vec![
        (
            "wide g8 decode",
            true,
            Case::new(WIDE_G8, 320, decode_rows(260), 128, 1.0, 1.0, 40),
            &[],
        ),
        (
            "wide g16 decode",
            true,
            Case::new(WIDE_G16, 320, decode_rows(260), 128, 1.0, 1.0, 41),
            &[],
        ),
        (
            "muse decode",
            true,
            Case::new(MUSE, 320, decode_rows(260), 128, 1.0, 0.0883883, 42),
            &[],
        ),
        (
            "plain g8 decode",
            true,
            Case::new(PLAIN_G8, 320, decode_rows(260), 128, 1.0, 0.0883883, 43),
            &[],
        ),
        (
            "wide g8 window prefill",
            false,
            Case::new(WIDE_G8, 256, window_rows(24, 100, 40), 64, 1.0, 1.0, 44),
            &[],
        ),
        // More than one key partition's worth of keys: the split and merge.
        // Over 600 keys the score scale is 1/sqrt(W): at scale 1 a softmax this
        // sharp turns a one-step bf16 difference between the host's and the
        // device's rotated query into output differences beyond the bound
        // (the short cases above keep scale 1).
        (
            "sliding long prefill",
            false,
            Case::new(
                Form {
                    p: 128,
                    s: 0,
                    ..WIDE_G8
                },
                700,
                window_rows(20, 600, 1000),
                128,
                1.0,
                0.0625,
                48,
            ),
            &[],
        ),
        (
            "wide g8 long prefill",
            false,
            Case::new(
                WIDE_G8,
                700,
                window_rows(20, 600, 1000),
                128,
                1.0,
                0.0441942,
                49,
            ),
            &[],
        ),
        (
            "wide g16 long prefill",
            false,
            Case::new(
                WIDE_G16,
                700,
                window_rows(20, 600, 1000),
                128,
                1.0,
                0.0441942,
                47,
            ),
            &[],
        ),
        (
            "muse window prefill",
            false,
            Case::new(MUSE, 256, window_rows(24, 100, 40), 64, 1.0, 0.0883883, 45),
            &[],
        ),
        (
            "plain g8 prefill",
            false,
            Case::new(
                PLAIN_G8,
                256,
                window_rows(24, 100, 1000),
                64,
                1.0,
                0.0883883,
                46,
            ),
            &[],
        ),
        // Sinks over wide heads and split key partitions (the merges).
        (
            "wide g8 sinks decode",
            true,
            Case::new(WIDE_G8, 320, decode_rows(260), 128, 1.0, 1.0, 50).with_sinks(60),
            &[],
        ),
        (
            "wide g8 sinks long prefill",
            false,
            Case::new(
                WIDE_G8,
                700,
                window_rows(20, 600, 1000),
                128,
                1.0,
                0.0441942,
                51,
            )
            .with_sinks(61),
            &[],
        ),
        // A Nemotron 3.5 Bolt draft block: 32 query heads over 2 kv heads,
        // whole blocks over a long history.
        (
            "bolt draft sinks decode",
            true,
            Case::new(BOLT_DRAFT, 700, block_rows(8, 600), 128, 1.0, 0.0883883, 52)
                .with_sinks(62),
            &[],
        ),
        (
            "bolt draft sinks prefill",
            false,
            Case::new(BOLT_DRAFT, 700, block_rows(16, 600), 128, 1.0, 0.0883883, 53)
                .with_sinks(63),
            &[],
        ),
    ]
}

/// A Nemotron 3.5 Bolt LiLiCorr draft layer: G = 16, W = 128, full rotation,
/// q/k norms, ungated, with per-head sinks.
const BOLT_DRAFT: Form = Form {
    kv: 2,
    g: 16,
    p: 64,
    s: 0,
    interleaved: false,
    separate: 0,
    softplus: false,
    fresh: true,
    query_norm: true,
    norm: true,
    value_norm: false,
    rotary: Rotary::HALF_SPLIT_LEADING,
    stored_key: false,
};

#[test]
fn wide_natives_match_portable_body() {
    for (name, decode, case, rejections) in wide_cases() {
        check_natives(name, decode, &case, rejections);
    }
}

#[test]
fn prefill_natives_match_portable_body() {
    for (name, decode, case, rejections) in portable_cases() {
        if !decode {
            check_natives(name, false, &case, rejections);
        }
    }
    for (name, case, rejections) in prefill_cases() {
        check_natives(name, false, &case, rejections);
    }
}

// ---------------------------------------------------------------------------
// Rotary layouts, the output rotary and a query norm apart from the key norm
// (`RA`, `RT`, `RO`, `NQ`): computed on CPU and Metal and by CUDA's prefill
// entry, refused by CUDA's decode declaration and by Vulkan's.

/// The backends whose decode entry refuses these forms: CUDA's and
/// Vulkan's.
const DECODE_REFUSALS: &[BackendName] = &[BackendName::Cuda, BackendName::Vulkan];

/// The backends whose prefill entry and dense append refuse these forms:
/// Vulkan's.
const PREFILL_REFUSALS: &[BackendName] = &[BackendName::Vulkan];

/// The backends whose `decode` (else prefill) entry refuses these forms.
fn new_form_refusals(decode: bool) -> &'static [BackendName] {
    if decode { DECODE_REFUSALS } else { PREFILL_REFUSALS }
}

/// A small DeepSeek V4.1-style layer: ungated, a key norm without a query
/// norm, sinks, adjacent pairs over the trailing columns, the attended row
/// turned back, values the stored key row. W = 64, so each lane of the
/// Metal forms holds whole pairs.
const SMALL_V41: Form = Form {
    kv: 1,
    g: 4,
    p: 8,
    s: 48,
    interleaved: false,
    separate: 0,
    softplus: false,
    fresh: true,
    query_norm: false,
    norm: true,
    value_norm: false,
    rotary: Rotary::ADJACENT_TRAILING_INVERSE,
    stored_key: true,
};

/// The layouts one axis at a time, each at the head width where a lane of
/// the Metal forms holds one column (W = 32) or several.
fn layout_cases() -> Vec<(&'static str, bool, Case)> {
    let adjacent_leading = Form {
        kv: 2,
        g: 2,
        p: 4,
        s: 24,
        rotary: Rotary {
            adjacent: true,
            trailing: false,
            inverse_output: false,
        },
        query_norm: true,
        stored_key: false,
        ..SMALL_V41
    };
    let half_split_trailing = Form {
        rotary: Rotary {
            adjacent: false,
            trailing: true,
            inverse_output: true,
        },
        query_norm: true,
        norm: false,
        stored_key: false,
        ..SMALL_V41
    };
    vec![
        (
            "v41 small decode",
            true,
            Case::new(SMALL_V41, 64, decode_rows(5), 16, 1.0, 0.125, 70).with_sinks(80),
        ),
        (
            "v41 small window prefill",
            false,
            Case::new(SMALL_V41, 160, window_rows(20, 30, 8), 16, 1.0, 0.125, 71).with_sinks(81),
        ),
        (
            "adjacent leading decode",
            true,
            Case::new(adjacent_leading, 64, decode_rows(5), 16, 1.0, 0.1767767, 72),
        ),
        (
            "half-split trailing inverse prefill",
            false,
            Case::new(half_split_trailing, 160, window_rows(20, 30, 8), 16, 1.0, 0.125, 73),
        ),
    ]
}

#[test]
fn layout_host_model_matches_portable_body() {
    for (name, decode, case) in layout_cases() {
        let entry = if decode {
            "attention_decode"
        } else {
            "attention_prefill"
        };
        check_host_model(entry, &case);
        eprintln!("{name}: host model agrees with the portable body");
    }
}

#[test]
fn layout_natives_match_portable_body() {
    for (name, decode, case) in layout_cases() {
        check_natives_against(name, decode, &case, new_form_refusals(decode), true, &case.expected());
    }
}

/// DeepSeek V4.1 Flash's window attention (layer 0): 64 query heads over one
/// 512-wide key/value head, its 32 rotary pairs adjacent over dims 448..511
/// at θ 10000, a key norm and no query norm, sinks, the attended row turned
/// back, scale 512^-1/2.
const V41: Form = Form {
    kv: 1,
    g: 64,
    p: 32,
    s: 448,
    ..SMALL_V41
};

impl Case {
    /// This case's rotary table as V4.1's window layers derive it: pair i at
    /// frequency θ^(-2i / 2P) on the position, amplitude 1.
    fn with_theta(mut self, theta: f64) -> Self {
        let p = self.form.p;
        self.components = vec![0; p];
        self.frequencies = (0..p)
            .map(|pair| theta.powf(-((2 * pair) as f64) / (2 * p) as f64) as f32)
            .collect();
        self.amplitudes = vec![1.0; p];
        self
    }

    /// The attention output as DeepSeek V4.1 computes it, in F32 from the
    /// case's inputs, written apart from the host model. The window KV is
    /// one row per token: RMS(kv) * norm, rounded, its trailing 2P dims
    /// turned as P complex numbers (x[S + 2i] + i x[S + 2i + 1]) times
    /// e^(i t θ_i), rounded to bf16 and stored once. Attention reads that
    /// one row as key and value (the value as wide as the key):
    /// softmax(q · kv * scale) over the visible and fresh rows with the
    /// head's sink logit in the denominator, times the same rows; the output
    /// rounded to bf16, its trailing dims times e^(-i t θ_i) (the conjugate
    /// rotation), rounded again. Storing the row as E4M3 is a history
    /// codec's concern, not this dense history's.
    fn v41_reference(&self) -> Vec<f32> {
        let form = self.form;
        assert!(form.kv == 1 && !form.query_norm && form.norm && form.fresh && form.stored_key);
        let (g, p, s, w) = (form.g, form.p, form.s, form.w());
        let twist = |x: &mut [f32], position: i32, sign: f32| {
            for i in 0..p {
                let angle = position as f32 * self.frequencies[i];
                let (re, im) = (angle.cos(), sign * angle.sin());
                let (a, b) = (x[s + 2 * i], x[s + 2 * i + 1]);
                x[s + 2 * i] = a * re - b * im;
                x[s + 2 * i + 1] = a * im + b * re;
            }
        };
        let position = |row: usize| self.coordinates[row * 4];
        let keys = (0..self.rows)
            .map(|row| {
                let raw = &self.key[row * w..][..w];
                let rms = (raw.iter().map(|x| x * x).sum::<f32>() / w as f32 + self.epsilon).sqrt();
                let mut k = raw
                    .iter()
                    .zip(&self.key_norm)
                    .map(|(x, n)| bf16(x / rms * n))
                    .collect::<Vec<_>>();
                twist(&mut k, position(row), 1.0);
                k.into_iter().map(bf16).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut out = vec![0.0; self.rows * g * w];
        for row in 0..self.rows {
            // Each visible row, read as both key and value.
            let mut visible: Vec<(&[f32], &[f32])> = Vec::new();
            for span in 0..self.spans {
                let (lo, hi) = (
                    self.visible[(row * self.spans + span) * 2],
                    self.visible[(row * self.spans + span) * 2 + 1],
                );
                for token in lo.max(0)..hi {
                    let kv = &self.history_key[token as usize * w..][..w];
                    visible.push((kv, kv));
                }
            }
            for token in self.fresh[row * 2].max(0)..self.fresh[row * 2 + 1] {
                let kv = &keys[token as usize][..];
                visible.push((kv, kv));
            }
            // The compressed rows the row's indexer picked (-1 none), each
            // read as key and value after the window's.
            for &pick in &self.selected_rows[row * self.picks..][..self.picks] {
                if pick >= 0 {
                    let kv = &self.selected[pick as usize * w..][..w];
                    visible.push((kv, kv));
                }
            }
            for head in 0..g {
                let mut q = self.query[(row * g + head) * w..][..w].to_vec();
                twist(&mut q, position(row), 1.0);
                let q = q.into_iter().map(bf16).collect::<Vec<_>>();
                let scores = visible
                    .iter()
                    .map(|(k, _)| q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * self.scale)
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores.iter().map(|score| (score - maximum).exp()).collect::<Vec<_>>();
                let sink = self.sinks.get(head).map_or(0.0, |sink| (sink - maximum).exp());
                let denominator = (weights.iter().sum::<f32>() + sink).max(1e-30);
                let mut o = (0..w)
                    .map(|column| {
                        bf16(
                            visible
                                .iter()
                                .zip(&weights)
                                .map(|((_, v), weight)| weight * v[column])
                                .sum::<f32>()
                                / denominator,
                        )
                    })
                    .collect::<Vec<_>>();
                twist(&mut o, position(row), -1.0);
                for (target, value) in out[(row * g + head) * w..][..w].iter_mut().zip(o) {
                    *target = bf16(value);
                }
            }
        }
        out
    }
}

/// V4.1's shapes at a decode step (single and verification rows over a
/// history past the 128-row window's reach) and over window prefill rows.
fn v41_cases() -> Vec<(&'static str, bool, Case)> {
    let scale = 1.0 / (512f32).sqrt();
    vec![
        (
            "v41 decode",
            true,
            Case::new(V41, 320, decode_rows(260), 128, 1.0, scale, 90)
                .with_theta(1.0e4)
                .with_sinks(91),
        ),
        (
            "v41 window prefill",
            false,
            Case::new(V41, 256, window_rows(24, 100, 128), 64, 1.0, scale, 92)
                .with_theta(1.0e4)
                .with_sinks(93),
        ),
    ]
}

/// The host model of the portable body equals the F32 reference of map §1
/// at V4.1's shapes, and so stands for it below.
#[test]
fn v41_host_model_matches_reference() {
    for (name, _, case) in v41_cases() {
        let expected = case.expected();
        check(name, &(case.v41_reference(), expected.1.clone(), expected.2.clone()), &expected, &case);
    }
}

/// At V4.1's shapes the CPU and Metal natives equal the F32 reference: the
/// decode entry on CPU, and the prefill entry at the decode rows on Metal,
/// whose decode declaration does not hold G = 64 at W = 512 (the executor
/// routes those classes through its prefill); prefill rows on both, and on
/// CUDA, whose decode declaration refuses the forms as Vulkan's do.
#[test]
fn v41_natives_match_reference() {
    for (name, decode, case) in v41_cases() {
        let host = case.expected();
        let expected = (case.v41_reference(), host.1, host.2);
        if decode {
            let refusals = [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan];
            check_natives_against(name, true, &case, &refusals, true, &expected);
        }
        check_natives_against(name, false, &case, PREFILL_REFUSALS, true, &expected);
    }
}

// ---------------------------------------------------------------------------
// Stored-key values (`VK = 1`).

/// `rows`, every one appending: a row without a destination takes
/// `first + its index`, so the history holds every row's prepared key.
fn appending(rows: Vec<Row>, first: i32) -> Vec<Row> {
    rows.into_iter()
        .enumerate()
        .map(|(index, row)| Row {
            destination: if row.destination < 0 {
                first + index as i32
            } else {
                row.destination
            },
            ..row
        })
        .collect()
}

impl Case {
    /// This stored-key case over two planes: its own values the rows'
    /// prepared keys `keys` (`[M, KV * W]`, which a value without a value
    /// norm passes through), its value plane a copy of its key plane.
    fn key_valued(&self, keys: Vec<f32>) -> Self {
        assert!(self.form.stored_key && !self.form.value_norm);
        Self {
            form: Form {
                stored_key: false,
                ..self.form
            },
            value: keys,
            history_value: self.history_key.clone(),
            ..self.clone()
        }
    }

    /// Every row's prepared key, from a run's final key plane at the row's
    /// destination.
    fn prepared_keys(&self, history_key: &[f32]) -> Vec<f32> {
        let row = self.form.kv * self.form.w();
        self.destinations
            .iter()
            .flat_map(|destination| {
                let destination = usize::try_from(*destination).expect("every row appends");
                history_key[destination * row..][..row].to_vec()
            })
            .collect()
    }

    /// Every row's prepared key rounded without its holds, in
    /// [`Self::prepared_keys`]'s order.
    fn unheld_keys(&self) -> Vec<f32> {
        let (kv, w) = (self.form.kv, self.form.w());
        (0..self.rows)
            .flat_map(|row| {
                (0..kv).flat_map(move |head| {
                    self.prepare_held(
                        &self.key[(row * kv + head) * w..][..w],
                        self.form.norm.then_some(&self.key_norm[..]),
                        true,
                        row,
                        self.rounding_scalars(),
                        false,
                    )
                })
            })
            .collect()
    }
}

/// Stored-key layers at V4.1's shapes and a small one of its form (decode
/// rows on Metal's decode entry), and a half-split leading one, a form CUDA
/// and Vulkan hold but for its stored-key values.
fn stored_key_cases() -> Vec<(&'static str, bool, Case)> {
    let scale = 1.0 / (512f32).sqrt();
    let qwen = Form {
        stored_key: true,
        ..QWEN_SMALL
    };
    vec![
        (
            "small v41 decode",
            true,
            Case::new(SMALL_V41, 64, appending(decode_rows(5), 47), 16, 1.0, 0.125, 70)
                .with_sinks(80),
        ),
        (
            "small v41 window prefill",
            false,
            Case::new(SMALL_V41, 160, window_rows(20, 30, 8), 16, 1.0, 0.125, 71).with_sinks(81),
        ),
        (
            "v41 decode",
            true,
            Case::new(V41, 320, appending(decode_rows(260), 302), 128, 1.0, scale, 90)
                .with_theta(1.0e4)
                .with_sinks(91),
        ),
        (
            "v41 window prefill",
            false,
            Case::new(V41, 256, window_rows(24, 100, 128), 64, 1.0, scale, 92)
                .with_theta(1.0e4)
                .with_sinks(93),
        ),
        (
            "half-split stored-key decode",
            true,
            Case::new(qwen, 64, appending(decode_rows(5), 47), 16, 1.0, 0.1767767, 74),
        ),
    ]
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Whether `backend` may refuse `case`'s form by declaration: Vulkan holds
/// no stored-key values, nor CUDA's decode entry, and Metal's decode entry
/// no head group of V4.1's width.
fn may_refuse(backend: BackendName, decode: bool, case: &Case) -> bool {
    new_form_refusals(decode).contains(&backend)
        || (backend == BackendName::Metal && decode && case.form.g == 64)
}

/// Attention over the one key plane equals the same inputs over two planes
/// whose values are the prepared keys, bit for bit, on every configuration
/// of the CPU and Metal entries and CUDA's prefill entry: the result, and
/// the key plane the row's keys are appended to. The two-plane run's value
/// plane ends equal to its key plane. Vulkan and CUDA's decode entry refuse
/// every stored-key form.
#[test]
fn stored_key_attention_equals_key_valued_planes() {
    let mut failures = Vec::new();
    for (name, decode, case) in stored_key_cases() {
        for device in devices() {
            let backend = device.backend();
            for config in configurations(&device, decode, case.form) {
                let specialization = |form| {
                    config
                        .iter()
                        .fold(statics(&device, form, case.sink_rows()), |s, (n, v)| {
                            s.with_param(*n, *v)
                        })
                };
                let label = format!("{name} {backend:?} {config:?}");
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let stored = match run(&device, decode, &specialization(case.form), &case) {
                        Err(error) if error.contains("violates the native `where` condition") => {
                            assert!(may_refuse(backend, decode, &case), "{label}: {error}");
                            return;
                        }
                        Err(error) => panic!("{label}: {error}"),
                        Ok(_) if new_form_refusals(decode).contains(&backend) => {
                            panic!("{label}: admitted stored-key values")
                        }
                        Ok(stored) => stored,
                    };
                    assert!(stored.2.is_empty(), "{label}: a value plane");
                    let twin = case.key_valued(case.prepared_keys(&stored.1));
                    let planes = run(&device, decode, &specialization(twin.form), &twin)
                        .unwrap_or_else(|error| panic!("{label} over two planes: {error}"));
                    assert_eq!(bits(&stored.0), bits(&planes.0), "{label}: result");
                    assert_eq!(bits(&stored.1), bits(&planes.1), "{label}: key plane");
                    assert_eq!(bits(&planes.2), bits(&planes.1), "{label}: value plane");
                    eprintln!("{label}: one plane equals two");
                }));
                if outcome.is_err() {
                    failures.push(label);
                }
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}

/// The dense append of `case` on `device`: its final key and value planes.
fn run_append(device: &Device, case: &Case) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut bound = Bound::new(device, case);
    let form = case.form;
    let [_, ra, rt, _, vk, kr] = case.extents();
    let append_form = Tensor::from_host(
        device,
        Element::i32(),
        &[0, ra as u64, rt as u64, vk as u64, kr as u64],
        &[],
    )
    .unwrap();
    // The appends read no query-head, gate, sink, output-rotary or
    // index-list axis.
    let specialization = statics(device, form, 0)
        .statics()
        .iter()
        .filter(|(name, _)| !matches!(name.as_str(), "G" | "I" | "U" | "NQ" | "RO" | "SK"))
        .fold(NativeSpecialization::new(), |s, (name, value)| {
            s.with_static(name, *value)
        })
        .with_static("KR", kr as u64);
    let rounding = case.rounding_scalars();
    let kernel = attention_append_dense::native_for_device_with(
        device,
        attention_append_dense::Elements { A: Element::bf16() },
        &specialization,
    )
    .map_err(|error| error.to_string())?;
    kernel
        .call(attention_append_dense::Args {
            key: &bound.key,
            value: &bound.value,
            key_norm: &bound.key_norm,
            value_norm: &bound.value_norm,
            rotary_components: &bound.components,
            rotary_frequencies: &bound.frequencies,
            rotary_amplitudes: &bound.amplitudes,
            coordinates: &bound.coordinates,
            destinations: &bound.destinations,
            history_key: &mut bound.history_key,
            history_value: &mut bound.history_value,
            form: &append_form,
            epsilon: case.epsilon,
            slab_rows: case.slab_rows as u32,
            key_block: rounding.block,
            key_element: rounding.element,
            key_scale: rounding.scale,
            key_amax_floor: rounding.amax_floor,
        })
        .map_err(|error| format!("{error:?}"))?;
    Ok((bf16_values(&bound.history_key), bf16_values(&bound.history_value)))
}

/// The dense append of a stored-key layer writes the key plane alone, and
/// it equals the two-plane append of the same rows with the prepared keys
/// as values, bit for bit; Vulkan refuses it.
#[test]
fn stored_key_append_writes_the_key_plane_alone() {
    let mut failures = Vec::new();
    for (name, _, case) in stored_key_cases() {
        for device in devices() {
            let backend = device.backend();
            let label = format!("{name} append {backend:?}");
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let stored = match run_append(&device, &case) {
                    Err(error) if error.contains("violates the native `where` condition") => {
                        assert!(PREFILL_REFUSALS.contains(&backend), "{label}: {error}");
                        return;
                    }
                    Err(error) => panic!("{label}: {error}"),
                    Ok(_) if PREFILL_REFUSALS.contains(&backend) => {
                        panic!("{label}: admitted stored-key values")
                    }
                    Ok(stored) => stored,
                };
                assert!(stored.1.is_empty(), "{label}: a value plane");
                // The appended keys are the host model's, within a bf16 step.
                let expected = case.expected().1;
                for (index, (a, e)) in stored.0.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - e).abs() <= 8.0e-3 * e.abs().max(1.0),
                        "{label}: history_key[{index}] {a}, host model {e}"
                    );
                }
                let twin = case.key_valued(case.prepared_keys(&stored.0));
                let planes = run_append(&device, &twin)
                    .unwrap_or_else(|error| panic!("{label} over two planes: {error}"));
                assert_eq!(bits(&stored.0), bits(&planes.0), "{label}: key plane");
                assert_eq!(bits(&planes.1), bits(&planes.0), "{label}: value plane");
                eprintln!("{label}: one plane equals two");
            }));
            if outcome.is_err() {
                failures.push(label);
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}

/// The portable bodies of a stored-key layer equal the host model, which
/// reads the key plane as values and appends the key alone.
#[test]
fn stored_key_host_model_matches_portable_body() {
    for (name, decode, case) in stored_key_cases() {
        // The interpreter at the small shapes; the host model stands for it
        // at V4.1's (`v41_host_model_matches_reference`).
        if case.form.w() > 64 {
            continue;
        }
        let entry = if decode {
            "attention_decode"
        } else {
            "attention_prefill"
        };
        check_host_model(entry, &case);
        eprintln!("{name}: host model agrees with the portable body");
    }
}

// ---------------------------------------------------------------------------
// The index-list source (`KS`, `selected_rows` over `selected`) and the
// trained key rounding (`KR`).

/// V4.1's compressed-attention layers: the window's rows and KS = 512
/// compressed rows a row (every seventh -1) of a 1000-row plane in slabs of
/// 256, which the picks cross, read as key and value in the same softmax;
/// at a decode step and over window prefill rows.
fn two_source_cases() -> Vec<(&'static str, bool, Case)> {
    let mut cases = v41_cases().into_iter();
    let (_, _, decode) = cases.next().unwrap();
    let (_, _, prefill) = cases.next().unwrap();
    vec![
        ("v41 two-source decode", true, decode.with_picks(512, 1000, 256, 94)),
        ("v41 two-source window prefill", false, prefill.with_picks(512, 1000, 256, 95)),
    ]
}

/// The two-source form at the interpreter's shapes: SMALL_V41 with 24 picks
/// a row of a 40-row plane in slabs of 16.
fn small_two_source_cases() -> Vec<(&'static str, bool, Case)> {
    vec![
        (
            "small v41 two-source decode",
            true,
            Case::new(SMALL_V41, 64, decode_rows(5), 16, 1.0, 0.125, 70)
                .with_sinks(80)
                .with_picks(24, 40, 16, 96),
        ),
        (
            "small v41 two-source window prefill",
            false,
            Case::new(SMALL_V41, 160, window_rows(20, 30, 8), 16, 1.0, 0.125, 71)
                .with_sinks(81)
                .with_picks(24, 40, 16, 97),
        ),
    ]
}

/// The backends whose declarations refuse an index-list source or a key
/// rounding: Vulkan everywhere, the CUDA and Metal decode entries too.
fn two_source_refusals(decode: bool) -> &'static [BackendName] {
    if decode {
        &[BackendName::Metal, BackendName::Cuda, BackendName::Vulkan]
    } else {
        PREFILL_REFUSALS
    }
}

/// The portable bodies of the two-source form equal the host model, which
/// reads each pick (none for -1) as key and value after the row's fresh
/// span.
#[test]
fn two_source_host_model_matches_portable_body() {
    for (name, decode, case) in small_two_source_cases() {
        check_host_model(if decode { "attention_decode" } else { "attention_prefill" }, &case);
        eprintln!("{name}: host model agrees with the portable body");
    }
}

/// At V4.1's shapes the host model of the two-source form equals the F32
/// reference, which reads the picked compressed rows after the window's.
#[test]
fn two_source_host_model_matches_reference() {
    for (name, _, case) in two_source_cases() {
        let expected = case.expected();
        check(name, &(case.v41_reference(), expected.1.clone(), expected.2.clone()), &expected, &case);
    }
}

/// The CPU and Metal natives of the two-source form equal the host model at
/// the small shapes and the F32 reference at V4.1's: the decode entry on
/// CPU, the prefill entry (decode rows too) on CPU and Metal; the other
/// declarations refuse it.
#[test]
fn two_source_natives_match_reference() {
    for (name, decode, case) in small_two_source_cases() {
        if decode {
            check_natives_against(name, true, &case, two_source_refusals(true), true, &case.expected());
        }
        check_natives_against(name, false, &case, two_source_refusals(false), true, &case.expected());
    }
    for (name, decode, case) in two_source_cases() {
        let host = case.expected();
        let expected = (case.v41_reference(), host.1, host.2);
        if decode {
            check_natives_against(name, true, &case, two_source_refusals(true), true, &expected);
        }
        check_natives_against(name, false, &case, two_source_refusals(false), true, &expected);
    }
}

/// A two-source row whose picks are all -1 reads the window alone: its
/// natives' results and histories are held, within the precision bar, to
/// the host model of the one-source run (KS = 0, the entries' form before
/// the source).
#[test]
fn two_source_without_picks_equals_one_source() {
    let mut failures = Vec::new();
    for (name, decode, case) in two_source_cases() {
        let mut empty = case.clone();
        empty.selected_rows.fill(-1);
        let one_source = Case {
            picks: 0,
            selected_rows: Vec::new(),
            selected: Vec::new(),
            selected_slab_rows: 1,
            ..case.clone()
        };
        let expected = one_source.expected();
        for device in devices() {
            let backend = device.backend();
            if !matches!(backend, BackendName::Cpu | BackendName::Metal) {
                continue;
            }
            // The decode rows through both entries; Metal's decode entry
            // holds neither form at V4.1's group.
            let entries = if decode { vec![true, false] } else { vec![false] };
            for decode in entries.into_iter().filter(|d| !(*d && backend == BackendName::Metal)) {
                for config in case_configurations(&device, decode, &empty) {
                    let specialization = config
                        .iter()
                        .fold(statics(&device, empty.form, empty.sink_rows()), |s, (n, v)| s.with_param(*n, *v));
                    let label = format!("{name} {backend:?} decode {decode} {config:?}");
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let picked = run(&device, decode, &specialization, &empty)
                            .unwrap_or_else(|error| panic!("{label}: {error}"));
                        check(&label, &picked, &expected, &one_source);
                    }));
                    if outcome.is_err() {
                        failures.push(label);
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}

/// V4.1's window-key rounding: E4M3 elements in runs of 32 with UE8M0
/// (power-of-two) scales, which leaves every rounded value at most three
/// explicit mantissa bits.
const KEY_ROUNDING: Rounding = rounding_model::WINDOW;

/// Whether every value of `keys` has at most three explicit mantissa bits
/// (an E4M3 code times a power of two), and some is nonzero: the BF16 keys
/// before rounding carry seven.
fn on_rounding_grid(keys: &[f32]) -> bool {
    keys.iter().all(|key| key.to_bits() & 0x000f_ffff == 0) && keys.iter().any(|key| *key != 0.0)
}

/// Rounded keys `actual` agree with `expected`: identical but for at most
/// one in 1024, which may sit one E4M3 step away (an F32 input on the other
/// side of a rounding boundary).
fn rounded_keys_agree(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: key count");
    let differing = actual
        .iter()
        .zip(expected)
        .filter(|(a, e)| a.to_bits() != e.to_bits())
        .inspect(|(a, e)| {
            assert!((*a - *e).abs() <= 0.25 * a.abs().max(e.abs()), "{label}: rounded key {a} vs {e}");
        })
        .count();
    assert!(differing * 1024 <= actual.len(), "{label}: {differing} of {} rounded keys differ", actual.len());
}

/// The key-rounding form at the interpreter's shapes: every row appends.
fn small_key_rounding_cases() -> Vec<(&'static str, bool, Case)> {
    vec![
        (
            "small v41 rounded decode",
            true,
            Case::new(SMALL_V41, 64, appending(decode_rows(5), 47), 16, 1.0, 0.125, 72)
                .with_sinks(82)
                .with_key_rounding(KEY_ROUNDING),
        ),
        (
            "small v41 rounded window prefill",
            false,
            Case::new(SMALL_V41, 160, window_rows(20, 30, 8), 16, 1.0, 0.125, 73)
                .with_sinks(83)
                .with_key_rounding(KEY_ROUNDING),
        ),
    ]
}

/// The rounded portable bodies equal the host model, which holds each
/// prepared key in BF16 after its norm and after its rotation and rounds it
/// with the independent rounding model, and their appended keys lie on the
/// rounding's grid; the holds move some rounded key. The CPU and Metal
/// natives, and the CPU, Metal and CUDA dense appends, store the host
/// model's rounded keys, and the natives' results are the host model's over
/// the keys they stored; the other declarations refuse the form.
#[test]
fn key_rounding_matches_portable_body() {
    let (mut failures, mut moved) = (Vec::new(), 0);
    for (name, decode, case) in small_key_rounding_cases() {
        let entry = if decode { "attention_decode" } else { "attention_prefill" };
        check_host_model(entry, &case);
        let keys = case.prepared_keys(&case.expected().1);
        assert!(on_rounding_grid(&keys), "{name}: rounded keys off the rounding grid");
        moved += case.unheld_keys().iter().zip(&keys).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        for device in devices() {
            let backend = device.backend();
            let refused = two_source_refusals(decode).contains(&backend);
            for config in case_configurations(&device, decode, &case) {
                let specialization = config
                    .iter()
                    .fold(statics(&device, case.form, case.sink_rows()), |s, (n, v)| s.with_param(*n, *v));
                let label = format!("{name} {backend:?} {config:?}");
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    match run(&device, decode, &specialization, &case) {
                        Err(error) if error.contains("violates the native `where` condition") => {
                            assert!(refused, "{label}: {error}");
                        }
                        Err(error) => panic!("{label}: {error}"),
                        Ok(_) if refused => panic!("{label}: admitted a key rounding"),
                        Ok(actual) => {
                            let own = case.prepared_keys(&actual.1);
                            rounded_keys_agree(&label, &own, &keys);
                            check(&label, &actual, &case.expected_with(Some(&own)), &case);
                        }
                    }
                }));
                if outcome.is_err() {
                    failures.push(label);
                }
            }
            let label = format!("{name} append {backend:?}");
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match run_append(&device, &case) {
                    Err(error) if error.contains("violates the native `where` condition") => {
                        assert!(PREFILL_REFUSALS.contains(&backend), "{label}: {error}");
                    }
                    Err(error) => panic!("{label}: {error}"),
                    Ok(appended) => rounded_keys_agree(&label, &case.prepared_keys(&appended.0), &keys),
                }
            }));
            if outcome.is_err() {
                failures.push(label);
            }
        }
    }
    assert!(moved > 0, "the holds move no rounded key");
    assert!(failures.is_empty(), "failed: {failures:?}");
}

/// V4.1's compressed-attention layers whole: the window's keys rounded
/// (every row appends) and the index-list source.
fn v41_compressed_cases() -> Vec<(&'static str, bool, Case)> {
    let scale = 1.0 / (512f32).sqrt();
    vec![
        (
            "v41 compressed decode",
            true,
            Case::new(V41, 320, appending(decode_rows(260), 302), 128, 1.0, scale, 90)
                .with_theta(1.0e4)
                .with_sinks(91)
                .with_picks(512, 1000, 256, 98)
                .with_key_rounding(KEY_ROUNDING),
        ),
        (
            "v41 compressed window prefill",
            false,
            Case::new(V41, 256, window_rows(24, 100, 128), 64, 1.0, scale, 92)
                .with_theta(1.0e4)
                .with_sinks(93)
                .with_picks(512, 1000, 256, 99)
                .with_key_rounding(KEY_ROUNDING),
        ),
    ]
}

/// At V4.1's shapes each CPU, Metal and CUDA native its declaration admits
/// stores the host model's rounded keys (on the rounding's grid), and its
/// result is the host model's over the keys it stored and its picks; so does
/// each dense append its declaration admits (a window run of 32 spans two
/// lanes of a 512-wide row).
#[test]
fn v41_compressed_natives_round_and_attend() {
    let mut failures = Vec::new();
    for (name, decode, case) in v41_compressed_cases() {
        let rounded = case.prepared_keys(&case.expected().1);
        assert!(on_rounding_grid(&rounded), "{name}: rounded keys off the rounding grid");
        for device in devices() {
            let backend = device.backend();
            for decode in if decode { vec![true, false] } else { vec![false] } {
                let refused = two_source_refusals(decode).contains(&backend);
                for config in case_configurations(&device, decode, &case) {
                    let specialization = config
                        .iter()
                        .fold(statics(&device, case.form, case.sink_rows()), |s, (n, v)| s.with_param(*n, *v));
                    let label = format!("{name} {backend:?} decode {decode} {config:?}");
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        match run(&device, decode, &specialization, &case) {
                            Err(error) if error.contains("violates the native `where` condition") => {
                                assert!(refused, "{label}: {error}");
                            }
                            Err(error) => panic!("{label}: {error}"),
                            Ok(_) if refused => panic!("{label}: admitted the compressed form"),
                            Ok(actual) => {
                                let own = case.prepared_keys(&actual.1);
                                rounded_keys_agree(&label, &own, &rounded);
                                check(&label, &actual, &case.expected_with(Some(&own)), &case);
                            }
                        }
                    }));
                    if outcome.is_err() {
                        failures.push(label);
                    }
                }
            }
            let label = format!("{name} append {backend:?}");
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match run_append(&device, &case) {
                    Err(error) if error.contains("violates the native `where` condition") => {
                        assert!(PREFILL_REFUSALS.contains(&backend), "{label}: {error}");
                    }
                    Err(error) => panic!("{label}: {error}"),
                    Ok(appended) => rounded_keys_agree(&label, &case.prepared_keys(&appended.0), &rounded),
                }
            }));
            if outcome.is_err() {
                failures.push(label);
            }
        }
    }
    assert!(failures.is_empty(), "failed: {failures:?}");
}
