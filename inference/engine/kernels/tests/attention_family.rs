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

use magnitude_kernels::{attention_decode, attention_prefill};
use seismic::{
    BackendName, Device, DeviceCatalog, Element, NativeSpecialization, SlabRegion, SlabTensor,
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
    norm: bool,
    value_norm: bool,
}

impl Form {
    fn w(&self) -> usize {
        2 * self.p + self.s
    }
    fn heads(&self) -> usize {
        self.kv * self.g
    }
    fn i(&self) -> usize {
        if self.interleaved {
            self.w()
        } else {
            0
        }
    }
    /// The gate values of one query head.
    fn gates(&self) -> usize {
        self.i().max(self.separate)
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
        let value = values(if form.fresh { m * form.kv * w } else { 0 }, 1.0);
        let history_key = values(history_rows * form.kv * w, 3.0);
        let history_value = values(history_rows * form.kv * w, 1.0);
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
        }
    }

    /// The portable `head_rotary` (or `head_norm` with `rotate` false) of one
    /// head row, rounded to bf16.
    fn prepare(&self, raw: &[f32], norm: Option<&[f32]>, rotate: bool, row: usize) -> Vec<f32> {
        let (p, w) = (if rotate { self.form.p } else { 0 }, self.form.w());
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
        (0..w)
            .map(|i| {
                if i >= 2 * p {
                    return normalized[i];
                }
                let pair = i % p;
                let coordinate = self.coordinates[row * 4 + self.components[pair] as usize];
                let angle = coordinate as f32 * self.frequencies[pair];
                let (c, s) = (
                    angle.cos() * self.amplitudes[pair],
                    angle.sin() * self.amplitudes[pair],
                );
                if i < p {
                    normalized[i] * c - normalized[i + p] * s
                } else {
                    normalized[i] * c + normalized[i - p] * s
                }
            })
            .map(bf16)
            .collect()
    }

    /// The portable body's result and final histories, computed on the host.
    fn expected(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let form = self.form;
        let (kv, g, w, i) = (form.kv, form.g, form.w(), form.i());
        fn norm(form: Form, weights: &[f32]) -> Option<&[f32]> {
            form.norm.then_some(weights)
        }
        let norm = |weights| norm(form, weights);
        let fresh_keys = (0..if form.fresh { self.rows } else { 0 })
            .map(|row| {
                (0..kv)
                    .map(|head| {
                        self.prepare(
                            &self.key[(row * kv + head) * w..][..w],
                            norm(&self.key_norm),
                            true,
                            row,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let fresh_values = (0..if form.fresh { self.rows } else { 0 })
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
            .collect::<Vec<_>>();
        let mut result = vec![0.0; self.rows * kv * g * w];
        for row in 0..self.rows {
            for head in 0..kv * g {
                let kv_head = head / g;
                let query = self.prepare(
                    &self.query[(row * kv * g + head) * (w + i)..][..w],
                    norm(&self.query_norm),
                    true,
                    row,
                );
                let mut entries: Vec<(&[f32], &[f32])> = Vec::new();
                for span in 0..self.spans {
                    let lo = self.visible[(row * self.spans + span) * 2];
                    let hi = self.visible[(row * self.spans + span) * 2 + 1];
                    for token in lo.max(0)..hi {
                        let at = (token as usize * kv + kv_head) * w;
                        entries
                            .push((&self.history_key[at..][..w], &self.history_value[at..][..w]));
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
                let denominator = weights.iter().sum::<f64>().max(1e-30);
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
                history_value[at..at + w].copy_from_slice(&fresh_values[row][head]);
            }
        }
        (result, history_key, history_value)
    }
}

// ---------------------------------------------------------------------------
// The reference interpreter.

/// Checks the host model against the portable body of `entry`.
fn check_host_model(entry: &str, case: &Case) {
    use seismic_lang::{
        checked::{check_source, SourceFile},
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
    let module = check_source(sources).unwrap();
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
    let args = vec![
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
        tensor(DType::BF16, vec![f, m, kv * w], floats(&case.value)),
        tensor(DType::F32, vec![n, w], rows(&case.query_norm, n)),
        tensor(DType::F32, vec![n, w], rows(&case.key_norm, n)),
        tensor(DType::F32, vec![nv, w], rows(&case.value_norm, nv)),
        tensor(DType::I32, vec![form.p], ints(&case.components)),
        tensor(DType::F32, vec![form.p], floats(&case.frequencies)),
        tensor(DType::F32, vec![form.p], floats(&case.amplitudes)),
        tensor(DType::I32, vec![m, 4], ints(&case.coordinates)),
        tensor(DType::I32, vec![m, case.spans, 2], ints(&case.visible)),
        tensor(DType::I32, vec![m, 2], ints(&case.fresh)),
        tensor(DType::I32, vec![m], ints(&case.destinations)),
        tensor(DType::BF16, vec![t, kv, w], floats(&case.history_key)),
        tensor(DType::BF16, vec![t, kv, w], floats(&case.history_value)),
        Arg::Scalar(ReferenceScalar::F32(case.epsilon.to_bits())),
        Arg::Scalar(ReferenceScalar::F32(case.scale.to_bits())),
        Arg::Scalar(ReferenceScalar::I32(i32::from(form.softplus))),
        Arg::Scalar(ReferenceScalar::U32(t as u32)),
    ];
    let outcome = interpreter.run(&args).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("{entry} portable body failed: {failure}");
    }
    let expected = case.expected();
    let result = outcome.results().next().unwrap();
    let OutcomeValue::Tensor(tensor) = result.value() else {
        panic!("{entry} returns a tensor")
    };
    for (index, e) in expected.0.iter().enumerate() {
        let a = tensor.read(index).unwrap() as f32;
        assert!(
            (a - e).abs() <= 1.0e-3 + 8.0e-3 * e.abs(),
            "{entry} {form:?}: portable result[{index}] {a}, host model {e}"
        );
    }
    let inputs = outcome.inputs().collect::<Vec<_>>();
    for (ordinal, expected, name) in [
        (14, &expected.1, "history_key"),
        (15, &expected.2, "history_value"),
    ] {
        let input = inputs
            .iter()
            .find(|input| input.ordinal() == ordinal)
            .unwrap();
        let tensor = input.tensor();
        for (index, e) in expected.iter().enumerate() {
            let a = tensor.read(index).unwrap() as f32;
            assert!(
                (a - e).abs() <= 8.0e-3 * e.abs().max(1.0),
                "{entry} {form:?}: portable {name}[{index}] {a}, host model {e}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Devices.

fn devices() -> Vec<Device> {
    let catalog = DeviceCatalog::discover().unwrap();
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
    components: Tensor,
    frequencies: Tensor,
    amplitudes: Tensor,
    coordinates: Tensor,
    visible: Tensor,
    fresh: Tensor,
    destinations: Tensor,
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
        let (f, n, nv) = (
            usize::from(form.fresh),
            usize::from(form.norm),
            usize::from(form.value_norm),
        );
        let slab_rows = case.slab_rows as u64;
        let region = || SlabRegion {
            element: Element::bf16(),
            row_shape: vec![kv as u64, w as u64],
        };
        let mut slabs =
            SlabTensor::new(device, slab_rows, t as u64, vec![region(), region()]).unwrap();
        let row_bytes = kv * w * 2;
        let planes = [&case.history_key, &case.history_value].map(|values| {
            values
                .iter()
                .flat_map(|v| bf16_bits(*v).to_le_bytes())
                .collect::<Vec<_>>()
        });
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
        Self {
            query: bf16_tensor(device, &[m, heads, w + form.i()], &case.query),
            gate: bf16_tensor(device, &[m, heads, form.separate], &case.gate),
            key: bf16_tensor(device, &[f, m, kv * w], &case.key),
            value: bf16_tensor(device, &[f, m, kv * w], &case.value),
            query_norm: norm(&case.query_norm, n),
            key_norm: norm(&case.key_norm, n),
            value_norm: norm(&case.value_norm, nv),
            components: i32_tensor(device, &[form.p], &case.components),
            frequencies: f32_tensor(device, &[form.p], &case.frequencies),
            amplitudes: f32_tensor(device, &[form.p], &case.amplitudes),
            coordinates: i32_tensor(device, &[m, 4], &case.coordinates),
            visible: i32_tensor(device, &[m, case.spans, 2], &case.visible),
            fresh: i32_tensor(device, &[m, 2], &case.fresh),
            destinations: i32_tensor(device, &[m], &case.destinations),
            history_key: slabs.logical_region(0).unwrap(),
            history_value: slabs.logical_region(1).unwrap(),
            _slabs: slabs,
        }
    }
}

/// The statics of `form` on `device` (CPU fixes only its declared ones).
fn statics(device: &Device, form: Form) -> NativeSpecialization {
    let common = [
        ("P", form.p),
        ("S", form.s),
        ("I", form.i()),
        ("U", form.separate),
        ("F", usize::from(form.fresh)),
        ("N", usize::from(form.norm)),
        ("NV", usize::from(form.value_norm)),
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
                vec![("QT", 16), ("HEADS", heads), ("SPLIT_GROUPS", 256)],
                vec![("QT", 8), ("HEADS", heads), ("SPLIT_GROUPS", 1)],
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
                ],
                vec![
                    ("PARTS", 48),
                    ("WARPS", 8),
                    ("SLICES", slices),
                    ("MATRIX", 0),
                    ("STAGES", 2),
                    ("COLUMNS", 1),
                ],
            ];
            configurations.extend(cuda_matrix_configurations(
                form,
                &[(12, 1, 2, 4), (48, 2, 3, 2), (24, 4, 4, 1)],
            ));
            configurations
        }
        (BackendName::Cuda, false) => {
            let mut configurations = vec![
                vec![
                    ("WARPS", 4),
                    ("SPLIT_GROUPS", 1),
                    ("STAGES", 2),
                    ("COLUMNS", 1),
                    ("QREG", 0),
                ],
                vec![
                    ("WARPS", 2),
                    ("SPLIT_GROUPS", 256),
                    ("STAGES", 2),
                    ("COLUMNS", 2),
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

/// Runs the decode or prefill entry of `case` with `specialization`; the
/// result and final histories, or the preparation error.
fn run(
    device: &Device,
    decode: bool,
    specialization: &NativeSpecialization,
    case: &Case,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let mut bound = Bound::new(device, case);
    macro_rules! call {
        ($module:ident) => {{
            let kernel = $module::native_for_device_with(
                device,
                $module::Elements { A: Element::bf16() },
                specialization,
            )
            .map_err(|error| error.to_string())?;
            kernel
                .call($module::Args {
                    query: &bound.query,
                    gate: &bound.gate,
                    key: &bound.key,
                    value: &bound.value,
                    query_norm: &bound.query_norm,
                    key_norm: &bound.key_norm,
                    value_norm: &bound.value_norm,
                    rotary_components: &bound.components,
                    rotary_frequencies: &bound.frequencies,
                    rotary_amplitudes: &bound.amplitudes,
                    coordinates: &bound.coordinates,
                    visible: &bound.visible,
                    fresh: &bound.fresh,
                    destinations: &bound.destinations,
                    history_key: &mut bound.history_key,
                    history_value: &mut bound.history_value,
                    epsilon: case.epsilon,
                    scale: case.scale,
                    gate_function: i32::from(case.form.softplus),
                    slab_rows: case.slab_rows as u32,
                })
                .map_err(|error| format!("{error:?}"))?
                .value
        }};
    }
    let result = if decode {
        call!(attention_decode)
    } else {
        call!(attention_prefill)
    };
    Ok((
        bf16_values(&result),
        bf16_values(&bound.history_key),
        bf16_values(&bound.history_value),
    ))
}

/// Results agree within bf16 publication plus reduced-precision operands;
/// histories exactly but keys, which may differ by one bf16 step.
fn check(
    label: &str,
    actual: &(Vec<f32>, Vec<f32>, Vec<f32>),
    expected: &(Vec<f32>, Vec<f32>, Vec<f32>),
    w: usize,
) {
    let mut worst = 0.0f32;
    assert_eq!(actual.0.len(), expected.0.len(), "{label}: result shape");
    let outside: Vec<(usize, f32, f32)> = actual
        .0
        .iter()
        .zip(&expected.0)
        .enumerate()
        .inspect(|(_, (a, e))| worst = worst.max((*a - *e).abs()))
        .filter(|(_, (a, e))| (*a - *e).abs() > 4.0e-3 + 1.6e-2 * e.abs())
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
    eprintln!("{label}: max |result error| {worst:.2e}");
}

/// Every device's natives of the decode or prefill entry against the host
/// model. A form a declaration's `where` rejects is reported, not run: it is
/// the typed plan-time refusal, allowed only on the backends `rejections`
/// lists (for some or all of their configurations).
fn check_natives(name: &str, decode: bool, case: &Case, rejections: &[BackendName]) {
    let expected = case.expected();
    let mut failures = Vec::new();
    for device in devices() {
        for config in configurations(&device, decode, case.form) {
            let specialization = config
                .iter()
                .fold(statics(&device, case.form), |s, (n, v)| {
                    s.with_param(*n, *v)
                });
            let label = format!("{name} {:?} {config:?}", device.backend());
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match run(&device, decode, &specialization, case) {
                    Ok(actual) => check(&label, &actual, &expected, case.form.w()),
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
    norm: true,
    value_norm: false,
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
    norm: false,
    value_norm: false,
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
    norm: true,
    value_norm: false,
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
    norm: true,
    value_norm: false,
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
    norm: true,
    value_norm: true,
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
    norm: true,
    value_norm: false,
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
    norm: true,
    value_norm: false,
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
    norm: true,
    value_norm: true,
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
    norm: true,
    value_norm: false,
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
    norm: false,
    value_norm: false,
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
    ]
}

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
// Decode timings (`--ignored decode_timings`): one decode row over a long
// visible history at production head shapes, the bindings prepared once.

/// Qwen3.5 4B: interleaved sigmoid gates, partial M-RoPE.
const QWEN_4B: Form = Form {
    kv: 4,
    g: 4,
    p: 32,
    s: 192,
    interleaved: true,
    separate: 0,
    softplus: false,
    fresh: true,
    norm: true,
    value_norm: false,
};
/// Gemma 4 31B full layer: W = 512, 4 kv heads of 8 query heads, value norm.
const GEMMA_FULL: Form = Form {
    kv: 4,
    g: 8,
    p: 256,
    s: 0,
    interleaved: false,
    separate: 0,
    softplus: false,
    fresh: true,
    norm: true,
    value_norm: true,
};
/// Gemma 4 31B sliding layer: W = 256, 16 kv heads of 2.
const GEMMA_SLIDING: Form = Form {
    kv: 16,
    g: 2,
    p: 128,
    s: 0,
    ..GEMMA_FULL
};
/// Muse Glimmer 30B: W = 128, 2 kv heads of 16, a separate sigmoid gate per
/// column.
const MUSE_30B: Form = Form {
    kv: 2,
    g: 16,
    p: 64,
    s: 0,
    interleaved: false,
    separate: 128,
    softplus: false,
    fresh: true,
    norm: true,
    value_norm: false,
};

/// One decode row at position `context` seeing the last `window` history
/// rows (all of them when `window` is at least `context`).
fn timing_case(form: Form, context: usize, window: usize) -> Case {
    let (context, window) = (context as i32, window as i32);
    let row = Row {
        spans: vec![((context - window + 1).max(0), context)],
        fresh: (0, 1),
        destination: context,
        position: context,
    };
    Case::new(form, context as usize + 1, vec![row], 512, 1.0, 0.0625, 90)
}

/// The median device time (microseconds) of one decode call.
fn time_decode(
    device: &Device,
    specialization: &NativeSpecialization,
    case: &Case,
) -> Result<f64, String> {
    let mut bound = Bound::new(device, case);
    let kernel = attention_decode::native_for_device_with(
        device,
        attention_decode::Elements { A: Element::bf16() },
        specialization,
    )
    .map_err(|error| error.to_string())?;
    let options = seismic::MeasureOptions {
        samples: 15,
        min_sample_seconds: 0.005,
    };
    let seconds = kernel
        .measure(
            vec![attention_decode::Args {
                query: &bound.query,
                gate: &bound.gate,
                key: &bound.key,
                value: &bound.value,
                query_norm: &bound.query_norm,
                key_norm: &bound.key_norm,
                value_norm: &bound.value_norm,
                rotary_components: &bound.components,
                rotary_frequencies: &bound.frequencies,
                rotary_amplitudes: &bound.amplitudes,
                coordinates: &bound.coordinates,
                visible: &bound.visible,
                fresh: &bound.fresh,
                destinations: &bound.destinations,
                history_key: &mut bound.history_key,
                history_value: &mut bound.history_value,
                epsilon: case.epsilon,
                scale: case.scale,
                gate_function: 0,
                slab_rows: case.slab_rows as u32,
            }],
            &options,
        )
        .map_err(|error| format!("{error:?}"))?
        .median;
    Ok(seconds * 1.0e6)
}

#[test]
#[ignore]
fn decode_timings() {
    let shapes = [
        ("qwen 4b full 16k", QWEN_4B, 16384, 16384),
        ("gemma 31b full 16k", GEMMA_FULL, 16384, 16384),
        ("gemma 31b sliding 1024 at 16k", GEMMA_SLIDING, 16384, 1024),
        ("muse 30b full 16k", MUSE_30B, 16384, 16384),
        ("muse 30b sliding 2048 at 16k", MUSE_30B, 16384, 2048),
    ];
    for (name, form, context, window) in shapes {
        let case = timing_case(form, context, window);
        let keys = window.min(context);
        let history_bytes = (2 * keys * form.kv * form.w() * 2) as f64;
        for device in devices() {
            // The fastest admissible configuration, as the tuner would pick.
            let mut best: Option<(f64, Vec<(&str, u64)>)> = None;
            for config in timing_configurations(&device, form) {
                let specialization = config
                    .iter()
                    .fold(statics(&device, form), |s, (n, v)| s.with_param(*n, *v));
                if let Ok(micros) = time_decode(&device, &specialization, &case) {
                    if best.as_ref().is_none_or(|(fastest, _)| micros < *fastest) {
                        best = Some((micros, config));
                    }
                }
            }
            match best {
                Some((micros, config)) => eprintln!(
                    "timing {name} {:?}: {micros:.1} us, history {:.1} GB/s, {config:?}",
                    device.backend(),
                    history_bytes / micros / 1.0e3
                ),
                None => eprintln!(
                    "timing {name} {:?}: no admissible configuration",
                    device.backend()
                ),
            }
        }
    }
}

/// The decode configurations a timing sweeps: every slicing of the group
/// with each simdgroup/warp count and a few partition counts.
fn timing_configurations(device: &Device, form: Form) -> Vec<Vec<(&'static str, u64)>> {
    let g = form.g as u64;
    let slicings = |groups: u64| {
        [1u64, 2, 4, 8]
            .into_iter()
            .filter(move |s| g % s == 0 && groups % s == 0)
    };
    let mut configurations = Vec::new();
    match device.backend() {
        BackendName::Metal | BackendName::Vulkan => {
            for simds in [4u64, 8] {
                for slices in slicings(simds) {
                    for parts in [8u64, 16, 32] {
                        let mut configuration = vec![
                            ("SPAN", 32),
                            ("PARTS", parts),
                            ("SIMDS", simds),
                            ("SLICES", slices),
                            ("MATRIX", 0),
                        ];
                        if device.backend() == BackendName::Metal {
                            configuration.extend([("KEYS", 16), ("TOKENS", 1)]);
                        }
                        configurations.push(configuration);
                    }
                }
            }
            if device.backend() == BackendName::Metal {
                let sweep = [16u64, 32, 64, 128]
                    .into_iter()
                    .flat_map(|parts| [2u64, 4, 8].into_iter().map(move |simds| (parts, simds)))
                    .flat_map(|(parts, simds)| {
                        [8u64, 16]
                            .into_iter()
                            .map(move |keys| (128, parts, simds, keys))
                    })
                    .collect::<Vec<_>>();
                configurations.extend(metal_matrix_configurations(form, &sweep));
            } else {
                let sweep = [16u64, 32, 64, 128]
                    .into_iter()
                    .flat_map(|parts| [1u64, 2, 4].into_iter().map(move |simds| (parts, simds)))
                    .collect::<Vec<_>>();
                configurations.extend(vulkan_matrix_configurations(form, &sweep));
            }
        }
        BackendName::Cuda => {
            for warps in [4u64, 8] {
                for slices in slicings(warps) {
                    for parts in [12u64, 24, 48] {
                        configurations.push(vec![
                            ("PARTS", parts),
                            ("WARPS", warps),
                            ("SLICES", slices),
                            ("MATRIX", 0),
                            ("STAGES", 2),
                            ("COLUMNS", 1),
                        ]);
                    }
                }
            }
            let sweep = [12u64, 24, 48, 96]
                .into_iter()
                .flat_map(|parts| [1u64, 2, 4].into_iter().map(move |warps| (parts, warps)))
                .flat_map(|(parts, warps)| {
                    [2u64, 3, 4]
                        .into_iter()
                        .map(move |stages| (parts, warps, stages))
                })
                .flat_map(|(parts, warps, stages)| {
                    [1u64, 2, 4]
                        .into_iter()
                        .map(move |columns| (parts, warps, stages, columns))
                })
                .collect::<Vec<_>>();
            configurations.extend(cuda_matrix_configurations(form, &sweep));
        }
        _ => {}
    }
    configurations
}
