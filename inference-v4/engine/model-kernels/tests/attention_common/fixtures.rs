// Shared fixtures of the gated-attention entry tests (`attention.rs`,
// `cuda_attention.rs`, `attention_k8v4.rs`, `attention_segments.rs`), included
// textually: cases of rows with their controls, projections, weights and dense
// history, the host model of the portable body, the devices, device tensors
// and the row builders.

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

#[derive(Clone, Copy)]
struct Geometry {
    kv: usize,
    g: usize,
    p: usize,
    s: usize,
}

impl Geometry {
    fn w(&self) -> usize {
        2 * self.p + self.s
    }
}

/// One invocation: rows, their controls, projections, weights and history.
#[derive(Clone)]
struct Case {
    geometry: Geometry,
    rows: usize,
    history_rows: usize,
    spans: usize,
    query_gate: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    query_norm: Vec<f32>,
    key_norm: Vec<f32>,
    components: Vec<i32>,
    frequencies: Vec<f32>,
    coordinates: Vec<i32>,
    visible: Vec<i32>,
    fresh: Vec<i32>,
    destinations: Vec<i32>,
    history_key: Vec<f32>,
    history_value: Vec<f32>,
    epsilon: f32,
    scale: f32,
}

/// A row's controls: visible history spans (padded to the class with empty
/// spans), its fresh span over the batch rows, its destination and position.
struct Row {
    spans: Vec<(i32, i32)>,
    fresh: (i32, i32),
    destination: i32,
    position: i32,
}

impl Case {
    fn new(geometry: Geometry, history_rows: usize, spans: usize, rows: &[Row], seed: u64) -> Self {
        let mut noise = Noise(seed);
        let w = geometry.w();
        let heads = geometry.kv * geometry.g;
        let m = rows.len();
        let mut values = |count: usize, scale: f32| {
            (0..count)
                .map(|_| bf16(noise.next() * scale))
                .collect::<Vec<_>>()
        };
        let query_gate = values(m * heads * 2 * w, 2.0);
        let key = values(m * geometry.kv * w, 2.0);
        let value = values(m * geometry.kv * w, 1.0);
        let history_key = values(history_rows * geometry.kv * w, 3.0);
        let history_value = values(history_rows * geometry.kv * w, 1.0);
        let query_norm = (0..w).map(|i| 0.6 + (i % 7) as f32 * 0.1).collect();
        let key_norm = (0..w).map(|i| 1.3 - (i % 5) as f32 * 0.1).collect();
        // Qwen3.5 sections (11, 11, 10) interleaved over 32 pairs, scaled down
        // to P pairs: axis 1 at pairs 1 mod 3, axis 2 at pairs 2 mod 3.
        let components = (0..geometry.p)
            .map(|pair| match pair % 3 {
                1 if pair < 3 * geometry.p * 11 / 32 => 1,
                2 if pair < 3 * geometry.p * 10 / 32 => 2,
                _ => 0,
            })
            .collect();
        let coordinates = rows
            .iter()
            .flat_map(|row| {
                [
                    row.position,
                    row.position + 3,
                    row.position / 2,
                    0,
                ]
            })
            .collect();
        let visible = rows
            .iter()
            .flat_map(|row| {
                assert!(row.spans.len() <= spans);
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
            geometry,
            rows: m,
            history_rows,
            spans,
            query_gate,
            key,
            value,
            query_norm,
            key_norm,
            components,
            // Qwen3.5's rotary base 1e7, as the engine's static table.
            frequencies: (0..geometry.p)
                .map(|pair| 1.0e7f64.powf(-((2 * pair) as f64) / (2 * geometry.p) as f64) as f32)
                .collect(),
            coordinates,
            visible,
            fresh: rows.iter().flat_map(|row| [row.fresh.0, row.fresh.1]).collect(),
            destinations: rows.iter().map(|row| row.destination).collect(),
            history_key,
            history_value,
            epsilon: 1.0e-6,
            scale: 1.0 / (w as f32).sqrt(),
        }
    }

    fn prepare(&self, raw: &[f32], norm: &[f32], row: usize) -> Vec<f32> {
        let (p, w) = (self.geometry.p, self.geometry.w());
        let squares = raw.iter().fold(0.0f32, |sum, x| x.mul_add(*x, sum));
        let inverse = 1.0 / (squares / w as f32 + self.epsilon).sqrt();
        let normalized = raw
            .iter()
            .zip(norm)
            .map(|(x, n)| x * inverse * n)
            .collect::<Vec<_>>();
        (0..w)
            .map(|i| {
                if i >= 2 * p {
                    return normalized[i];
                }
                let pair = i % p;
                let coordinate = self.coordinates[row * 4 + self.components[pair] as usize];
                let angle = coordinate as f32 * self.frequencies[pair];
                if i < p {
                    normalized[i] * angle.cos() - normalized[i + p] * angle.sin()
                } else {
                    normalized[i] * angle.cos() + normalized[i - p] * angle.sin()
                }
            })
            .map(bf16)
            .collect()
    }

    /// The portable body's result and final histories, computed on the host.
    fn expected(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let Geometry { kv, g, .. } = self.geometry;
        let w = self.geometry.w();
        let keys = (0..self.rows)
            .map(|row| {
                (0..kv)
                    .map(|head| {
                        self.prepare(
                            &self.key[(row * kv + head) * w..][..w],
                            &self.key_norm,
                            row,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut gated = vec![0.0; self.rows * kv * g * w];
        for row in 0..self.rows {
            for head in 0..kv * g {
                let kv_head = head / g;
                let query = self.prepare(
                    &self.query_gate[(row * kv * g + head) * 2 * w..][..w],
                    &self.query_norm,
                    row,
                );
                let mut entries: Vec<(&[f32], &[f32])> = Vec::new();
                for span in 0..self.spans {
                    let lo = self.visible[(row * self.spans + span) * 2];
                    let hi = self.visible[(row * self.spans + span) * 2 + 1];
                    for token in lo.max(0)..hi {
                        let at = (token as usize * kv + kv_head) * w;
                        entries.push((&self.history_key[at..][..w], &self.history_value[at..][..w]));
                    }
                }
                for token in self.fresh[row * 2].max(0)..self.fresh[row * 2 + 1] {
                    let token = token as usize;
                    entries.push((
                        &keys[token][kv_head],
                        &self.value[(token * kv + kv_head) * w..][..w],
                    ));
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
                    let gate = f64::from(self.query_gate[(row * kv * g + head) * 2 * w + w + column]);
                    gated[(row * kv * g + head) * w + column] =
                        bf16((attended / (1.0 + (-gate).exp())) as f32);
                }
            }
        }
        let mut history_key = self.history_key.clone();
        let mut history_value = self.history_value.clone();
        for row in 0..self.rows {
            let destination = self.destinations[row];
            if destination < 0 {
                continue;
            }
            for head in 0..kv {
                let at = (destination as usize * kv + head) * w;
                history_key[at..at + w].copy_from_slice(&keys[row][head]);
                history_value[at..at + w]
                    .copy_from_slice(&self.value[(row * kv + head) * w..][..w]);
            }
        }
        (gated, history_key, history_value)
    }
}

/// Metal on macOS, elsewhere Vulkan when present (both run the Metal
/// decode/prefill contract with the same decode parameters), and the CPU
/// device.
fn devices() -> Vec<Device> {
    let catalog = DeviceCatalog::discover().unwrap();
    let gpu = if cfg!(target_os = "macos") {
        Some(catalog.open_backend(BackendName::Metal).unwrap())
    } else {
        catalog.open_backend(BackendName::Vulkan).ok()
    };
    gpu.into_iter()
        .chain(std::iter::once(catalog.open_backend(BackendName::Cpu).unwrap()))
        .collect()
}

fn is_cpu(device: &Device) -> bool {
    device.backend() == BackendName::Cpu
}

fn bf16_tensor(device: &Device, shape: &[usize], values: &[f32]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| bf16_bits(*value).to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::bf16(), &shape, &bytes).unwrap()
}

fn f32_tensor(device: &Device, shape: &[usize], values: &[f32]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::f32(), &shape, &bytes).unwrap()
}

fn i32_tensor(device: &Device, shape: &[usize], values: &[i32]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::i32(), &shape, &bytes).unwrap()
}

fn bf16_values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(2)
        .map(|bytes| f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16))
        .collect()
}

fn statics(geometry: Geometry) -> NativeSpecialization {
    NativeSpecialization::new()
        .with_static("KV", geometry.kv as u64)
        .with_static("G", geometry.g as u64)
        .with_static("P", geometry.p as u64)
        .with_static("S", geometry.s as u64)
}

const SMALL: Geometry = Geometry { kv: 2, g: 2, p: 4, s: 24 };
const QWEN: Geometry = Geometry { kv: 4, g: 4, p: 32, s: 192 };

/// Decode rows of two sequences: multi-span history with a gap, speculative
/// rows seeing earlier fresh rows, a row with no destination, and a padding
/// row that sees nothing.
fn decode_rows(base: i32) -> Vec<Row> {
    vec![
        Row { spans: vec![(0, base), (base + 7, base + 19)], fresh: (0, 1), destination: base + 40, position: base + 12 },
        Row { spans: vec![(0, base), (base + 7, base + 19)], fresh: (0, 2), destination: base + 41, position: base + 13 },
        Row { spans: vec![(base + 20, base + 33)], fresh: (2, 3), destination: -1, position: 13 },
        Row { spans: vec![], fresh: (0, 0), destination: -1, position: 0 },
    ]
}

/// Prefill rows of two sequences sharing tiles: history spans with a gap,
/// causal fresh spans, a mid-batch sequence boundary, rows without
/// destinations and trailing padding rows that see nothing.
fn prefill_rows(rows: usize, history: i32) -> Vec<Row> {
    let boundary = rows * 2 / 3;
    (0..rows)
        .map(|row| {
            let r = row as i32;
            if row + 2 >= rows {
                Row { spans: vec![], fresh: (0, 0), destination: -1, position: 0 }
            } else if row < boundary {
                Row {
                    spans: vec![(0, history / 2), (history / 2 + 9, history)],
                    fresh: (0, r + 1),
                    destination: if row % 5 == 3 { -1 } else { history + 64 + r },
                    position: history - 9 + r,
                }
            } else {
                let first = boundary as i32;
                Row {
                    spans: vec![(history + 3, history + 17)],
                    fresh: (first, r + 1),
                    destination: history + 64 + r,
                    position: 14 + r - first,
                }
            }
        })
        .collect()
}

/// One slot of up to 8 speculative decode rows after `context` history rows,
/// as the engine's tuner builds it: every row sees the history and the slot's
/// earlier rows, and appends after the history.
fn speculative_rows(rows: usize, context: i32) -> Vec<Row> {
    (0..rows)
        .map(|row| Row {
            spans: vec![(0, context)],
            fresh: (0, row as i32 + 1),
            destination: context + row as i32,
            position: context + row as i32,
        })
        .collect()
}

const TIMING: seismic::MeasureOptions = seismic::MeasureOptions {
    samples: 15,
    min_sample_seconds: 0.005,
};
