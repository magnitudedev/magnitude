// `attention_decode_k8v4` and `attention_prefill_k8v4` (affine K8/V4 history)
// in Qwen's form on every accelerator present and the CPU: against their
// portable body (the reference interpreter, small shapes) and against a host
// model of that body (the dense host model over the decoded history, Qwen3.5
// geometries and 512-column heads, long contexts). Timings are ignored tests.
#![allow(dead_code)]

use magnitude_kernels::{
    attention_append_k8v4, attention_decode, attention_decode_k8v4, attention_prefill,
    attention_prefill_k8v4,
};
use seismic::{
    BackendName, Device, DeviceCatalog, Element, NativeSpecialization, SlabRegion, SlabTensor,
    Tensor,
};
use seismic_lang::registry::{f16_bits, f16_to_f32};

include!("attention_common/fixtures.rs");

const KEY_BITS: u32 = 8;
const VALUE_BITS: u32 = 4;
/// Values per (scale, zero) pair: the codec's group (model-state
/// `AFFINE_GROUP`).
const GROUP: usize = 32;

/// Two affine groups per head vector, for the decode portable-body comparison.
const GROUPED: Geometry = Geometry {
    kv: 2,
    g: 2,
    p: 8,
    s: 48,
};

/// F16 coefficient elements of one head vector: a (scale, zero) pair per
/// group.
fn coefficient_elements(width: usize) -> usize {
    2 * width / GROUP
}

/// One vector's affine encoding: codes packed B bits each into u32 words
/// (code i at bits B * (i % (32 / B)) of word i / (32 / B)) and the F16 bits
/// of (scale, zero) per group of GROUP values, exactly as the portable body
/// defines it.
fn encode(x: &[f32], bits: u32) -> (Vec<u32>, Vec<u16>) {
    let levels = (1u32 << bits) - 1;
    let per = (32 / bits) as usize;
    let mut words = vec![0u32; x.len() / per];
    let mut coefficients = Vec::with_capacity(coefficient_elements(x.len()));
    for (group, values) in x.chunks_exact(GROUP).enumerate() {
        let low = values.iter().copied().fold(f32::INFINITY, f32::min);
        let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let zero = f16_bits(low);
        let scale = f16_bits((high - low) / levels as f32);
        let inverse = if f16_to_f32(scale) > 0.0 {
            1.0 / f16_to_f32(scale)
        } else {
            0.0
        };
        for (offset, value) in values.iter().enumerate() {
            let i = group * GROUP + offset;
            let code = ((value - f16_to_f32(zero)).mul_add(inverse, 0.5) as u32).min(levels);
            words[i / per] |= code << ((i % per) as u32 * bits);
        }
        coefficients.extend([scale, zero]);
    }
    (words, coefficients)
}

fn decode(words: &[u32], coefficients: &[u16], bits: u32, width: usize) -> Vec<f32> {
    let per = (32 / bits) as usize;
    let mask = (1u32 << bits) - 1;
    (0..width)
        .map(|i| {
            let code = (words[i / per] >> ((i % per) as u32 * bits)) & mask;
            let pair = &coefficients[i / GROUP * 2..][..2];
            (code as f32).mul_add(f16_to_f32(pair[0]), f16_to_f32(pair[1]))
        })
        .collect()
}

/// Affine planes of a history: per (row, kv head) vector, its code words and
/// coefficient bits.
#[derive(Clone, PartialEq, Debug)]
struct Planes {
    key_codes: Vec<u32>,
    key_coefficients: Vec<u16>,
    value_codes: Vec<u32>,
    value_coefficients: Vec<u16>,
}

impl Planes {
    fn encode(geometry: Geometry, keys: &[f32], values: &[f32]) -> Self {
        let w = geometry.w();
        let mut planes = Planes {
            key_codes: Vec::new(),
            key_coefficients: Vec::new(),
            value_codes: Vec::new(),
            value_coefficients: Vec::new(),
        };
        for (key, value) in keys.chunks_exact(w).zip(values.chunks_exact(w)) {
            let (codes, coefficients) = encode(key, KEY_BITS);
            planes.key_codes.extend(codes);
            planes.key_coefficients.extend(coefficients);
            let (codes, coefficients) = encode(value, VALUE_BITS);
            planes.value_codes.extend(codes);
            planes.value_coefficients.extend(coefficients);
        }
        planes
    }

    /// The decoded key and value of vector `vector`.
    fn decoded(&self, geometry: Geometry, vector: usize) -> (Vec<f32>, Vec<f32>) {
        let w = geometry.w();
        let (kw, vw) = (w * KEY_BITS as usize / 32, w * VALUE_BITS as usize / 32);
        let c = coefficient_elements(w);
        (
            decode(
                &self.key_codes[vector * kw..][..kw],
                &self.key_coefficients[vector * c..][..c],
                KEY_BITS,
                w,
            ),
            decode(
                &self.value_codes[vector * vw..][..vw],
                &self.value_coefficients[vector * c..][..c],
                VALUE_BITS,
                w,
            ),
        )
    }
}

/// A case whose history is the affine encoding of its dense history.
struct Encoded {
    case: Case,
    planes: Planes,
}

impl Encoded {
    fn new(case: Case) -> Self {
        let planes = Planes::encode(case.geometry, &case.history_key, &case.history_value);
        Self { case, planes }
    }

    /// The host model: the dense host model over the decoded history, and the
    /// planes after the rows' appends (encoded prepared keys and values).
    fn expected(&self) -> (Vec<f32>, Planes) {
        let geometry = self.case.geometry;
        let w = geometry.w();
        let mut decoded = self.case.clone();
        for vector in 0..self.case.history_rows * geometry.kv {
            let (key, value) = self.planes.decoded(geometry, vector);
            decoded.history_key[vector * w..][..w].copy_from_slice(&key);
            decoded.history_value[vector * w..][..w].copy_from_slice(&value);
        }
        let (gated, keys, values) = decoded.expected();
        let mut planes = self.planes.clone();
        let (kw, vw) = (w * KEY_BITS as usize / 32, w * VALUE_BITS as usize / 32);
        let c = coefficient_elements(w);
        for &destination in &self.case.destinations {
            if destination < 0 {
                continue;
            }
            for head in 0..geometry.kv {
                let vector = destination as usize * geometry.kv + head;
                let (codes, coefficients) = encode(&keys[vector * w..][..w], KEY_BITS);
                planes.key_codes[vector * kw..][..kw].copy_from_slice(&codes);
                planes.key_coefficients[vector * c..][..c].copy_from_slice(&coefficients);
                let (codes, coefficients) = encode(&values[vector * w..][..w], VALUE_BITS);
                planes.value_codes[vector * vw..][..vw].copy_from_slice(&codes);
                planes.value_coefficients[vector * c..][..c].copy_from_slice(&coefficients);
            }
        }
        (gated, planes)
    }
}

fn u32_tensor(device: &Device, shape: &[usize], values: &[u32]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::u32(), &shape, &bytes).unwrap()
}

fn f16_tensor(device: &Device, shape: &[usize], bits: &[u16]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    let bytes = bits
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::f16(), &shape, &bytes).unwrap()
}

fn words(tensor: &Tensor) -> Vec<u32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

fn halves(tensor: &Tensor) -> Vec<u16> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect()
}

/// Device tensors of one case, in the family's Qwen form: queries with their
/// interleaved gates, no separate gate, one fresh layer, q/k norms, no value
/// norm, unit rotary amplitudes.
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
    history_tiles: Tensor,
    _history_slabs: SlabTensor,
    key_codes: Tensor,
    key_coefficients: Tensor,
    value_codes: Tensor,
    value_coefficients: Tensor,
    slab_rows: u32,
}

impl Bound {
    fn new(device: &Device, encoded: &Encoded) -> Self {
        Self::new_with_slab_rows(device, encoded, encoded.case.history_rows, false)
    }

    fn new_with_slab_rows(
        device: &Device,
        encoded: &Encoded,
        slab_rows: usize,
        reuse: bool,
    ) -> Self {
        let case = &encoded.case;
        let Geometry { kv, g, p, .. } = case.geometry;
        let (m, w, t) = (case.rows, case.geometry.w(), case.history_rows);
        assert_eq!(t % slab_rows, 0);
        let mut slabs = SlabTensor::new(
            device,
            slab_rows as u64,
            t as u64,
            vec![
                SlabRegion {
                    element: Element::u32(),
                    row_shape: vec![kv as u64, (w / 4) as u64],
                },
                SlabRegion {
                    element: Element::f16(),
                    row_shape: vec![kv as u64, coefficient_elements(w) as u64],
                },
                SlabRegion {
                    element: Element::u32(),
                    row_shape: vec![kv as u64, (w / 8) as u64],
                },
                SlabRegion {
                    element: Element::f16(),
                    row_shape: vec![kv as u64, coefficient_elements(w) as u64],
                },
            ],
        )
        .unwrap();
        for index in 0..t / slab_rows {
            assert_eq!(slabs.add_slab().unwrap(), index);
        }
        if reuse {
            assert!(t / slab_rows >= 3);
            slabs.free_slab(1).unwrap();
            assert_eq!(slabs.add_slab().unwrap(), 1);
        }
        for (region, bytes) in [
            encoded
                .planes
                .key_codes
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            encoded
                .planes
                .key_coefficients
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            encoded
                .planes
                .value_codes
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
            encoded
                .planes
                .value_coefficients
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>(),
        ]
        .into_iter()
        .enumerate()
        {
            let row_bytes = bytes.len() / t;
            for start in (0..t).step_by(slab_rows) {
                slabs
                    .region_rows(region, start as u64, slab_rows as u64)
                    .unwrap()
                    .write_from_host(&bytes[start * row_bytes..(start + slab_rows) * row_bytes])
                    .unwrap();
            }
        }
        let key_codes = slabs.logical_region(0).unwrap();
        let key_coefficients = slabs.logical_region(1).unwrap();
        let value_codes = slabs.logical_region(2).unwrap();
        let value_coefficients = slabs.logical_region(3).unwrap();
        Self {
            query: bf16_tensor(device, &[m, kv * g, 2 * w], &case.query_gate),
            gate: bf16_tensor(device, &[m, kv * g, 0], &[]),
            key: bf16_tensor(device, &[1, m, kv * w], &case.key),
            value: bf16_tensor(device, &[1, m, kv * w], &case.value),
            query_norm: f32_tensor(device, &[1, w], &case.query_norm),
            key_norm: f32_tensor(device, &[1, w], &case.key_norm),
            value_norm: f32_tensor(device, &[0, w], &[]),
            components: i32_tensor(device, &[p], &case.components),
            frequencies: f32_tensor(device, &[p], &case.frequencies),
            amplitudes: f32_tensor(device, &[p], &vec![1.0; p]),
            coordinates: i32_tensor(device, &[m, 4], &case.coordinates),
            visible: i32_tensor(device, &[m, case.spans, 2], &case.visible),
            fresh: i32_tensor(device, &[m, 2], &case.fresh),
            destinations: i32_tensor(device, &[m], &case.destinations),
            history_tiles: {
                let tiles = history_tiles(&case.visible);
                i32_tensor(device, &[1, tiles.len()], &tiles)
            },
            _history_slabs: slabs,
            key_codes,
            key_coefficients,
            value_codes,
            value_coefficients,
            slab_rows: slab_rows as u32,
        }
    }

    fn planes(&self) -> Planes {
        Planes {
            key_codes: words(&self.key_codes),
            key_coefficients: halves(&self.key_coefficients),
            value_codes: words(&self.value_codes),
            value_coefficients: halves(&self.value_coefficients),
        }
    }
}

macro_rules! args {
    (attention_append_k8v4, $bound:expr, $case:expr) => {
        attention_append_k8v4::Args {
            key: &$bound.key,
            value: &$bound.value,
            key_norm: &$bound.key_norm,
            value_norm: &$bound.value_norm,
            rotary_components: &$bound.components,
            rotary_frequencies: &$bound.frequencies,
            rotary_amplitudes: &$bound.amplitudes,
            coordinates: &$bound.coordinates,
            destinations: &$bound.destinations,
            history_key_codes: &mut $bound.key_codes,
            history_key_coefficients: &mut $bound.key_coefficients,
            history_value_codes: &mut $bound.value_codes,
            history_value_coefficients: &mut $bound.value_coefficients,
            epsilon: $case.epsilon,
            slab_rows: $bound.slab_rows,
        }
    };
    ($module:ident, $bound:expr, $case:expr) => {
        $module::Args {
            query: &$bound.query,
            gate: &$bound.gate,
            key: &$bound.key,
            value: &$bound.value,
            query_norm: &$bound.query_norm,
            key_norm: &$bound.key_norm,
            value_norm: &$bound.value_norm,
            rotary_components: &$bound.components,
            rotary_frequencies: &$bound.frequencies,
            rotary_amplitudes: &$bound.amplitudes,
            coordinates: &$bound.coordinates,
            visible: &$bound.visible,
            fresh: &$bound.fresh,
            destinations: &$bound.destinations,
            history_key_codes: &mut $bound.key_codes,
            history_key_coefficients: &mut $bound.key_coefficients,
            history_value_codes: &mut $bound.value_codes,
            history_value_coefficients: &mut $bound.value_coefficients,
            epsilon: $case.epsilon,
            scale: $case.scale,
            gate_function: 0,
            slab_rows: $bound.slab_rows,
        }
    };
}

/// The affine prefill entry's arguments: `args!`'s, and the history row tiles
/// the case's rows see.
macro_rules! prefill_args {
    ($bound:expr, $case:expr) => {
        attention_prefill_k8v4::Args {
            query: &$bound.query,
            gate: &$bound.gate,
            key: &$bound.key,
            value: &$bound.value,
            query_norm: &$bound.query_norm,
            key_norm: &$bound.key_norm,
            value_norm: &$bound.value_norm,
            rotary_components: &$bound.components,
            rotary_frequencies: &$bound.frequencies,
            rotary_amplitudes: &$bound.amplitudes,
            coordinates: &$bound.coordinates,
            visible: &$bound.visible,
            fresh: &$bound.fresh,
            destinations: &$bound.destinations,
            history_tiles: &$bound.history_tiles,
            history_key_codes: &mut $bound.key_codes,
            history_key_coefficients: &mut $bound.key_coefficients,
            history_value_codes: &mut $bound.value_codes,
            history_value_coefficients: &mut $bound.value_coefficients,
            epsilon: $case.epsilon,
            scale: $case.scale,
            gate_function: 0,
            slab_rows: $bound.slab_rows,
        }
    };
}

/// The devices with k8v4 natives: every accelerator present (Metal, CUDA,
/// Vulkan) and the CPU. A selected backend must open so qualification cannot
/// pass by silently running on another device.
fn k8v4_devices() -> Vec<Device> {
    let catalog = DeviceCatalog::discover().unwrap();
    if let Ok(selected) = std::env::var("SEISMIC_TEST_BACKEND") {
        let backend = match selected.as_str() {
            "metal" => BackendName::Metal,
            "cuda" => BackendName::Cuda,
            "vulkan" => BackendName::Vulkan,
            "cpu" => BackendName::Cpu,
            other => panic!("unsupported SEISMIC_TEST_BACKEND={other}"),
        };
        return vec![catalog
            .open_backend(backend)
            .expect("selected backend opens")];
    }
    [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan]
        .into_iter()
        .filter_map(|backend| catalog.open_backend(backend).ok())
        .chain(std::iter::once(
            catalog.open_backend(BackendName::Cpu).unwrap(),
        ))
        .collect()
}

/// Whether Vulkan's vector decode form (SIMDS, SLICES, KEYWISE) is within the
/// declaration's `where`: the per-key walk needs four subgroups, and the
/// walk's shared bytes (the keywise walk's staging tiles among them) fit.
fn vulkan_vector_fits(geometry: Geometry, simds: u64, slices: u64, keywise: u64) -> bool {
    let (g, w) = (geometry.g as u64, geometry.w() as u64);
    let heads = g / slices;
    let bytes = if keywise == 0 {
        simds * heads * 8 + simds.max(g.div_ceil(2)) * w * 4
    } else {
        let pitch = (w.min(128) / 16).max(w / 32 + w.div_ceil(128)) + 1;
        simds * heads * 8
            + (simds * w * 4)
                .max(32 + g * w * 4 + simds * heads * 128 + g * w / 8 + simds * 512 * pitch)
    };
    (keywise == 1 || simds >= 4) && bytes <= 32768
}

/// Vulkan's grouped-query matrix decode configurations (PARTS, SIMDS) whose
/// SIMDS * 16 matrix rows hold the query group, within the declaration's
/// shared bytes (the K/V tile, the subgroup scratch, the query tile).
fn vulkan_matrix_configs(
    geometry: Geometry,
    configs: &[(u64, u64)],
) -> Vec<Vec<(&'static str, u64)>> {
    let w = geometry.w() as u64;
    configs
        .iter()
        .filter(|(_, simds)| {
            geometry.g as u64 <= simds * 16
                && 64 * (w.min(256) + 8) + simds * 2048 + simds * 16 * (w + 8) * 2 <= 32768
        })
        .map(|&(parts, simds)| {
            vec![
                ("SPAN", 32),
                ("PARTS", parts),
                ("SIMDS", simds),
                ("SLICES", 1),
                ("MATRIX", 1),
                ("KEYWISE", 0),
            ]
        })
        .collect()
}

/// CUDA's grouped-query matrix decode configurations (PARTS, WARPS) whose
/// WARPS * 16 matrix rows hold the query group, within the declaration's
/// shared bytes.
fn cuda_matrix_configs(
    geometry: Geometry,
    configs: &[(u64, u64, u64, u64)],
) -> Vec<Vec<(&'static str, u64)>> {
    configs
        .iter()
        .filter(|(_, warps, stages, columns)| {
            // The declaration's shared bound: the query tile, STAGES K-piece +
            // V-window stages (16-key tiles above W = 256), the exchange.
            let w = geometry.w() as u64;
            let keys = if w > 256 { 16 } else { 32 };
            geometry.g as u64 <= warps * 16
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

/// Metal's grouped-query matrix decode form at `geometry`: its column slices
/// (a simdgroup's outputs cover at most 128 columns of every 8-row block),
/// each team being one simdgroup per slice, and whether `simds` simdgroups of `keys`-key tiles fit the declaration's
/// threadgroup bytes.
fn metal_matrix_slices(geometry: Geometry, tokens: u64) -> u64 {
    ((tokens * geometry.g as u64).div_ceil(8) * geometry.w() as u64 / 128).max(1)
}

fn metal_matrix_admits(geometry: Geometry, tokens: u64, simds: u64, keys: u64) -> bool {
    // The packed form: one simdgroup per token, K and V tiles staged apart.
    if geometry.g == 8 && matches!(geometry.w(), 128 | 256) && tokens == 4 && simds == 4 {
        let w = geometry.w() as u64;
        return (4 * 8 * w * 2).max(keys * w * 4).max(8 * w * 4) + 8 * 8 <= 32768;
    }
    let (w, rows) = (
        geometry.w() as u64,
        (tokens * geometry.g as u64).div_ceil(8) * 8,
    );
    let cols = metal_matrix_slices(geometry, tokens);
    let wc = w / cols;
    let exchange = if cols > 1 {
        2 * simds * rows * keys * 4
    } else {
        0
    };
    let tile = simds * keys * wc * 2;
    let middle = if simds == cols {
        tile.max(exchange)
    } else {
        (simds * keys * (wc + 8) * 2 + exchange).max(rows * w * 4) + simds / cols * rows * 8
    };
    w % cols == 0 && wc % 32 == 0 && simds % cols == 0 && rows * w * 2 + middle <= 32768
}

/// Metal's grouped-query matrix decode configurations admissible at
/// `geometry`, as (span, parts, simds, keys), each at one row and at four
/// rows per threadgroup; with several column slices at least one team's
/// simdgroups.
fn metal_matrix_configs(
    geometry: Geometry,
    configs: &[(u64, u64, u64, u64)],
) -> Vec<Vec<(&'static str, u64)>> {
    let mut admitted: Vec<Vec<(&'static str, u64)>> = Vec::new();
    for tokens in [1u64, 2, 4] {
        let cols = metal_matrix_slices(geometry, tokens);
        for &(span, parts, simds, keys) in configs {
            let simds = simds.max(cols);
            let config = vec![
                ("SPAN", span),
                ("PARTS", parts),
                ("SIMDS", simds),
                ("SLICES", 1),
                ("MATRIX", 1),
                ("KEYS", keys),
                ("TOKENS", tokens),
            ];
            if metal_matrix_admits(geometry, tokens, simds, keys) && !admitted.contains(&config) {
                admitted.push(config);
            }
        }
    }
    admitted
}

/// Declared decode configurations of `backend` at `geometry`.
fn decode_configs(backend: BackendName, geometry: Geometry) -> Vec<Vec<(&'static str, u64)>> {
    let group = geometry.g;
    match backend {
        // The last configuration slices the query group as finely as the
        // group and the warps admit.
        BackendName::Cuda => {
            let slices = [8u64, 4, 2, 1]
                .into_iter()
                .find(|s| group as u64 % s == 0)
                .unwrap();
            let mut configs = [(12, 4, 1), (24, 8, 1), (48, 4, 1), (48, 8, slices)]
                .into_iter()
                .map(|(parts, warps, slices)| {
                    vec![
                        ("PARTS", parts),
                        ("WARPS", warps),
                        ("SLICES", slices),
                        ("MATRIX", 0),
                        ("STAGES", 2),
                        ("COLUMNS", 1),
                    ]
                })
                .collect::<Vec<_>>();
            configs.extend(cuda_matrix_configs(
                geometry,
                &[(12, 1, 2, 4), (48, 2, 3, 2), (24, 4, 4, 1)],
            ));
            configs
        }
        BackendName::Cpu => [8, 4, 16, 1]
            .into_iter()
            .map(|parts| vec![("PARTS", parts)])
            .collect(),
        // The last configuration slices the query group as finely as the
        // group and the simdgroups admit.
        BackendName::Metal => {
            let slices = [8u64, 4, 2, 1]
                .into_iter()
                .find(|s| group as u64 % s == 0)
                .unwrap();
            let mut configs = [
                (32, 16, 4, 1),
                (64, 32, 8, 1),
                (256, 8, 8, 1),
                (32, 16, 8, slices),
            ]
            .into_iter()
            .map(|(span, parts, simds, slices)| {
                vec![
                    ("SPAN", span),
                    ("PARTS", parts),
                    ("SIMDS", simds),
                    ("SLICES", slices),
                    ("MATRIX", 0),
                    ("KEYS", 16),
                    ("TOKENS", 1),
                ]
            })
            .collect::<Vec<_>>();
            configs.extend(metal_matrix_configs(
                geometry,
                &[
                    (32, 16, 4, 16),
                    (64, 32, 8, 8),
                    (256, 8, 4, 32),
                    (32, 16, 2, 16),
                ],
            ));
            configs
        }
        // A slice count must divide the geometry's query group. The portable
        // comparison also uses a two-query group. Each vector configuration
        // runs both walks (KEYWISE) where the declaration admits them; the
        // keywise walk's staging tiles need one or two subgroups at wide
        // heads.
        BackendName::Vulkan => [
            (32, 16, 4, 1),
            (64, 32, 8, 2),
            (256, 8, 8, 4),
            (32, 16, 8, 2),
            (32, 16, 4, 4),
            (32, 16, 8, 8),
            (32, 16, 2, 1),
            (64, 32, 2, 2),
            (64, 16, 1, 1),
        ]
        .into_iter()
        .filter(|(_, _, _, slices)| group.is_multiple_of(*slices as usize))
        .flat_map(|(span, parts, simds, slices)| {
            [0, 1]
                .into_iter()
                .filter(move |&keywise| vulkan_vector_fits(geometry, simds, slices, keywise))
                .map(move |keywise| {
                    vec![
                        ("SPAN", span),
                        ("PARTS", parts),
                        ("SIMDS", simds),
                        ("SLICES", slices),
                        ("MATRIX", 0),
                        ("KEYWISE", keywise),
                    ]
                })
        })
        .chain(vulkan_matrix_configs(
            geometry,
            &[(16, 1), (64, 2), (32, 4)],
        ))
        .collect(),
    }
}

/// Declared prefill configurations of `backend` admissible at `geometry`
/// (CUDA's 8-warp block fits shared memory up to 192-column heads unless the
/// queries stay in registers).
fn prefill_configs(backend: BackendName, geometry: Geometry) -> Vec<Vec<(&'static str, u64)>> {
    match backend {
        // (WARPS, SPLIT_GROUPS, STAGES, COLUMNS, QREG, PRODUCERS), or
        // `K8V4_PREFILL_CONFIGS` as `;`-separated comma sextuples.
        // PRODUCERS = 0 decodes in the MMA warps (heads of 128 to 256
        // columns, with code stages beside the operand stages).
        BackendName::Cuda => match std::env::var("K8V4_PREFILL_CONFIGS") {
            Ok(configs) => configs
                .split(';')
                .map(|config| {
                    let v = config
                        .split(',')
                        .map(|x| x.trim().parse::<usize>().unwrap())
                        .collect::<Vec<_>>();
                    (v[0], v[1] as u64, v[2], v[3], v[4], v[5])
                })
                .collect::<Vec<_>>(),
            Err(_) => vec![
                (4, 1, 2, 1, 0, 4),
                (2, 1, 2, 1, 0, 2),
                (4, 256, 2, 1, 0, 4),
                (2, 512, 2, 1, 0, 1),
                (8, 1, 2, 1, 0, 2),
                (8, 256, 2, 1, 0, 8),
                (4, 256, 2, 2, 0, 2),
                (2, 1, 2, 2, 0, 4),
                (2, 1, 3, 1, 0, 2),
                (4, 256, 2, 1, 1, 1),
                (8, 256, 2, 1, 1, 2),
                (8, 1, 2, 1, 1, 0),
                (8, 256, 2, 1, 1, 0),
                (4, 1, 3, 1, 0, 0),
                (2, 256, 2, 2, 0, 0),
                (8, 256, 2, 1, 0, 0),
            ],
        }
        .into_iter()
        .filter(|&(warps, _, stages, columns, qreg, producers)| {
            // The declaration's shared bound: the query tile and STAGES
            // K-piece + V-window stages (16-key tiles above W = 256), plus
            // code stages when the MMA warps decode, which share one region
            // when the queries stay in registers.
            let w = geometry.w();
            let keys = if w > 256 { 16 } else { 32 };
            let codes = if producers == 0 { stages * 56 * w } else { 0 };
            let (queries, stage_bytes) = (
                warps * 16 * w * 2,
                stages * keys * (w.min(256) + w.min(256 * columns)) * 2 + codes,
            );
            let region = if qreg == 1 {
                queries.max(stage_bytes)
            } else {
                queries + stage_bytes
            };
            w.min(256 * columns) % (16 * columns) == 0
                && (qreg == 0 || w <= 256)
                && (producers > 0 || (128..=256).contains(&w))
                && region <= 98304
        })
        .map(|(warps, split, stages, columns, qreg, producers)| {
            vec![
                ("WARPS", warps as u64),
                ("SPLIT_GROUPS", split),
                ("STAGES", stages as u64),
                ("COLUMNS", columns as u64),
                ("QREG", qreg as u64),
                ("PRODUCERS", producers as u64),
            ]
        })
        .collect(),
        // One CPU form, without parameters.
        BackendName::Cpu => vec![Vec::new()],
        // ROWS = 64 is admissible at every tested geometry (ROWS / G <= 32,
        // and the W = 256 tile fits the shared bound).
        BackendName::Vulkan => [1, 256, 512]
            .into_iter()
            .map(|split| vec![("ROWS", 64), ("SPLIT_GROUPS", split)])
            .collect(),
        // Each query tile with one head group and with groups of 4 and 2
        // heads, staged and (for whole 16-row simdgroups) in the direct
        // form. QT = 32 where its 32 HEADS rows fit the device's threadgroup
        // memory with the tensor-operation form (32 HEADS <= 128).
        _ => {
            let single = single_head_group(geometry.g);
            let mut heads = vec![single];
            heads.extend([4, 2].into_iter().filter(|&heads| heads < single));
            [(16, 1), (8, 1), (16, 256), (8, 256), (32, 1), (32, 256)]
                .into_iter()
                .flat_map(|(qt, split)| heads.iter().map(move |&heads| (qt, heads, split)))
                .filter(|&(qt, heads, _)| qt < 32 || qt * heads.min(geometry.g as u64) <= 128)
                .flat_map(|(qt, heads, split)| {
                    (0..1 + u64::from(qt >= 16)).map(move |direct| {
                        vec![
                            ("QT", qt),
                            ("HEADS", heads),
                            ("SPLIT_GROUPS", split),
                            ("DIRECT", direct),
                        ]
                    })
                })
                .collect()
        }
    }
}

/// The specialization of `params` on `device`: GPU forms take the geometry's
/// statics; CPU forms have no statics.
fn specialization_on(
    device: &Device,
    geometry: Geometry,
    params: &[(&'static str, u64)],
) -> NativeSpecialization {
    // The CPU forms fix only the head width, which their codec requires to be
    // whole 32-column groups.
    let base = if is_cpu(device) {
        NativeSpecialization::new()
            .with_static("P", geometry.p as u64)
            .with_static("S", geometry.s as u64)
    } else {
        statics(geometry)
    };
    let base = qwen_form(base, geometry);
    let spec = params
        .iter()
        .fold(base, |spec, (name, value)| spec.with_param(*name, *value));
    // Metal's prefill splits a kv head's query heads into groups of HEADS and
    // has the DIRECT form; unless the configuration names them, a single
    // group and the staged form.
    let names = |name: &str| params.iter().any(|(param, _)| *param == name);
    if device.backend() != BackendName::Metal || !names("QT") {
        return spec;
    }
    [("HEADS", single_head_group(geometry.g)), ("DIRECT", 0)]
        .into_iter()
        .filter(|(name, _)| !names(name))
        .fold(spec, |spec, (name, value)| spec.with_param(name, value))
}

/// The smallest declared HEADS holding all of a kv head's `g` query heads.
fn single_head_group(g: usize) -> u64 {
    (g.next_power_of_two() as u64).min(16)
}

fn decode_kernel(
    device: &Device,
    geometry: Geometry,
    params: &[(&'static str, u64)],
) -> seismic::NativeKernel<attention_decode_k8v4::Entry> {
    attention_decode_k8v4::native_for_device_with(
        device,
        attention_decode_k8v4::Elements { A: Element::bf16() },
        &specialization_on(device, geometry, params),
    )
    .unwrap()
}

/// The affine prefill's specialization of `params` on `device`. Metal's has
/// the COISSUE form, which a configuration takes only by naming it.
fn prefill_specialization(
    device: &Device,
    geometry: Geometry,
    params: &[(&'static str, u64)],
) -> NativeSpecialization {
    let spec = specialization_on(device, geometry, params);
    let names = |name: &str| params.iter().any(|(param, _)| *param == name);
    if device.backend() != BackendName::Metal {
        return spec;
    }
    // The cases list the history row tiles their rows see.
    let spec = spec.with_static("L", 1);
    if !names("QT") {
        return spec;
    }
    if names("COISSUE") {
        spec
    } else {
        spec.with_param("COISSUE", 0)
    }
}

fn prefill_kernel(
    device: &Device,
    geometry: Geometry,
    params: &[(&'static str, u64)],
) -> seismic::NativeKernel<attention_prefill_k8v4::Entry> {
    attention_prefill_k8v4::native_for_device_with(
        device,
        attention_prefill_k8v4::Elements { A: Element::bf16() },
        &prefill_specialization(device, geometry, params),
    )
    .unwrap()
}

/// Metal's COISSUE configurations of a 128-, 256- or 512-column geometry: the
/// threadgroups of two to four simdgroup pairs that fit the device's
/// threadgroup memory (a 512-column head's two pairs per 8 rows, with their
/// partial scores, fit 8-row query tiles alone), unsplit and split key
/// partitions.
fn coissue_prefill_configs(geometry: Geometry) -> Vec<Vec<(&'static str, u64)>> {
    let w = geometry.w() as u64;
    if w != 128 && w != 256 && w != 512 {
        return Vec::new();
    }
    let keys = if w > 128 { 32 } else { 96 };
    let windows = (w / 256).max(1);
    let pair = 48 + 2 * (keys * 16 + 32) + 32 * w.min(256) + 2048 * (windows - 1);
    [(16, 1), (16, 2), (32, 1), (8, 1)]
        .into_iter()
        .filter(|&(qt, _)| (qt == 8) == (w == 512))
        .filter(|&(qt, heads)| {
            32 + qt * heads.min(geometry.g as u64) / 8 * windows * pair + 64 * 16 <= 32768
        })
        .flat_map(|(qt, heads)| [1, 256].into_iter().map(move |split| (qt, heads, split)))
        .map(|(qt, heads, split)| {
            vec![
                ("QT", qt),
                ("HEADS", heads),
                ("SPLIT_GROUPS", split),
                ("DIRECT", 1),
                ("COISSUE", 1),
            ]
        })
        .collect()
}

/// Metal's COISSUE form (Q K^T on the matrix pipe, P V as scalar F16 products
/// in paired simdgroups) against the host model within the entry's tolerance
/// and against the entry's default configuration on the same inputs: the form
/// accumulates each step's products in F16 where the default accumulates in
/// F32, and every gated output stays within the production tolerance of a
/// BF16 result (0.01 + 1%), the bound tuning holds a configuration to; both
/// leave the same history planes. With tensor operations COISSUE runs the
/// direct form.
#[test]
fn metal_coissue_prefill_agrees_with_the_default() {
    metal_form_agrees_with_the_default(
        "COISSUE",
        &[MINICPM5, QWEN],
        &[
            (40, 300, 512, None),
            (128, 1000, 1280, None),
            (512, 4096, 4736, None),
            (64, 4096, 4352, Some(1088)),
            (512, 16384, 17024, None),
        ],
        coissue_prefill_configs,
    );
}

/// The COISSUE form of a 512-column head: a pair per 256-column window, the
/// two score simdgroups of 8 rows adding their partial scores (a score is the
/// sum of two F32 sums of 256 products where the default holds one sum of
/// 512), each product simdgroup forming its window's output columns.
#[test]
fn metal_coissue_prefill_of_two_windows_agrees_with_the_default() {
    metal_form_agrees_with_the_default(
        "COISSUE",
        &[GEMMA_E4B_FULL, GEMMA26_FULL],
        &[
            (40, 300, 512, None),
            (128, 1000, 1280, None),
            (512, 4096, 4736, None),
            (64, 4096, 4352, Some(1088)),
        ],
        coissue_prefill_configs,
    );
}

/// The configurations `configs` gives of Metal's form `form`, for each
/// geometry and each case (rows, history rows, store rows, rows per slab):
/// against the host model within the entry's tolerance, and against the
/// entry's default configuration on the same inputs within the production
/// tolerance of a BF16 result, leaving the same history planes.
fn metal_form_agrees_with_the_default(
    form: &str,
    geometries: &[Geometry],
    cases: &[(usize, i32, usize, Option<usize>)],
    configs: fn(Geometry) -> Vec<Vec<(&'static str, u64)>>,
) {
    let Some(device) = k8v4_devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    for (geometry, rows, history, total, slab_rows) in geometries.iter().flat_map(|&geometry| {
        cases
            .iter()
            .map(move |&(rows, history, total, slab_rows)| (geometry, rows, history, total, slab_rows))
    }) {
        let encoded = Encoded::new(Case::new(
            geometry,
            total,
            2,
            &prefill_rows(rows, history),
            7,
        ));
        let expected = encoded.expected();
        let bind = |device: &Device| match slab_rows {
            Some(slab_rows) => Bound::new_with_slab_rows(device, &encoded, slab_rows, false),
            None => Bound::new(device, &encoded),
        };
        // The entry's default configuration, the reference tuning validates
        // a configuration against.
        let default = [
            ("QT", 16),
            ("HEADS", single_head_group(geometry.g)),
            ("SPLIT_GROUPS", 256),
            ("DIRECT", 0),
        ];
        let mut direct_bound = bind(&device);
        let reference = bf16_values(
            &prefill_kernel(&device, geometry, &default)
                .call(prefill_args!(direct_bound, encoded.case))
                .unwrap()
                .value,
        );
        for config in configs(geometry) {
            let mut bound = bind(&device);
            let gated = prefill_kernel(&device, geometry, &config)
                .call(prefill_args!(bound, encoded.case))
                .unwrap()
                .value;
            let label = format!(
                "{form} prefill kv {} g {} w {} {rows} rows after {history} {config:?}",
                geometry.kv,
                geometry.g,
                geometry.w()
            );
            check(&label, &encoded, &gated, &bound, &expected);
            let actual = bf16_values(&gated);
            assert_eq!(actual.len(), reference.len());
            let (mut error, mut norm, mut peak, mut outside) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            for (a, e) in actual.iter().zip(&reference) {
                assert!(a.is_finite(), "{label}: output is finite");
                let difference = f64::from(a - e);
                error += difference * difference;
                norm += f64::from(*e) * f64::from(*e);
                peak = peak.max(difference.abs());
                outside += usize::from(difference.abs() > 1.0e-2 + 1.0e-2 * f64::from(e.abs()));
            }
            let rms = (norm / reference.len() as f64).sqrt();
            let relative = (error / norm).sqrt();
            eprintln!(
                "{label}: relative RMS difference from the default {relative:.2e}, largest {:.2e} reference RMS, {outside} of {} elements outside the BF16 tolerance",
                peak / rms,
                actual.len()
            );
            assert_eq!(outside, 0, "{label}: elements outside the BF16 tolerance of the default");
            let (planes, direct_planes) = (bound.planes(), direct_bound.planes());
            for vector in 0..encoded.case.history_rows * geometry.kv {
                assert_eq!(
                    planes.decoded(geometry, vector),
                    direct_planes.decoded(geometry, vector),
                    "{label}: history vector {vector}"
                );
            }
        }
    }
}

/// Two requests in one batch over a store much larger than what they see:
/// each history is spans in different slabs, out of address order, one
/// continuing from a slab's last row into another slab, and the second
/// request is a fork sharing rows of the first's. The query tile that holds
/// rows of both sees, per span index, two spans with thousands of rows the
/// call does not read between them. `compact` is the same call with the used
/// slabs moved, in address order, to the start of a store of just those
/// slabs.
fn scattered_history(geometry: Geometry) -> (Encoded, Encoded, usize) {
    const SLAB: i32 = 768;
    let (first, second) = ((5 * SLAB + 300, 6 * SLAB), (2 * SLAB, 2 * SLAB + 200));
    let fork = [
        (5 * SLAB + 300, 5 * SLAB + 556),
        (11 * SLAB + 512, 11 * SLAB + 700),
        (9 * SLAB, 9 * SLAB + 100),
    ];
    let rows = (0..48)
        .map(|row| match row {
            0..=23 => Row {
                spans: vec![first, second],
                fresh: (0, row + 1),
                destination: second.1 + row,
                position: 668 + row,
            },
            24..=43 => Row {
                spans: fork.to_vec(),
                fresh: (24, row + 1),
                destination: if row % 7 == 3 { -1 } else { fork[2].1 + row - 24 },
                position: 544 + row - 24,
            },
            _ => Row { spans: vec![], fresh: (0, 0), destination: -1, position: 0 },
        })
        .collect::<Vec<_>>();
    let scattered = Case::new(geometry, 16 * SLAB as usize, 3, &rows, 31);
    // Slabs 2, 5, 9 and 11 become slabs 0 to 3.
    let moved = |row: i32| match row / SLAB {
        2 => row - 2 * SLAB,
        5 => row - 4 * SLAB,
        9 => row - 7 * SLAB,
        11 => row - 8 * SLAB,
        _ => panic!("row {row} is in no used slab"),
    };
    let mut compact = scattered.clone();
    compact.history_rows = 4 * SLAB as usize;
    let vector = geometry.kv * geometry.w();
    let slabs = |plane: &[f32]| {
        [2, 5, 9, 11]
            .into_iter()
            .flat_map(|slab| plane[slab * SLAB as usize * vector..][..SLAB as usize * vector].to_vec())
            .collect::<Vec<_>>()
    };
    compact.history_key = slabs(&scattered.history_key);
    compact.history_value = slabs(&scattered.history_value);
    for span in compact.visible.chunks_exact_mut(2) {
        if span[1] > span[0] {
            (span[0], span[1]) = (moved(span[0]), moved(span[1] - 1) + 1);
        }
    }
    for destination in &mut compact.destinations {
        if *destination >= 0 {
            *destination = moved(*destination);
        }
    }
    (Encoded::new(scattered), Encoded::new(compact), SLAB as usize)
}

/// Metal's prefill forms read only the history row tiles the call lists,
/// wherever they lie in the store: over `scattered_history` every
/// configuration matches the host model, and an unsplit configuration's
/// result is bit for bit that of the compact layout (a moved slab keeps every
/// span's place in its slab, so the call takes the same steps; a split
/// configuration counts the rows between two requests' spans into its
/// partitions).
#[test]
fn metal_prefill_reads_scattered_history_tiles() {
    let Some(device) = k8v4_devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    for geometry in [MINICPM5, QWEN] {
        let (scattered, compact, slab_rows) = scattered_history(geometry);
        let expected = scattered.expected();
        let run = |encoded: &Encoded, config: &[(&'static str, u64)]| {
            let mut bound = Bound::new_with_slab_rows(&device, encoded, slab_rows, false);
            let gated = prefill_kernel(&device, geometry, config)
                .call(prefill_args!(bound, encoded.case))
                .unwrap()
                .value;
            (gated, bound)
        };
        let default = [
            ("QT", 16),
            ("HEADS", single_head_group(geometry.g)),
            ("SPLIT_GROUPS", 256),
            ("DIRECT", 0),
        ];
        let reference = bf16_values(&run(&scattered, &default).0);
        for config in prefill_configs(BackendName::Metal, geometry)
            .into_iter()
            .chain(coissue_prefill_configs(geometry))
        {
            let label = format!("scattered prefill {config:?}");
            let (gated, bound) = run(&scattered, &config);
            check(&label, &scattered, &gated, &bound, &expected);
            if config.contains(&("SPLIT_GROUPS", 1)) {
                assert!(
                    gated.read_to_host().unwrap() == run(&compact, &config).0.read_to_host().unwrap(),
                    "{label}: differs from the compact layout"
                );
            }
        }
        // A call that lists no tiles (L = 0) is served by the staged form
        // alone, which reads the history in place: the default's result bit
        // for bit. The forms over decoded history are outside its domain.
        let unlisted = |direct: u64| {
            attention_prefill_k8v4::native_for_device_with(
                &device,
                attention_prefill_k8v4::Elements { A: Element::bf16() },
                &specialization_on(
                    &device,
                    geometry,
                    &[("QT", 16), ("SPLIT_GROUPS", 256), ("DIRECT", direct)],
                )
                .with_static("L", 0)
                .with_param("COISSUE", 0),
            )
        };
        assert!(unlisted(1).is_err(), "the direct form takes a call without a tile list");
        let mut bound = Bound::new_with_slab_rows(&device, &scattered, slab_rows, false);
        bound.history_tiles = i32_tensor(&device, &[0, 1], &[]);
        let gated = unlisted(0)
            .unwrap()
            .call(prefill_args!(bound, scattered.case))
            .unwrap()
            .value;
        assert!(
            bf16_values(&gated) == reference,
            "the staged form without a tile list differs from the default"
        );
    }
}

/// The forms over decoded history attend a history their window does not
/// hold in rounds, folding each round's split records into the state the
/// window keeps.
/// Sixteen rows of Qwen heads have a window of 7936 rows unsplit and of 3840
/// at 256 split groups (64 key partitions, half of the most): 40 history
/// tiles (with eight tiles between them that no row sees) take two rounds
/// and three, each round splitting its own keys.
///
/// A round is to a row what a key partition is: its probabilities are
/// rounded to F16 against its own maximum and its state merges by the merge
/// rule. With spans that start on tile boundaries in address order, where
/// the rounds take the keys in the order one round does, the result agrees
/// with the same rows in a launch of 32 (whose window holds the whole
/// history) as a split configuration agrees with an unsplit one: within two
/// bf16 steps, or 2^-11 of the output's largest magnitude where that is
/// more. Spans out of address order and off the tile grid, differing
/// between rows, match the host model like every configuration.
#[test]
fn metal_direct_prefill_resumes_across_rounds() {
    let Some(device) = k8v4_devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let geometry = QWEN;
    const STORE: usize = 12544;
    let case = |spans: &dyn Fn(i32) -> Vec<(i32, i32)>, launch: i32| {
        let rows = (0..launch)
            .map(|row| match row {
                0..=15 => Row {
                    spans: spans(row),
                    fresh: (0, row + 1),
                    destination: 12288 + row,
                    position: 10240 + row,
                },
                _ => Row { spans: vec![], fresh: (0, 0), destination: -1, position: 0 },
            })
            .collect::<Vec<_>>();
        Case::new(geometry, STORE, 2, &rows, 17)
    };
    // `form` names the form over the window (with its head group size);
    // `listed` pads the tile list to that many entries: the call of a class
    // that lists more tiles than its rows see.
    let run_form = |encoded: &Encoded, split: u64, form: &[(&'static str, u64)], listed: Option<usize>| {
        let mut bound = Bound::new(&device, encoded);
        if let Some(listed) = listed {
            let mut tiles = history_tiles(&encoded.case.visible);
            tiles.resize(listed, -1);
            bound.history_tiles = i32_tensor(&device, &[1, listed], &tiles);
        }
        let config = [("QT", 16), ("SPLIT_GROUPS", split), ("DIRECT", 1)]
            .into_iter()
            .chain(form.iter().copied())
            .collect::<Vec<_>>();
        let gated = prefill_kernel(&device, geometry, &config)
            .call(prefill_args!(bound, encoded.case))
            .unwrap()
            .value;
        (gated, bound)
    };
    let direct = [("HEADS", single_head_group(geometry.g))];
    let run_listing =
        |encoded: &Encoded, split: u64, listed: Option<usize>| run_form(encoded, split, &direct, listed);
    let run = |encoded: &Encoded, split: u64| run_listing(encoded, split, None);

    // The launch of 32 is the sixteen rows and sixteen that see nothing;
    // the launch of 16 is its first half.
    let aligned = |_: i32| vec![(0, 6144), (8192, 12288)];
    let one_round = case(&aligned, 32);
    let mut rounds = one_round.clone();
    rounds.rows = 16;
    for plane in [&mut rounds.query_gate, &mut rounds.key, &mut rounds.value] {
        plane.truncate(plane.len() / 2);
    }
    for controls in [
        &mut rounds.coordinates,
        &mut rounds.visible,
        &mut rounds.fresh,
        &mut rounds.destinations,
    ] {
        controls.truncate(controls.len() / 2);
    }
    let (rounds, one_round) = (Encoded::new(rounds), Encoded::new(one_round));
    let expected = rounds.expected();
    for split in [1, 256] {
        let (gated, bound) = run(&rounds, split);
        check(&format!("aligned rounds, {split} split groups"), &rounds, &gated, &bound, &expected);
    }
    // The co-issue form (on simdgroup matrices; two pairs a threadgroup)
    // takes the same rounds over the window, and matches the host model.
    for split in [1, 256] {
        let coissue = [("HEADS", 1), ("COISSUE", 1)];
        let (gated, bound) = run_form(&rounds, split, &coissue, None);
        check(&format!("aligned rounds, co-issue, {split} split groups"), &rounds, &gated, &bound, &expected);
    }
    let resumed = bf16_values(&run(&rounds, 1).0);
    let whole = bf16_values(&run(&one_round, 1).0);
    assert_eq!(resumed.len() * 2, whole.len());
    let scale = whole.iter().fold(0f32, |largest, value| largest.max(value.abs()));
    for (index, (resumed, whole)) in resumed.iter().zip(&whole).enumerate() {
        let step = f32::from_bits((resumed.abs().max(whole.abs()).to_bits() & 0x7f80_0000).saturating_sub(7 << 23));
        assert!(
            (resumed - whole).abs() <= (2.0 * step).max(scale / 2048.0),
            "element {index}: {resumed} over rounds, {whole} in one"
        );
    }

    // A class that lists far more tiles than the call's rows see dispatches
    // rounds the call does not need: the first takes everything, the rest
    // return at once, and the result is that of the exact list bit for bit.
    for split in [1, 256] {
        assert!(
            run_listing(&one_round, split, Some(1024)).0.read_to_host().unwrap()
                == run(&one_round, split).0.read_to_host().unwrap(),
            "a longer tile list changes the result at {split} split groups"
        );
    }

    let ragged = |row: i32| match row {
        0..=7 => vec![(8292, 12288), (37, 6000)],
        _ => vec![(8492, 12000), (0, 5000 + 16 * row)],
    };
    let rounds = Encoded::new(case(&ragged, 16));
    let expected = rounds.expected();
    for split in [1, 256] {
        let (gated, bound) = run(&rounds, split);
        check(&format!("ragged rounds, {split} split groups"), &rounds, &gated, &bound, &expected);
    }
}

/// Gated outputs agree with the host model within bf16 publication plus
/// reduced-precision staging. Planes: vectors nothing appends to are
/// untouched; appended vectors decode within one code step of the host
/// model's encoding (the key may differ by one bf16 step from transcendental
/// rounding, which can move a code).
fn check(
    label: &str,
    encoded: &Encoded,
    gated: &Tensor,
    bound: &Bound,
    expected: &(Vec<f32>, Planes),
) {
    let case = &encoded.case;
    let geometry = case.geometry;
    let actual = bf16_values(gated);
    assert_eq!(actual.len(), expected.0.len(), "{label}: gated shape");
    let mut worst = 0.0f32;
    for (index, (a, e)) in actual.iter().zip(&expected.0).enumerate() {
        let error = (a - e).abs();
        worst = worst.max(error);
        assert!(
            error <= 4.0e-3 + 1.6e-2 * e.abs(),
            "{label}: gated[{index}] (row {}, head {}, column {}) device {a} expected {e}",
            index / (geometry.kv * geometry.g * geometry.w()),
            index / geometry.w() % (geometry.kv * geometry.g),
            index % geometry.w()
        );
    }
    let planes = bound.planes();
    let appended = case
        .destinations
        .iter()
        .filter(|destination| **destination >= 0)
        .flat_map(|destination| {
            (0..geometry.kv).map(move |head| *destination as usize * geometry.kv + head)
        })
        .collect::<Vec<_>>();
    let w = geometry.w();
    for vector in 0..case.history_rows * geometry.kv {
        let (key, value) = planes.decoded(geometry, vector);
        let (expected_key, expected_value) = expected.1.decoded(geometry, vector);
        if appended.contains(&vector) {
            for column in 0..w {
                let step = |coefficients: &[u16]| {
                    f16_to_f32(coefficients[vector * coefficient_elements(w) + column / GROUP * 2])
                };
                let (key_step, value_step) = (
                    step(&expected.1.key_coefficients),
                    step(&expected.1.value_coefficients),
                );
                assert!(
                    (key[column] - expected_key[column]).abs()
                        <= 1.01 * key_step + 8.0e-3 * expected_key[column].abs().max(1.0),
                    "{label}: appended key vector {vector}[{column}] device {} expected {}",
                    key[column],
                    expected_key[column]
                );
                assert!(
                    (value[column] - expected_value[column]).abs() <= 1.01 * value_step,
                    "{label}: appended value vector {vector}[{column}] device {} expected {}",
                    value[column],
                    expected_value[column]
                );
            }
        } else {
            assert_eq!(
                (key, value),
                (expected_key, expected_value),
                "{label}: vector {vector} was not appended to"
            );
        }
    }
    eprintln!("{label}: max |gated error| {worst:.2e}");
}

/// The portable body (reference interpreter) agrees with the host model of
/// it, so the host model stands for the body at shapes the interpreter cannot
/// reach. Value planes agree exactly; keys within the dense tests' key
/// tolerance.
fn check_host_model_against_portable_body(entry: &str, encoded: &Encoded) {
    use seismic_lang::{
        checked::{check_source, SourceFile},
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        reference_math::ReferenceScalar,
        registry,
        types::DType,
    };
    let case = &encoded.case;
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
    let Geometry { kv, g, p, .. } = case.geometry;
    let (m, w, t) = (case.rows, case.geometry.w(), case.history_rows);
    let mut interpreter = Interpreter::new(&logical);
    let mut tensor = |dtype, shape: Vec<usize>, values: Vec<f64>| {
        Arg::Tensor(interpreter.add_tensor(TensorData::dense(dtype, shape, values)))
    };
    let floats = |values: &[f32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let ints = |values: &[i32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let unsigned = |values: &[u32]| values.iter().map(|x| f64::from(*x)).collect::<Vec<_>>();
    let halves = |values: &[u16]| {
        values
            .iter()
            .map(|x| f64::from(f16_to_f32(*x)))
            .collect::<Vec<_>>()
    };
    let planes = &encoded.planes;
    let mut args = vec![
        tensor(
            DType::BF16,
            vec![m, kv * g, 2 * w],
            floats(&case.query_gate),
        ),
        tensor(DType::BF16, vec![m, kv * g, 0], Vec::new()),
        tensor(DType::BF16, vec![1, m, kv * w], floats(&case.key)),
        tensor(DType::BF16, vec![1, m, kv * w], floats(&case.value)),
        tensor(DType::F32, vec![1, w], floats(&case.query_norm)),
        tensor(DType::F32, vec![1, w], floats(&case.key_norm)),
        tensor(DType::F32, vec![0, w], Vec::new()),
        tensor(DType::I32, vec![p], ints(&case.components)),
        tensor(DType::F32, vec![p], floats(&case.frequencies)),
        tensor(DType::F32, vec![p], vec![1.0; p]),
        tensor(DType::I32, vec![m, 4], ints(&case.coordinates)),
        tensor(DType::I32, vec![m, case.spans, 2], ints(&case.visible)),
        tensor(DType::I32, vec![m, 2], ints(&case.fresh)),
        tensor(DType::I32, vec![m], ints(&case.destinations)),
        tensor(DType::U32, vec![t, kv, w / 4], unsigned(&planes.key_codes)),
        tensor(
            DType::F16,
            vec![t, kv, coefficient_elements(w)],
            halves(&planes.key_coefficients),
        ),
        tensor(
            DType::U32,
            vec![t, kv, w / 8],
            unsigned(&planes.value_codes),
        ),
        tensor(
            DType::F16,
            vec![t, kv, coefficient_elements(w)],
            halves(&planes.value_coefficients),
        ),
        Arg::Scalar(ReferenceScalar::F32(case.epsilon.to_bits())),
        Arg::Scalar(ReferenceScalar::F32(case.scale.to_bits())),
        Arg::Scalar(ReferenceScalar::I32(0)),
        Arg::Scalar(ReferenceScalar::U32(t as u32)),
    ];
    // The prefill entry also takes the history row tiles its rows see, after
    // the destinations.
    if entry == "attention_prefill_k8v4" {
        let tiles = history_tiles(&case.visible);
        args.insert(14, tensor(DType::I32, vec![1, tiles.len()], ints(&tiles)));
    }
    let outcome = interpreter.run(&args).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("{entry} portable body failed: {failure}");
    }
    let expected = encoded.expected();
    let result = outcome.results().next().unwrap();
    let OutcomeValue::Tensor(gated) = result.value() else {
        panic!("{entry} returns a tensor")
    };
    for (index, e) in expected.0.iter().enumerate() {
        let a = gated.read(index).unwrap() as f32;
        assert!(
            (a - e).abs() <= 1.0e-3 + 8.0e-3 * e.abs(),
            "{entry} portable gated[{index}] {a}, host model {e}"
        );
    }
    let inputs = outcome.inputs().collect::<Vec<_>>();
    let read = |ordinal: usize| {
        let input = inputs
            .iter()
            .find(|input| input.ordinal() == ordinal)
            .unwrap();
        let tensor = input.tensor();
        (0..tensor.element_count())
            .map(|index| tensor.read(index).unwrap())
            .collect::<Vec<_>>()
    };
    // The planes follow the row tables (and the prefill entry's tiles).
    let planes = args.len() - 8;
    let portable = Planes {
        key_codes: read(planes).into_iter().map(|x| x as u32).collect(),
        key_coefficients: read(planes + 1).into_iter().map(|x| f16_bits(x as f32)).collect(),
        value_codes: read(planes + 2).into_iter().map(|x| x as u32).collect(),
        value_coefficients: read(planes + 3).into_iter().map(|x| f16_bits(x as f32)).collect(),
    };
    assert_eq!(
        portable.value_codes, expected.1.value_codes,
        "{entry} portable value codes"
    );
    assert_eq!(
        portable.value_coefficients, expected.1.value_coefficients,
        "{entry} portable value coefficients"
    );
    for vector in 0..t * kv {
        let (key, _) = portable.decoded(case.geometry, vector);
        let (expected_key, _) = expected.1.decoded(case.geometry, vector);
        for column in 0..w {
            let step = f16_to_f32(
                expected.1.key_coefficients[vector * coefficient_elements(w) + column / GROUP * 2],
            );
            assert!(
                (key[column] - expected_key[column]).abs()
                    <= 1.01 * step + 8.0e-3 * expected_key[column].abs().max(1.0),
                "{entry} portable key vector {vector}[{column}] {}, host model {}",
                key[column],
                expected_key[column]
            );
        }
    }
}

#[test]
fn decode_matches_portable_body() {
    let encoded = Encoded::new(Case::new(GROUPED, 64, 2, &decode_rows(5), 11));
    check_host_model_against_portable_body("attention_decode_k8v4", &encoded);
    for device in k8v4_devices() {
        decode_matches_portable_body_on(&device, &encoded);
    }
}

fn decode_matches_portable_body_on(device: &Device, encoded: &Encoded) {
    let backend = device.backend();
    let expected = encoded.expected();
    for config in decode_configs(backend, GROUPED) {
        let kernel = decode_kernel(device, GROUPED, &config);
        let mut bound = Bound::new(device, encoded);
        let gated = kernel
            .call(args!(attention_decode_k8v4, bound, encoded.case))
            .unwrap()
            .value;
        check(
            &format!("{backend:?} grouped decode {config:?}"),
            encoded,
            &gated,
            &bound,
            &expected,
        );
    }
}

#[test]
fn decode_reads_and_writes_across_affine_history_slabs() {
    let rows = [
        Row {
            spans: vec![(0, 32), (32, 48), (64, 80)],
            fresh: (0, 1),
            destination: 81,
            position: 80,
        },
        Row {
            spans: vec![(16, 32), (48, 64), (64, 80)],
            fresh: (1, 2),
            destination: 82,
            position: 80,
        },
    ];
    let encoded = Encoded::new(Case::new(GROUPED, 96, 3, &rows, 73));
    let expected = encoded.expected();
    for device in k8v4_devices() {
        let backend = device.backend();
        let config = decode_configs(backend, GROUPED).into_iter().next().unwrap();
        let kernel = decode_kernel(&device, GROUPED, &config);
        for reused in [false, true] {
            let mut bound = Bound::new_with_slab_rows(&device, &encoded, 32, reused);
            let gated = kernel
                .call(args!(attention_decode_k8v4, bound, encoded.case))
                .unwrap()
                .value;
            check(
                &format!("{backend:?} affine slab decode reused={reused}"),
                &encoded,
                &gated,
                &bound,
                &expected,
            );
        }
    }
}

/// Visible spans crossing slab edges, as the device measurement binds them
/// (one span over the whole synthetic history, several slabs deep), at every
/// decode configuration: the entry walks each slab's part of a span.
#[test]
fn decode_reads_spans_crossing_affine_history_slabs() {
    let rows = [
        Row {
            spans: vec![(0, 200)],
            fresh: (0, 1),
            destination: 200,
            position: 200,
        },
        Row {
            spans: vec![(7, 150), (170, 199)],
            fresh: (1, 2),
            destination: 201,
            position: 199,
        },
    ];
    let encoded = Encoded::new(Case::new(GROUPED, 224, 2, &rows, 79));
    let expected = encoded.expected();
    for device in k8v4_devices() {
        let backend = device.backend();
        for config in decode_configs(backend, GROUPED) {
            let kernel = decode_kernel(&device, GROUPED, &config);
            // A reused middle slab puts consecutive slabs at unrelated
            // addresses.
            for reused in [false, true] {
                let mut bound = Bound::new_with_slab_rows(&device, &encoded, 32, reused);
                let gated = kernel
                    .call(args!(attention_decode_k8v4, bound, encoded.case))
                    .unwrap()
                    .value;
                check(
                    &format!("{backend:?} affine crossing-span decode {config:?} reused={reused}"),
                    &encoded,
                    &gated,
                    &bound,
                    &expected,
                );
            }
        }
    }
}

#[test]
fn prefill_matches_portable_body() {
    // 300 history rows are 10 key tiles: the split configurations split the
    // first sequence's history spans over two partitions and merge them.
    // One affine group per vector keeps the interpreter's prefill affordable;
    // the Qwen geometry test covers eight groups.
    let encoded = Encoded::new(Case::new(SMALL, 400, 2, &prefill_rows(20, 300), 23));
    check_host_model_against_portable_body("attention_prefill_k8v4", &encoded);
    for device in k8v4_devices() {
        prefill_matches_portable_body_on(&device, &encoded);
    }
}

fn prefill_matches_portable_body_on(device: &Device, encoded: &Encoded) {
    let backend = device.backend();
    let expected = encoded.expected();
    for config in prefill_configs(backend, SMALL) {
        let kernel = prefill_kernel(device, SMALL, &config);
        let mut bound = Bound::new(device, encoded);
        let gated = kernel
            .call(prefill_args!(bound, encoded.case))
            .unwrap()
            .value;
        check(
            &format!("{backend:?} small prefill {config:?}"),
            encoded,
            &gated,
            &bound,
            &expected,
        );
    }
}

#[test]
fn prefill_reads_and_writes_across_affine_history_slabs() {
    let rows = [
        Row {
            spans: vec![(0, 32), (32, 48), (64, 80)],
            fresh: (0, 1),
            destination: 81,
            position: 80,
        },
        Row {
            spans: vec![(0, 32), (32, 48), (64, 80)],
            fresh: (0, 2),
            destination: 82,
            position: 81,
        },
        Row {
            spans: vec![(16, 32), (48, 64), (64, 80)],
            fresh: (0, 3),
            destination: 83,
            position: 82,
        },
    ];
    let encoded = Encoded::new(Case::new(GROUPED, 96, 3, &rows, 79));
    let expected = encoded.expected();
    for device in k8v4_devices() {
        let backend = device.backend();
        let config = prefill_configs(backend, GROUPED)
            .into_iter()
            .next()
            .unwrap();
        let kernel = prefill_kernel(&device, GROUPED, &config);
        let mut bound = Bound::new_with_slab_rows(&device, &encoded, 32, true);
        let gated = kernel
            .call(prefill_args!(bound, encoded.case))
            .unwrap()
            .value;
        check(
            &format!("{backend:?} affine slab prefill reused=true"),
            &encoded,
            &gated,
            &bound,
            &expected,
        );
    }
}

#[test]
fn qwen_geometry_decode_and_prefill_match_host_model() {
    for device in k8v4_devices() {
        qwen_geometry_decode_and_prefill_match_host_model_on(&device);
    }
}

fn qwen_geometry_decode_and_prefill_match_host_model_on(device: &Device) {
    let backend = device.backend();
    for context in [256, 4096, 16384] {
        let encoded = Encoded::new(Case::new(
            QWEN,
            context + 128,
            2,
            &decode_rows(context as i32 - 40),
            5,
        ));
        let expected = encoded.expected();
        for config in decode_configs(backend, QWEN) {
            let kernel = decode_kernel(device, QWEN, &config);
            let mut bound = Bound::new(device, &encoded);
            let gated = kernel
                .call(args!(attention_decode_k8v4, bound, encoded.case))
                .unwrap()
                .value;
            check(
                &format!("{backend:?} decode context {context} {config:?}"),
                &encoded,
                &gated,
                &bound,
                &expected,
            );
        }
        for rows in [1, 8] {
            let encoded = Encoded::new(Case::new(
                QWEN,
                context + rows,
                1,
                &speculative_rows(rows, context as i32),
                13,
            ));
            let expected = encoded.expected();
            // The first configuration, and on Vulkan the first of the keywise
            // walk too.
            let configs = decode_configs(backend, QWEN);
            let keywise = configs
                .iter()
                .find(|config| config.contains(&("KEYWISE", 1)));
            for config in std::iter::once(&configs[0]).chain(keywise) {
                let kernel = decode_kernel(device, QWEN, config);
                let mut bound = Bound::new(device, &encoded);
                let gated = kernel
                    .call(args!(attention_decode_k8v4, bound, encoded.case))
                    .unwrap()
                    .value;
                check(
                    &format!(
                        "{backend:?} speculative decode {rows} rows context {context} {config:?}"
                    ),
                    &encoded,
                    &gated,
                    &bound,
                    &expected,
                );
            }
        }
    }
    for (rows, history) in [(40, 300), (128, 1000), (64, 4096), (48, 16384)] {
        let encoded = Encoded::new(Case::new(
            QWEN,
            history as usize + 256,
            2,
            &prefill_rows(rows, history),
            7,
        ));
        let expected = encoded.expected();
        for config in prefill_configs(backend, QWEN) {
            let kernel = prefill_kernel(device, QWEN, &config);
            let mut bound = Bound::new(device, &encoded);
            let gated = kernel
                .call(prefill_args!(bound, encoded.case))
                .unwrap()
                .value;
            check(
                &format!("{backend:?} prefill {rows} rows after {history} {config:?}"),
                &encoded,
                &gated,
                &bound,
                &expected,
            );
        }
    }
}

/// Qwen3.6-35B-A3B attention: 2 kv heads of 8 query heads each.
const QWEN35B: Geometry = Geometry {
    kv: 2,
    g: 8,
    p: 32,
    s: 192,
};

/// Qwen3.5-122B-A10B attention: 2 kv heads of 16 query heads each.
const QWEN122B: Geometry = Geometry {
    kv: 2,
    g: 16,
    p: 32,
    s: 192,
};

/// Eight query heads per kv head over 4k of affine history: decode and a
/// prefill chunk.
#[test]
fn eight_query_group_decode_and_prefill_at_4k_match_host_model() {
    large_group_decode_and_prefill_at_4k_match_host_model(QWEN35B);
}

/// Sixteen query heads per kv head over 4k of affine history: decode and a
/// prefill chunk.
#[test]
fn sixteen_query_group_decode_and_prefill_at_4k_match_host_model() {
    large_group_decode_and_prefill_at_4k_match_host_model(QWEN122B);
}

/// 512-column heads (Gemma 4 full layers: 4 kv heads of 8, 64 rotated pairs):
/// decode and a prefill chunk over 1k of affine history, on the backends
/// whose declarations take them (the others refuse by `where`).
#[test]
fn wide_head_decode_and_prefill_match_host_model() {
    let geometry = Geometry {
        kv: 2,
        g: 8,
        p: 64,
        s: 384,
    };
    for device in k8v4_devices() {
        let backend = device.backend();
        let context = 1024;
        let decode = Encoded::new(Case::new(
            geometry,
            context + 128,
            2,
            &decode_rows(context as i32 - 40),
            5,
        ));
        let prefill = Encoded::new(Case::new(
            geometry,
            context + 256,
            2,
            &prefill_rows(40, context as i32),
            7,
        ));
        for (encoded, configs, is_decode) in [
            (&decode, decode_configs(backend, geometry), true),
            (&prefill, prefill_configs(backend, geometry), false),
        ] {
            let expected = encoded.expected();
            for config in configs {
                let specialization = specialization_on(&device, geometry, &config);
                let label = format!(
                    "{backend:?} wide {} {config:?}",
                    if is_decode { "decode" } else { "prefill" }
                );
                let mut bound = Bound::new(&device, encoded);
                let result = if is_decode {
                    attention_decode_k8v4::native_for_device_with(
                        &device,
                        attention_decode_k8v4::Elements { A: Element::bf16() },
                        &specialization,
                    )
                    .map(|kernel| {
                        kernel
                            .call(args!(attention_decode_k8v4, bound, encoded.case))
                            .unwrap()
                            .value
                    })
                } else {
                    attention_prefill_k8v4::native_for_device_with(
                        &device,
                        attention_prefill_k8v4::Elements { A: Element::bf16() },
                        &prefill_specialization(&device, geometry, &config),
                    )
                    .map(|kernel| {
                        kernel
                            .call(prefill_args!(bound, encoded.case))
                            .unwrap()
                            .value
                    })
                };
                match result {
                    Ok(gated) => check(&label, encoded, &gated, &bound, &expected),
                    Err(error) => {
                        let error = error.to_string();
                        assert!(
                            error.contains("violates the native `where` condition")
                                && matches!(backend, BackendName::Cuda | BackendName::Vulkan),
                            "{label}: {error}"
                        );
                        eprintln!("{label}: rejected by the declaration's `where`");
                    }
                }
            }
        }
    }
}

fn large_group_decode_and_prefill_at_4k_match_host_model(geometry: Geometry) {
    let group = geometry.g;
    for device in k8v4_devices() {
        let backend = device.backend();
        let context = 4096;
        let encoded = Encoded::new(Case::new(
            geometry,
            context + 128,
            2,
            &decode_rows(context as i32 - 40),
            5,
        ));
        let expected = encoded.expected();
        for config in decode_configs(backend, geometry) {
            let kernel = decode_kernel(&device, geometry, &config);
            let mut bound = Bound::new(&device, &encoded);
            let gated = kernel
                .call(args!(attention_decode_k8v4, bound, encoded.case))
                .unwrap()
                .value;
            check(
                &format!("{backend:?} group-{group} decode context {context} {config:?}"),
                &encoded,
                &gated,
                &bound,
                &expected,
            );
        }
        let encoded = Encoded::new(Case::new(
            geometry,
            context + 256,
            2,
            &prefill_rows(64, context as i32),
            7,
        ));
        let expected = encoded.expected();
        for config in prefill_configs(backend, geometry) {
            let kernel = prefill_kernel(&device, geometry, &config);
            let mut bound = Bound::new(&device, &encoded);
            let gated = kernel
                .call(prefill_args!(bound, encoded.case))
                .unwrap()
                .value;
            check(
                &format!("{backend:?} group-{group} prefill 64 rows after {context} {config:?}"),
                &encoded,
                &gated,
                &bound,
                &expected,
            );
        }
    }
}

/// The dense history planes of a case, for timing the dense entries on the
/// same inputs.
struct DenseHistory {
    _slabs: SlabTensor,
    key: Tensor,
    value: Tensor,
}

impl DenseHistory {
    fn new(device: &Device, case: &Case) -> Self {
        let (kv, w, t) = (case.geometry.kv, case.geometry.w(), case.history_rows);
        let mut slabs = SlabTensor::new(
            device,
            t as u64,
            t as u64,
            vec![
                SlabRegion {
                    element: Element::bf16(),
                    row_shape: vec![kv as u64, w as u64],
                },
                SlabRegion {
                    element: Element::bf16(),
                    row_shape: vec![kv as u64, w as u64],
                },
            ],
        )
        .unwrap();
        slabs.add_slab().unwrap();
        for (region, values) in [&case.history_key, &case.history_value]
            .into_iter()
            .enumerate()
        {
            let bytes = values
                .iter()
                .flat_map(|value| bf16_bits(*value).to_le_bytes())
                .collect::<Vec<_>>();
            slabs
                .region_rows(region, 0, t as u64)
                .unwrap()
                .write_from_host(&bytes)
                .unwrap();
        }
        Self {
            key: slabs.logical_region(0).unwrap(),
            value: slabs.logical_region(1).unwrap(),
            _slabs: slabs,
        }
    }
}

macro_rules! dense_args {
    ($module:ident, $bound:expr, $dense:expr, $case:expr) => {
        $module::Args {
            query: &$bound.query,
            gate: &$bound.gate,
            key: &$bound.key,
            value: &$bound.value,
            query_norm: &$bound.query_norm,
            key_norm: &$bound.key_norm,
            value_norm: &$bound.value_norm,
            rotary_components: &$bound.components,
            rotary_frequencies: &$bound.frequencies,
            rotary_amplitudes: &$bound.amplitudes,
            coordinates: &$bound.coordinates,
            visible: &$bound.visible,
            fresh: &$bound.fresh,
            destinations: &$bound.destinations,
            history_key: &mut $dense.key,
            history_value: &mut $dense.value,
            epsilon: $case.epsilon,
            scale: $case.scale,
            gate_function: 0,
            slab_rows: $case.history_rows as u32,
        }
    };
}

/// Device time of one decode layer's attention at Qwen3.5 4B, 35B and 122B geometry over
/// affine history, next to the dense entry on the same inputs
/// (`--ignored --nocapture decode_timing`). "KV read" counts the history rows
/// read (affine: codes + coefficients).
#[test]
#[ignore]
fn decode_timing() {
    for device in k8v4_devices() {
        decode_timing_on(&device);
    }
}

/// MiniCPM5-2B's attention: 2 kv heads of 8 query heads, W = 128, full
/// rotation.
const MINICPM5: Geometry = Geometry {
    kv: 2,
    g: 8,
    p: 64,
    s: 0,
};

#[test]
fn metal_packed_four_minicpm5_matches_host_model() {
    let Some(device) = k8v4_devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let geometry = MINICPM5;
    let context = 4096;
    let encoded = Encoded::new(Case::new(
        geometry,
        context + 128,
        2,
        &decode_rows(context as i32 - 40),
        5,
    ));
    let expected = encoded.expected();
    for keys in [16, 32] {
        let config = [
            ("SPAN", 32),
            ("PARTS", 16),
            ("SIMDS", 4),
            ("SLICES", 1),
            ("MATRIX", 1),
            ("KEYS", keys),
            ("TOKENS", 4),
        ];
        let kernel = decode_kernel(&device, geometry, &config);
        let mut bound = Bound::new(&device, &encoded);
        let gated = kernel
            .call(args!(attention_decode_k8v4, bound, encoded.case))
            .unwrap()
            .value;
        check(
            &format!("Metal packed-four MiniCPM5 decode KEYS={keys}"),
            &encoded,
            &gated,
            &bound,
            &expected,
        );
    }
}

#[test]
fn metal_packed_four_qwen35b_matches_host_model() {
    let Some(device) = k8v4_devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let geometry = QWEN35B;
    let context = 4096;
    for (label, rows) in [
        (
            "M2",
            decode_rows(context as i32 - 40)
                .into_iter()
                .take(2)
                .collect(),
        ),
        ("M4", decode_rows(context as i32 - 40)),
        ("M8", prefill_rows(8, context as i32)),
    ] {
        let encoded = Encoded::new(Case::new(geometry, context + 128, 2, &rows, 5));
        let expected = encoded.expected();
        // W = 256 packs one token per simdgroup; 16-key K and V tiles fit.
        for keys in [8, 16] {
            let config = [
                ("SPAN", 32),
                ("PARTS", 16),
                ("SIMDS", 4),
                ("SLICES", 1),
                ("MATRIX", 1),
                ("KEYS", keys),
                ("TOKENS", 4),
            ];
            let kernel = decode_kernel(&device, geometry, &config);
            let mut bound = Bound::new(&device, &encoded);
            let gated = kernel
                .call(args!(attention_decode_k8v4, bound, encoded.case))
                .unwrap()
                .value;
            check(
                &format!("Metal packed-four Qwen35B {label} KEYS={keys}"),
                &encoded,
                &gated,
                &bound,
                &expected,
            );
        }
    }
}

#[test]
fn metal_qwen35b_single_row_matrix_slices_match_host_model() {
    let Some(device) = k8v4_devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let geometry = QWEN35B;
    for context in [256, 4096] {
        let rows = decode_rows(context as i32 - 40)
            .into_iter()
            .take(1)
            .collect::<Vec<_>>();
        let encoded = Encoded::new(Case::new(geometry, context + 128, 2, &rows, 5));
        let expected = encoded.expected();
        let config = [
            ("SPAN", 64),
            ("PARTS", 64),
            ("SIMDS", 2),
            ("SLICES", 1),
            ("MATRIX", 1),
            ("KEYS", 8),
            ("TOKENS", 1),
        ];
        let kernel = decode_kernel(&device, geometry, &config);
        let mut bound = Bound::new(&device, &encoded);
        let gated = kernel
            .call(args!(attention_decode_k8v4, bound, encoded.case))
            .unwrap()
            .value;
        check(
            &format!("Metal Qwen35B M1 matrix slices context={context}"),
            &encoded,
            &gated,
            &bound,
            &expected,
        );
    }
}

#[test]
fn metal_packed_four_qwen35b_long_context_matches_host_model() {
    let Some(device) = k8v4_devices()
        .into_iter()
        .find(|device| device.backend() == BackendName::Metal)
    else {
        return;
    };
    let geometry = QWEN35B;
    let context = 65_536;
    let rows = decode_rows(context as i32 - 40);
    let encoded = Encoded::new(Case::new(geometry, context + 128, 2, &rows, 5));
    let expected = encoded.expected();
    let config = [
        ("SPAN", 32),
        ("PARTS", 64),
        ("SIMDS", 4),
        ("SLICES", 1),
        ("MATRIX", 1),
        ("KEYS", 16),
        ("TOKENS", 4),
    ];
    let kernel = decode_kernel(&device, geometry, &config);
    let mut bound = Bound::new(&device, &encoded);
    let gated = kernel
        .call(args!(attention_decode_k8v4, bound, encoded.case))
        .unwrap()
        .value;
    check(
        "Metal packed-four Qwen35B M4 65k P64",
        &encoded,
        &gated,
        &bound,
        &expected,
    );
}

/// Gemma 4 31B's full-attention layers: 4 kv heads of 8 query heads,
/// W = 512.
const GEMMA31_FULL: Geometry = Geometry {
    kv: 4,
    g: 8,
    p: 64,
    s: 384,
};

/// Gemma 4 26B-A4B's sliding layers: 8 kv heads of 2 query heads, W = 256.
const GEMMA26_SLIDING: Geometry = Geometry {
    kv: 8,
    g: 2,
    p: 128,
    s: 0,
};

/// Gemma 4 26B-A4B's full-attention layers: 2 kv heads of 8 query heads,
/// W = 512.
const GEMMA26_FULL: Geometry = Geometry {
    kv: 2,
    g: 8,
    p: 256,
    s: 0,
};

/// Gemma 4 E4B's full-attention layers: 2 kv heads of 4 query heads,
/// W = 512.
const GEMMA_E4B_FULL: Geometry = Geometry {
    kv: 2,
    g: 4,
    p: 256,
    s: 0,
};

/// The decode configurations a timing sweeps: the test configurations plus
/// the larger partition counts long histories over few kv heads need.
fn timing_decode_configs(
    backend: BackendName,
    geometry: Geometry,
) -> Vec<Vec<(&'static str, u64)>> {
    let group = geometry.g;
    let mut configs = decode_configs(backend, geometry);
    if backend == BackendName::Metal {
        let sweep = [8u64, 16, 32, 64, 128, 256, 512]
            .into_iter()
            .flat_map(|parts| [2u64, 4, 8].into_iter().map(move |simds| (parts, simds)))
            .flat_map(|(parts, simds)| {
                [8u64, 16, 32]
                    .into_iter()
                    .map(move |keys| (parts, simds, keys))
            })
            .flat_map(|(parts, simds, keys)| {
                [32u64, 128]
                    .into_iter()
                    .map(move |span| (span, parts, simds, keys))
            })
            .collect::<Vec<_>>();
        configs.extend(metal_matrix_configs(geometry, &sweep));
        // The packed form: one simdgroup per token of four.
        if geometry.g == 8 && geometry.w() == 256 {
            for parts in [16, 32, 64, 128, 256] {
                for keys in [8, 16] {
                    configs.push(vec![
                        ("SPAN", 32),
                        ("PARTS", parts),
                        ("SIMDS", 4),
                        ("SLICES", 1),
                        ("MATRIX", 1),
                        ("KEYS", keys),
                        ("TOKENS", 4),
                    ]);
                }
            }
        }
    }
    if backend == BackendName::Vulkan {
        let sweep = [16u64, 32, 64, 128, 256]
            .into_iter()
            .flat_map(|parts| [1u64, 2, 4].into_iter().map(move |simds| (parts, simds)))
            .collect::<Vec<_>>();
        configs.extend(vulkan_matrix_configs(geometry, &sweep));
    }
    if backend == BackendName::Cuda {
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
        configs.extend(cuda_matrix_configs(geometry, &sweep));
    }
    let slicings = [1u64, 2, 4, 8]
        .into_iter()
        .filter(|s| group as u64 % s == 0)
        .collect::<Vec<_>>();
    match backend {
        BackendName::Metal | BackendName::Vulkan => {
            // Vulkan also times both vector walks, the keywise one down to one
            // subgroup (the sweep skips what the `where` refuses).
            let simd_counts: &[u64] = if backend == BackendName::Vulkan {
                &[1, 2, 4, 8]
            } else {
                &[4, 8]
            };
            for parts in [64u64, 128, 256, 512] {
                for &simds in simd_counts {
                    for &slices in slicings.iter().filter(|s| simds % **s == 0) {
                        let mut config = vec![
                            ("SPAN", 32),
                            ("PARTS", parts),
                            ("SIMDS", simds),
                            ("SLICES", slices),
                        ];
                        config.push(("MATRIX", 0));
                        if backend == BackendName::Metal {
                            config.extend([("KEYS", 16), ("TOKENS", 1)]);
                            configs.push(config);
                        } else {
                            for keywise in [0, 1] {
                                let mut config = config.clone();
                                config.push(("KEYWISE", keywise));
                                configs.push(config);
                            }
                        }
                    }
                }
            }
        }
        BackendName::Cuda => {
            for warps in [4u64, 8] {
                for &slices in slicings.iter().filter(|s| warps % **s == 0) {
                    configs.push(vec![
                        ("PARTS", 96),
                        ("WARPS", warps),
                        ("SLICES", slices),
                        ("MATRIX", 0),
                        ("STAGES", 2),
                        ("COLUMNS", 1),
                    ]);
                }
            }
        }
        _ => {}
    }
    configs
}

fn decode_timing_on(device: &Device) {
    let backend = device.backend();
    // `K8V4_DECODE_GEOMETRY=<name>[,<name>...]` (qwen4b, qwen35b, qwen122b,
    // minicpm5, gemma31, gemma26sliding, gemma26) times those models' heads over the wider
    // configuration sweep instead of the Qwen geometries.
    let (geometries, contexts, sweep): (Vec<Geometry>, Vec<usize>, bool) =
        match std::env::var("K8V4_DECODE_GEOMETRY") {
            Ok(names) => (
                names
                    .split(',')
                    .map(|name| match name {
                        "qwen4b" => QWEN,
                        "qwen35b" => QWEN35B,
                        "qwen122b" => QWEN122B,
                        "minicpm5" => MINICPM5,
                        "gemma31" => GEMMA31_FULL,
                        "gemma26sliding" => GEMMA26_SLIDING,
                        "gemma26" => GEMMA26_FULL,
                        other => panic!("unknown K8V4_DECODE_GEOMETRY {other}"),
                    })
                    .collect(),
                vec![256, 4096, 16384, 65536, 131072],
                true,
            ),
            Err(_) => (
                vec![QWEN, QWEN35B, QWEN122B],
                vec![1, 256, 4096, 16384, 65536],
                false,
            ),
        };
    // Narrow a timing sweep to one context and matching native parameters.
    let selected_context = std::env::var("K8V4_DECODE_CONTEXT")
        .ok()
        .map(|value| value.parse::<usize>().unwrap());
    let selected_params = std::env::var("K8V4_DECODE_CONFIG_FILTER")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|part| {
                    let (name, value) = part
                        .split_once('=')
                        .expect("expected NAME=VALUE in K8V4_DECODE_CONFIG_FILTER");
                    (name.to_owned(), value.parse::<u64>().unwrap())
                })
                .collect::<Vec<_>>()
        });
    let contexts = selected_context.map_or(contexts, |context| vec![context]);
    for (geometry, context) in geometries.into_iter().flat_map(|geometry| {
        contexts
            .clone()
            .into_iter()
            .map(move |context| (geometry, context))
    }) {
        // `K8V4_DECODE_ROWS=M` times a verification step of M rows: row j sees
        // the history and the batch's first j + 1 rows.
        let verified =
            std::env::var("K8V4_DECODE_ROWS").map_or(1, |rows| rows.parse::<i32>().unwrap());
        let rows = (0..verified)
            .map(|j| Row {
                spans: vec![(0, context as i32 - 1)],
                fresh: (0, j + 1),
                destination: context as i32 - 1 + j,
                position: context as i32 - 1 + j,
            })
            .collect::<Vec<_>>();
        let encoded = Encoded::new(Case::new(geometry, context + 64, 1, &rows, 3));
        let mut dense = DenseHistory::new(device, &encoded.case);
        let configs = if sweep {
            timing_decode_configs(backend, geometry)
        } else {
            decode_configs(backend, geometry)
        };
        for config in configs {
            if selected_params.as_ref().is_some_and(|params| {
                params.iter().any(|(name, value)| {
                    config
                        .iter()
                        .find(|(key, _)| *key == name)
                        .is_none_or(|(_, actual)| actual != value)
                })
            }) {
                continue;
            }
            // A sweep configuration the declaration's `where` refuses at this
            // geometry is skipped; any other failure is a broken sweep.
            let kernel = match attention_decode_k8v4::native_for_device_with(
                device,
                attention_decode_k8v4::Elements { A: Element::bf16() },
                &specialization_on(device, geometry, &config),
            ) {
                Ok(kernel) => kernel,
                Err(error)
                    if error
                        .to_string()
                        .contains("violates the native `where` condition") =>
                {
                    continue
                }
                Err(error) => panic!("{backend:?} decode {config:?}: {error}"),
            };
            let mut bound = Bound::new(device, &encoded);
            let affine = kernel
                .measure(
                    vec![args!(attention_decode_k8v4, bound, encoded.case)],
                    &TIMING,
                )
                .unwrap()
                .median;
            let affine_bytes = (context
                * geometry.kv
                * (geometry.w() + geometry.w() / 2 + 4 * coefficient_elements(geometry.w())))
                as f64;
            let dense_bytes = (context * geometry.kv * geometry.w() * 4) as f64;
            // The dense entry's domain may not admit this configuration. It has
            // no keywise walk: Vulkan's configurations time it once, beside the
            // per-key walk.
            let dense_config = match config.iter().find(|(name, _)| *name == "KEYWISE") {
                Some(&(_, 0)) => Some(
                    config
                        .iter()
                        .copied()
                        .filter(|(name, _)| *name != "KEYWISE")
                        .collect::<Vec<_>>(),
                ),
                Some(_) => None,
                None => Some(config.clone()),
            };
            let dense_time = dense_config
                .and_then(|dense_config| {
                    attention_decode::native_for_device_with(
                        device,
                        attention_decode::Elements { A: Element::bf16() },
                        &specialization_on(device, geometry, &dense_config),
                    )
                    .ok()
                })
                .map(|dense_kernel| {
                    dense_kernel
                        .measure(
                            vec![dense_args!(attention_decode, bound, dense, encoded.case)],
                            &TIMING,
                        )
                        .unwrap()
                        .median
                });
            let dense_report = dense_time.map_or("dense not timed".to_owned(), |dense_time| {
                format!(
                    "dense {:.1} us ({:.0} GB/s), k8v4/dense {:.2}",
                    dense_time * 1e6,
                    dense_bytes / dense_time / 1e9,
                    affine / dense_time
                )
            });
            eprintln!(
                "{backend:?} decode group {} width {} context {context} {config:?}: k8v4 {:.1} us ({:.0} GB/s), {dense_report}",
                geometry.g,
                geometry.w(),
                affine * 1e6,
                affine_bytes / affine / 1e9,
            );
        }
    }
}

/// Device time of one prefill layer's attention at Qwen3.5-4B geometry over
/// affine history, next to the dense entry on the same inputs
/// (`--ignored --nocapture prefill_timing`). FLOP counts both products of
/// every visible (row, key) pair.
#[test]
#[ignore]
fn prefill_timing() {
    for device in k8v4_devices() {
        prefill_timing_on(&device);
    }
}

fn prefill_timing_on(device: &Device) {
    let backend = device.backend();
    let rows_history = match std::env::var("K8V4_PREFILL_SHAPES") {
        Ok(shapes) => shapes
            .split(',')
            .map(|shape| {
                let (rows, history) = shape.split_once('x').expect("shape is ROWSxHISTORY");
                (
                    rows.parse::<usize>().unwrap(),
                    history.parse::<i32>().unwrap(),
                )
            })
            .collect::<Vec<_>>(),
        Err(_) => vec![
            (128usize, 0i32),
            (512, 0),
            (512, 4096),
            (512, 16384),
            (512, 65536),
        ],
    };
    // `K8V4_PREFILL_GEOMETRY=minicpm5` times MiniCPM5-2B's heads (2 kv heads
    // of 8, W = 128), `gemma31`, `gemma26` and `gemma-e4b` those Gemma 4
    // models' full layers (W = 512), instead of Qwen3.5-4B's.
    let geometry = match std::env::var("K8V4_PREFILL_GEOMETRY").as_deref() {
        Ok("minicpm5") => MINICPM5,
        Ok("gemma31") => GEMMA31_FULL,
        Ok("gemma26") => GEMMA26_FULL,
        Ok("gemma-e4b") => GEMMA_E4B_FULL,
        _ => QWEN,
    };
    for (rows, history) in rows_history {
        let spans = if history > 0 {
            vec![(0, history)]
        } else {
            vec![]
        };
        let rows = (0..rows)
            .map(|row| Row {
                spans: spans.clone(),
                fresh: (0, row as i32 + 1),
                destination: history + row as i32,
                position: history + row as i32,
            })
            .collect::<Vec<_>>();
        let pairs = rows
            .iter()
            .map(|row| (history + row.fresh.1 - row.fresh.0) as f64)
            .sum::<f64>();
        let flop = pairs * (geometry.kv * geometry.g * geometry.w() * 4) as f64;
        let encoded = Encoded::new(Case::new(
            geometry,
            history as usize + rows.len(),
            1,
            &rows,
            9,
        ));
        let mut dense = DenseHistory::new(device, &encoded.case);
        let forms = if backend == BackendName::Metal {
            coissue_prefill_configs(geometry)
        } else {
            Vec::new()
        };
        // `K8V4_PREFILL_ONLY=NAME=VALUE[,NAME=VALUE...]` times only the
        // configurations with those parameter values (a parameter a
        // configuration does not name is 0).
        let only = std::env::var("K8V4_PREFILL_ONLY").map_or_else(
            |_| Vec::new(),
            |only| {
                only.split(',')
                    .map(|item| {
                        let (name, value) = item
                            .split_once('=')
                            .expect("expected NAME=VALUE in K8V4_PREFILL_ONLY");
                        (name.to_string(), value.parse::<u64>().unwrap())
                    })
                    .collect::<Vec<_>>()
            },
        );
        for config in prefill_configs(backend, geometry).into_iter().chain(forms) {
            let value_of = |name: &str| {
                config
                    .iter()
                    .find(|(param, _)| *param == name)
                    .map_or(0, |(_, value)| *value)
            };
            if only.iter().any(|(name, value)| value_of(name) != *value) {
                continue;
            }
            let kernel = prefill_kernel(device, geometry, &config);
            let mut bound = Bound::new(device, &encoded);
            let affine = kernel
                .measure(
                    vec![prefill_args!(bound, encoded.case)],
                    &TIMING,
                )
                .unwrap()
                .median;
            let dense_kernel = attention_prefill::native_for_device_with(
                device,
                attention_prefill::Elements { A: Element::bf16() },
                // The dense entry has no producer warps and no co-issue
                // form; its direct form takes 16-row query tiles, so an
                // 8-row configuration is timed staged.
                &specialization_on(
                    device,
                    geometry,
                    &config
                        .iter()
                        .copied()
                        .filter(|(name, _)| !["PRODUCERS", "COISSUE"].contains(name))
                        .map(|(name, value)| match name {
                            "DIRECT" if value_of("QT") == 8 => (name, 0),
                            _ => (name, value),
                        })
                        .collect::<Vec<_>>(),
                ),
            )
            .unwrap();
            let dense_time = dense_kernel
                .measure(
                    vec![dense_args!(attention_prefill, bound, dense, encoded.case)],
                    &TIMING,
                )
                .unwrap()
                .median;
            eprintln!(
                "{backend:?} prefill {} rows after {history} history {config:?}: k8v4 {:.1} us ({:.2} TFLOP/s), dense {:.1} us ({:.2} TFLOP/s), k8v4/dense {:.2}",
                rows.len(),
                affine * 1e6,
                flop / affine / 1e12,
                dense_time * 1e6,
                flop / dense_time / 1e12,
                affine / dense_time
            );
        }
    }
}

/// The incident's head geometry and four controlled configurations, using
/// synthetic Qwen-style inputs. This checks geometry coverage; it does not
/// reproduce or explain the historical Gemma output corruption.
#[test]
fn gemma_head_geometry_matches_synthetic_reference() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Metal)
        .unwrap();
    let geometry = Geometry {
        kv: 2,
        g: 8,
        p: 256,
        s: 0,
    };
    let rows = [Row {
        spans: vec![(3, 137), (149, 251)],
        fresh: (0, 1),
        destination: 256,
        position: 251,
    }];
    let encoded = Encoded::new(Case::new(geometry, 272, 2, &rows, 73));
    let expected = encoded.expected();
    for (simds, slices) in [(8, 4), (4, 1), (8, 1), (4, 4)] {
        let configuration = [
            ("PARTS", 16),
            ("SPAN", 32),
            ("SIMDS", simds),
            ("SLICES", slices),
            ("MATRIX", 0),
            ("KEYS", 16),
            ("TOKENS", 1),
        ];
        let kernel = decode_kernel(&device, geometry, &configuration);
        let mut bound = Bound::new(&device, &encoded);
        let output = kernel
            .call(args!(attention_decode_k8v4, bound, encoded.case))
            .unwrap()
            .value;
        check(
            &format!("Gemma G8 {configuration:?}"),
            &encoded,
            &output,
            &bound,
            &expected,
        );
    }
}

#[test]
#[ignore = "manual reference-preparation diagnostic: currently takes minutes in initialization-region construction"]
fn gemma_g8_tuning_uses_portable_reference() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Metal)
        .unwrap();
    let geometry = Geometry {
        kv: 2,
        g: 8,
        p: 256,
        s: 0,
    };
    let rows = [Row {
        spans: vec![(3, 17), (19, 25)],
        fresh: (0, 1),
        destination: 26,
        position: 25,
    }];
    let encoded = Encoded::new(Case::new(geometry, 32, 2, &rows, 73));
    let mut bound = Bound::new(&device, &encoded);
    let mut pristine = [
        &bound.key_codes,
        &bound.key_coefficients,
        &bound.value_codes,
        &bound.value_coefficients,
    ]
    .into_iter()
    .map(|tensor| (tensor.clone(), tensor.read_to_host().unwrap()))
    .collect::<Vec<_>>();
    let initialize: seismic::TuningInitializer<'_> = Box::new(move || {
        for (tensor, bytes) in &mut pristine {
            tensor.write_from_host(bytes)?;
        }
        Ok(())
    });
    let result = attention_decode_k8v4::native_tune_with(
        &device,
        attention_decode_k8v4::Elements { A: Element::bf16() },
        &specialization_on(&device, geometry, &[]),
        vec![seismic::TuningPoint {
            label: "gemma-g8".into(),
            weight: 1.,
            class: None,
            cost: 1.0,
            required: false,
            rotation: vec![args!(attention_decode_k8v4, bound, encoded.case)],
            initialize: Some(initialize),
            written: Default::default(),
        }],
        seismic::PrecisionPolicy::bounded(seismic::precision::Tolerance {
            absolute: seismic::precision::Limit::new(0.01).unwrap(),
            relative: seismic::precision::Limit::new(0.01).unwrap(),
            relative_floor: seismic::precision::Limit::ZERO,
            ulps: None,
        }),
        seismic::Strategy::Search(seismic::SearchPlan {
            allowance: std::time::Duration::from_secs(600),
            admission: std::time::Duration::from_secs(600),
            required: std::time::Duration::from_secs(600),
            settings: seismic::SearchSettings {
                improvement: 0.01,
                restarts: 2,
                confirmed: 3,
                default_margin: 0.02,
                samples: 1,
                confirmation_samples: 3,
            },
            min_sample_seconds: 0.0002,
            start: vec![[
                ("PARTS".into(), 16),
                ("SIMDS".into(), 8),
                ("SLICES".into(), 4),
                ("SPAN".into(), 32),
            ]
            .into()],
        }),
        seismic::TuningReference::Portable,
    );
    eprintln!("Gemma portable-reference tuning: {result:?}");
    assert!(result.is_ok());
}

/// Actual Gemma form from the incident cache: ungated, normalized values,
/// BF16, KV2/G8/P256. Compare the incident choices on short and long history.
#[test]
#[ignore = "manual diagnostic: the historical Metal configuration is numerically defective without shader instrumentation"]
fn gemma_form_decode_configurations_agree() {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Metal)
        .unwrap();
    let geometry = Geometry {
        kv: 2,
        g: 8,
        p: 256,
        s: 0,
    };
    let mut failures = 0;
    for history in [0, 1, 31, 32, 127, 256, 2048] {
        let rows = [Row {
            spans: vec![(0, history)],
            fresh: (0, 1),
            destination: history,
            position: history,
        }];
        let encoded = Encoded::new(Case::new(geometry, history as usize + 1, 1, &rows, 73));
        let mut baseline: Option<(Vec<f32>, Vec<Vec<u8>>)> = None;
        for (simds, slices) in [(4, 1), (8, 4), (8, 1), (4, 4)] {
            let mut bound = Bound::new(&device, &encoded);
            let w = geometry.w();
            let query: Vec<_> = encoded
                .case
                .query_gate
                .chunks_exact(2 * w)
                .flat_map(|head| head[..w].iter().copied())
                .collect();
            bound.query = bf16_tensor(&device, &[1, geometry.kv * geometry.g, w], &query);
            bound.value_norm = f32_tensor(&device, &[1, w], &vec![1.; w]);
            bound.components = i32_tensor(&device, &[geometry.p], &vec![0; geometry.p]);
            let spec = specialization_on(
                &device,
                geometry,
                &[
                    ("PARTS", 16),
                    ("SPAN", 32),
                    ("SIMDS", simds),
                    ("SLICES", slices),
                ],
            )
            .with_static("I", 0)
            .with_static("NV", 1);
            let kernel = attention_decode_k8v4::native_for_device_with(
                &device,
                attention_decode_k8v4::Elements { A: Element::bf16() },
                &spec,
            )
            .unwrap();
            let out = kernel
                .call(args!(attention_decode_k8v4, bound, encoded.case))
                .unwrap()
                .value;
            let actual = bf16_values(&out);
            eprintln!(
                "Gemma history={history} SIMDS={simds} SLICES={slices}: first {:?}",
                &actual[..4]
            );
            let state = [
                &bound.key_codes,
                &bound.key_coefficients,
                &bound.value_codes,
                &bound.value_coefficients,
            ]
            .into_iter()
            .map(|t| t.read_to_host().unwrap())
            .collect::<Vec<_>>();
            if let Some((reference, expected_state)) = &baseline {
                assert_eq!(
                    &state, expected_state,
                    "history {history} simds {simds} slices {slices}: writable state"
                );
                let mut worst = 0f32;
                let mut required_scale = 0f32;
                for (i, (&a, &r)) in actual.iter().zip(reference).enumerate() {
                    worst = worst.max((a - r).abs());
                    required_scale = required_scale.max((a - r).abs() / (0.01 + 0.01 * r.abs()));
                    if !(a.is_finite() && (a - r).abs() <= 0.01 + 0.01 * r.abs()) {
                        failures += 1;
                        if i == 0 {
                            eprintln!(
                                "FAIL history {history} simds {simds} slices {slices}: {a} != {r}"
                            );
                        }
                    }
                }
                eprintln!("Gemma history={history} SIMDS={simds} SLICES={slices}: max_abs={worst} required_scale={required_scale}");
            } else {
                baseline = Some((actual, state));
            }
        }
    }
    assert_eq!(failures, 0, "Gemma output elements outside bounded policy");
}

/// A single fresh key has a simple independent result: its normalized value,
/// repeated for every query head. This catches the actual Gemma form defect
/// without assuming the default native implementation is mathematically correct.
#[test]
fn gemma_tuner_rejects_corrupted_fresh_only_attention() {
    for scale in [0.25, 1., 4.] {
        gemma_fresh_only_with_scale(scale);
    }
}

fn gemma_fresh_only_with_scale(scale: f64) {
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Metal)
        .unwrap();
    let geometry = Geometry {
        kv: 2,
        g: 8,
        p: 256,
        s: 0,
    };
    let w = geometry.w();
    let rows = [Row {
        spans: vec![(0, 0)],
        fresh: (0, 1),
        destination: 0,
        position: 0,
    }];
    let encoded = Encoded::new(Case::new(geometry, 1, 1, &rows, 73));
    let mut bound = Bound::new(&device, &encoded);
    let query: Vec<_> = encoded
        .case
        .query_gate
        .chunks_exact(2 * w)
        .flat_map(|head| head[..w].iter().copied())
        .collect();
    bound.query = bf16_tensor(&device, &[1, geometry.kv * geometry.g, w], &query);
    bound.value_norm = f32_tensor(&device, &[1, w], &vec![1.; w]);
    bound.components = i32_tensor(&device, &[geometry.p], &vec![0; geometry.p]);
    let spec = specialization_on(&device, geometry, &[])
        .with_static("I", 0)
        .with_static("NV", 1);
    let mut pristine = [
        &bound.key_codes,
        &bound.key_coefficients,
        &bound.value_codes,
        &bound.value_coefficients,
    ]
    .into_iter()
    .map(|tensor| (tensor.clone(), tensor.read_to_host().unwrap()))
    .collect::<Vec<_>>();
    let result = attention_decode_k8v4::native_tune_with(
        &device,
        attention_decode_k8v4::Elements { A: Element::bf16() },
        &spec,
        vec![seismic::TuningPoint {
            label: "gemma-fresh-only".into(),
            weight: 1.,
            class: None,
            cost: 1.0,
            required: false,
            rotation: vec![args!(attention_decode_k8v4, bound, encoded.case)],
            initialize: Some(Box::new(move || {
                for (tensor, bytes) in &mut pristine {
                    tensor.write_from_host(bytes)?;
                }
                Ok(())
            })),
            written: Default::default(),
        }],
        seismic::PrecisionPolicy::bounded(seismic::Tolerance {
            absolute: seismic::Limit::new(0.01 * scale).unwrap(),
            relative: seismic::Limit::new(0.01 * scale).unwrap(),
            relative_floor: seismic::Limit::ZERO,
            ulps: None,
        }),
        seismic::Strategy::Survey(seismic::SurveyPlan {
            samples: 2,
            min_sample_seconds: 0.0002,
            domains: [
                ("PARTS", vec![16]),
                ("SPAN", vec![32]),
                ("SIMDS", vec![4, 8]),
                ("SLICES", vec![1, 4]),
            ]
            .into_iter()
            .map(|(k, v)| (k.into(), v))
            .collect(),
        }),
        seismic::TuningReference::NativeDefault,
    )
    .unwrap();
    let mut expected = Vec::new();
    for value in encoded.case.value.chunks_exact(w) {
        let squares = value.iter().map(|v| v * v).sum::<f32>();
        let inverse = 1. / (squares / w as f32 + encoded.case.epsilon).sqrt();
        for _ in 0..geometry.g {
            expected.extend(value.iter().map(|v| bf16(v * inverse)));
        }
    }
    for record in &result.configurations {
        if let seismic::Outcome::Measured { .. } = &record.outcome {
            let kernel = attention_decode_k8v4::native_for_device_with(
                &device,
                attention_decode_k8v4::Elements { A: Element::bf16() },
                &record.configuration.specialization(),
            )
            .unwrap();
            let actual = bf16_values(
                &kernel
                    .call(args!(attention_decode_k8v4, bound, encoded.case))
                    .unwrap()
                    .value,
            );
            for (index, (&a, &r)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    a.is_finite() && (a - r).abs() <= 0.01 + 0.01 * r.abs(),
                    "admitted {:?} element {index}: {a} vs independent {r}",
                    record.configuration.params
                );
            }
        }
    }
    let historical = result
        .configurations
        .iter()
        .find(|r| r.configuration.params["SIMDS"] == 8 && r.configuration.params["SLICES"] == 4)
        .unwrap();
    eprintln!(
        "Historical Gemma configuration at precision scale {scale}: {:?}",
        historical.outcome
    );
    assert!(result
        .configurations
        .iter()
        .any(|r| matches!(r.outcome, seismic::Outcome::Measured { .. })));
}


/// State-only publication agrees with the existing fused codec.
#[test]
fn ordered_append_matches_fused_codec() {
    for device in k8v4_devices() {
        for geometry in [
            GROUPED,
            QWEN,
            Geometry {
                kv: 1,
                g: 1,
                p: 128,
                s: 256,
            },
        ] {
            let encoded = Encoded::new(Case::new(geometry, 64, 2, &decode_rows(16), 731));
            let config = decode_configs(device.backend(), geometry).remove(0);
            let decode = decode_kernel(&device, geometry, &config);
            let append = attention_append_k8v4::native_for_device_with(
                &device,
                attention_append_k8v4::Elements { A: Element::bf16() },
                &append_specialization(&specialization_on(&device, geometry, &[])),
            )
            .unwrap();
            let mut fused = Bound::new(&device, &encoded);
            let mut ordered = Bound::new(&device, &encoded);
            decode
                .call(args!(attention_decode_k8v4, fused, encoded.case))
                .unwrap();
            append
                .call(args!(attention_append_k8v4, ordered, encoded.case))
                .unwrap();
            assert_eq!(
                ordered.planes(),
                fused.planes(),
                "{:?} ordered publication differs from fused codec",
                device.backend()
            );
        }
    }
}

fn append_specialization(specialization: &NativeSpecialization) -> NativeSpecialization {
    specialization
        .statics()
        .iter()
        .filter(|(name, _)| !matches!(name.as_str(), "G" | "I" | "U"))
        .fold(NativeSpecialization::new(), |spec, (name, value)| {
            spec.with_static(name, *value)
        })
}
