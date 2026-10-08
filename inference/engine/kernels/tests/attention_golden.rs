// Bit-exact golden of the attention entries over slab-stored history, on one
// backend: every decode and prefill entry (dense and affine K8/V4 history) in
// Qwen's form at the Qwen3.5-4B geometry and a small one, a few tuning
// configurations each, rows whose visible spans cross slab edges. `record`
// writes a hash of every result and history plane (running each call twice to
// reject nondeterminism); `compare` requires identical hashes for the same
// keys. The goldens on the lane hosts were recorded from the Qwen-only
// entries this family replaced, so a compare pins Qwen bit for bit.
//
// The kernel `golden` harness binds history as plain tensors through the
// dynamic API, which cannot express slab-stored history; this test binds real
// slab tensors through the typed entries instead.
//
// ```text
// ATTENTION_GOLDEN_MODE=record|compare ATTENTION_GOLDEN_DIR=<dir> \
//   ATTENTION_GOLDEN_BACKEND=metal|cuda|vulkan|cpu \
//   cargo test --release -p magnitude-kernels --test attention_golden -- --ignored --nocapture
// ```
#![allow(dead_code)]

use magnitude_kernels::{
    attention_decode, attention_decode_k8v4, attention_prefill, attention_prefill_k8v4,
};
use seismic::{
    BackendName, Device, DeviceCatalog, Element, NativeSpecialization, SlabRegion, SlabTensor,
    Tensor,
};
use seismic_lang::registry::{f16_bits, f16_to_f32};
use std::collections::BTreeMap;
use std::fmt::Write as _;

include!("attention_common/fixtures.rs");

/// A 128-bit content hash (the kernel golden harness's).
fn hash(bytes: &[u8]) -> String {
    let mix = |mut x: u64| {
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51afd7ed558ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
        x ^ (x >> 33)
    };
    let (mut a, mut b) = (0x243f_6a88_85a3_08d3u64, 0x1319_8a2e_0370_7344u64);
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().unwrap());
        a = mix(a ^ word).wrapping_add(0x9e37_79b9_7f4a_7c15);
        b = mix(b.rotate_left(29) ^ word.wrapping_mul(0x2545_f491_4f6c_dd1d));
    }
    let mut tail = [0u8; 8];
    tail[..chunks.remainder().len()].copy_from_slice(chunks.remainder());
    let word = u64::from_le_bytes(tail);
    a = mix(a ^ word ^ bytes.len() as u64);
    b = mix(b ^ word.rotate_left(17) ^ (bytes.len() as u64).wrapping_mul(3));
    format!("{a:016x}{b:016x}")
}

const KEY_BITS: u32 = 8;
const VALUE_BITS: u32 = 4;
const GROUP: usize = 32;

/// One vector's affine encoding (the portable `affine_encode`): code words
/// and (scale, zero) F16 bits per group.
fn encode(x: &[f32], bits: u32) -> (Vec<u32>, Vec<u16>) {
    let levels = (1u32 << bits) - 1;
    let per = (32 / bits) as usize;
    let mut words = vec![0u32; x.len() / per];
    let mut coefficients = Vec::new();
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Decode,
    Prefill,
    DecodeK8V4,
    PrefillK8V4,
}

impl Kind {
    fn affine(self) -> bool {
        matches!(self, Kind::DecodeK8V4 | Kind::PrefillK8V4)
    }
    fn decode(self) -> bool {
        matches!(self, Kind::Decode | Kind::DecodeK8V4)
    }
    fn name(self) -> &'static str {
        match self {
            Kind::Decode => "decode",
            Kind::Prefill => "prefill",
            Kind::DecodeK8V4 => "decode_k8v4",
            Kind::PrefillK8V4 => "prefill_k8v4",
        }
    }
}

/// History planes of one case: (element, per-row shape [KV, width], bytes of
/// every row).
struct Plane {
    element: Element,
    width: usize,
    bytes: Vec<u8>,
}

fn planes(case: &Case, affine: bool) -> Vec<Plane> {
    let w = case.geometry.w();
    let bf16_bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| bf16_bits(*v).to_le_bytes())
            .collect::<Vec<_>>()
    };
    if !affine {
        return vec![
            Plane {
                element: Element::bf16(),
                width: w,
                bytes: bf16_bytes(&case.history_key),
            },
            Plane {
                element: Element::bf16(),
                width: w,
                bytes: bf16_bytes(&case.history_value),
            },
        ];
    }
    let (mut kc, mut kf, mut vc, mut vf) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (key, value) in case
        .history_key
        .chunks_exact(w)
        .zip(case.history_value.chunks_exact(w))
    {
        let (codes, coefficients) = encode(key, KEY_BITS);
        kc.extend(codes.iter().flat_map(|c| c.to_le_bytes()));
        kf.extend(coefficients.iter().flat_map(|c| c.to_le_bytes()));
        let (codes, coefficients) = encode(value, VALUE_BITS);
        vc.extend(codes.iter().flat_map(|c| c.to_le_bytes()));
        vf.extend(coefficients.iter().flat_map(|c| c.to_le_bytes()));
    }
    vec![
        Plane {
            element: Element::u32(),
            width: w / 4,
            bytes: kc,
        },
        Plane {
            element: Element::f16(),
            width: w / 16,
            bytes: kf,
        },
        Plane {
            element: Element::u32(),
            width: w / 8,
            bytes: vc,
        },
        Plane {
            element: Element::f16(),
            width: w / 16,
            bytes: vf,
        },
    ]
}

/// The history planes in one slab tensor of `slab_rows` rows per slab, and
/// each plane's logical region.
fn history(
    device: &Device,
    case: &Case,
    affine: bool,
    slab_rows: usize,
) -> (SlabTensor, Vec<Tensor>) {
    let kv = case.geometry.kv;
    let rows = case.history_rows;
    let planes = planes(case, affine);
    let mut slabs = SlabTensor::new(
        device,
        slab_rows as u64,
        rows as u64,
        planes
            .iter()
            .map(|plane| SlabRegion {
                element: plane.element,
                row_shape: vec![kv as u64, plane.width as u64],
            })
            .collect(),
    )
    .unwrap();
    for index in 0..rows.div_ceil(slab_rows) {
        slabs.add_slab().unwrap();
        let start = index * slab_rows;
        let count = slab_rows.min(rows - start);
        for (region, plane) in planes.iter().enumerate() {
            let row_bytes = plane.bytes.len() / rows;
            slabs
                .region_rows(region, start as u64, count as u64)
                .unwrap()
                .write_from_host(&plane.bytes[start * row_bytes..(start + count) * row_bytes])
                .unwrap();
        }
    }
    let logical = (0..planes.len())
        .map(|region| slabs.logical_region(region).unwrap())
        .collect();
    (slabs, logical)
}

/// Splits each logical span at slab edges: a history span never crosses one.
fn slab_rows_spans(rows: Vec<Row>, slab_rows: i32) -> Vec<Row> {
    rows.into_iter()
        .map(|row| Row {
            spans: row
                .spans
                .iter()
                .flat_map(|&(lo, hi)| {
                    let mut pieces = Vec::new();
                    let mut at = lo;
                    while at < hi {
                        let end = hi.min((at / slab_rows + 1) * slab_rows);
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

/// One named case: geometry, rows and slab size.
struct Named {
    label: String,
    case: Case,
    slab_rows: usize,
}

fn named(
    label: &str,
    geometry: Geometry,
    history_rows: usize,
    rows: Vec<Row>,
    slab_rows: usize,
    seed: u64,
) -> Named {
    let rows = slab_rows_spans(rows, slab_rows as i32);
    let spans = rows.iter().map(|row| row.spans.len()).max().unwrap().max(1);
    let history_rows = history_rows.div_ceil(slab_rows) * slab_rows;
    Named {
        label: label.into(),
        case: Case::new(geometry, history_rows, spans, &rows, seed),
        slab_rows,
    }
}

fn cases(kind: Kind, geometry: Geometry) -> Vec<Named> {
    if kind.decode() {
        vec![
            named(
                "spec1@256",
                geometry,
                256 + 64,
                speculative_rows(1, 256),
                512,
                11,
            ),
            named(
                "spec8@4096",
                geometry,
                4096 + 64,
                speculative_rows(8, 4096),
                1024,
                12,
            ),
            named("mixed4@300", geometry, 300 + 128, decode_rows(300), 128, 13),
            named("spec2@1", geometry, 64, speculative_rows(2, 1), 64, 14),
        ]
    } else {
        vec![
            named("m16@0", geometry, 256, prefill_rows(16, 0), 256, 20),
            named(
                "m40@300",
                geometry,
                300 + 256,
                prefill_rows(40, 300),
                128,
                21,
            ),
            named(
                "m128@1000",
                geometry,
                1000 + 256,
                prefill_rows(128, 1000),
                512,
                22,
            ),
        ]
    }
}

/// Tuning configurations per backend: the declared defaults and a few
/// variations, each admissible at both geometries.
fn configurations(backend: BackendName, kind: Kind) -> Vec<Vec<(&'static str, u64)>> {
    match (backend, kind.decode()) {
        (BackendName::Cpu, true) => {
            vec![vec![("PARTS", 8)], vec![("PARTS", 1)], vec![("PARTS", 16)]]
        }
        (BackendName::Cpu, false) => vec![vec![]],
        (BackendName::Metal, true) => vec![
            vec![("SPAN", 32), ("PARTS", 16), ("SIMDS", 4)],
            vec![("SPAN", 64), ("PARTS", 8), ("SIMDS", 8)],
            vec![("SPAN", 256), ("PARTS", 32), ("SIMDS", 4)],
        ],
        (BackendName::Metal, false) => vec![
            vec![("QT", 16), ("SPLIT_GROUPS", 256)],
            vec![("QT", 8), ("SPLIT_GROUPS", 1)],
            vec![("QT", 16), ("SPLIT_GROUPS", 512)],
        ],
        (BackendName::Cuda, true) => vec![
            vec![("PARTS", 12), ("WARPS", 4)],
            vec![("PARTS", 48), ("WARPS", 8)],
            vec![("PARTS", 24), ("WARPS", 4)],
        ],
        (BackendName::Cuda, false) => vec![vec![("WARPS", 4)], vec![("WARPS", 2)]],
        (BackendName::Vulkan, true) => vec![
            vec![("SPAN", 32), ("PARTS", 16), ("SIMDS", 4), ("SLICES", 1)],
            vec![("SPAN", 64), ("PARTS", 8), ("SIMDS", 8), ("SLICES", 2)],
            vec![("SPAN", 128), ("PARTS", 32), ("SIMDS", 4), ("SLICES", 1)],
        ],
        (BackendName::Vulkan, false) => vec![
            vec![("ROWS", 64), ("SPLIT_GROUPS", 256)],
            vec![("ROWS", 64), ("SPLIT_GROUPS", 1)],
            vec![("ROWS", 128), ("SPLIT_GROUPS", 512)],
        ],
        (other, _) => panic!("no attention configurations for {other:?}"),
    }
}

/// Device inputs of one case, shared by every entry form.
struct Inputs {
    query_gate: Tensor,
    key: Tensor,
    value: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
    components: Tensor,
    frequencies: Tensor,
    coordinates: Tensor,
    visible: Tensor,
    fresh: Tensor,
    destinations: Tensor,
    history_tiles: Tensor,
}

impl Inputs {
    fn new(device: &Device, case: &Case) -> Self {
        let Geometry { kv, g, p, .. } = case.geometry;
        let (m, w) = (case.rows, case.geometry.w());
        Self {
            query_gate: bf16_tensor(device, &[m, kv * g * 2 * w], &case.query_gate),
            key: bf16_tensor(device, &[m, kv * w], &case.key),
            value: bf16_tensor(device, &[m, kv * w], &case.value),
            query_norm: f32_tensor(device, &[w], &case.query_norm),
            key_norm: f32_tensor(device, &[w], &case.key_norm),
            components: i32_tensor(device, &[p], &case.components),
            frequencies: f32_tensor(device, &[p], &case.frequencies),
            coordinates: i32_tensor(device, &[m, 4], &case.coordinates),
            visible: i32_tensor(device, &[m, case.spans, 2], &case.visible),
            fresh: i32_tensor(device, &[m, 2], &case.fresh),
            destinations: i32_tensor(device, &[m], &case.destinations),
            history_tiles: {
                let tiles = history_tiles(&case.visible);
                i32_tensor(device, &[1, tiles.len()], &tiles)
            },
        }
    }
}

/// The Qwen form of the `attention_*` family over the same inputs: queries
/// with interleaved gates (I = W), no separate gate, fresh rows, q/k norms,
/// no value norm, unit rotary amplitudes, sigmoid gates.
struct Family {
    query: Tensor,
    gate: Tensor,
    key: Tensor,
    value: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
    value_norm: Tensor,
    amplitudes: Tensor,
}

impl Family {
    fn new(device: &Device, inputs: &Inputs, case: &Case) -> Self {
        let Geometry { kv, g, p, .. } = case.geometry;
        let (m, w) = (case.rows as u64, case.geometry.w() as u64);
        let heads = (kv * g) as u64;
        let empty = |element: Element, shape: &[u64]| {
            Tensor::from_host(device, element, shape, &[]).unwrap()
        };
        Self {
            query: inputs.query_gate.reshape(&[m, heads, 2 * w]).unwrap(),
            gate: empty(Element::bf16(), &[m, heads, 0]),
            key: inputs.key.reshape(&[1, m, kv as u64 * w]).unwrap(),
            value: inputs.value.reshape(&[1, m, kv as u64 * w]).unwrap(),
            query_norm: inputs.query_norm.reshape(&[1, w]).unwrap(),
            key_norm: inputs.key_norm.reshape(&[1, w]).unwrap(),
            value_norm: empty(Element::f32(), &[0, w]),
            amplitudes: f32_tensor(device, &[p], &vec![1.0; p]),
        }
    }

    /// The family entries' statics of the Qwen form (CPU: only those its
    /// declarations fix).
    fn statics(device: &Device, geometry: Geometry) -> NativeSpecialization {
        let w = geometry.w() as u64;
        let form = [("I", w), ("U", 0), ("F", 1), ("N", 1), ("NV", 0)];
        let base = if device.backend() == BackendName::Cpu {
            NativeSpecialization::new()
                .with_static("P", geometry.p as u64)
                .with_static("S", geometry.s as u64)
        } else {
            statics(geometry)
        };
        form.into_iter()
            .fold(base, |s, (name, value)| s.with_static(name, value))
    }
}

/// Runs the `attention_*` entry of `kind` in the Qwen form; returns the
/// result and history planes as bytes.
fn run_family(
    device: &Device,
    kind: Kind,
    specialization: &NativeSpecialization,
    inputs: &Inputs,
    family: &Family,
    history: &mut [Tensor],
    named: &Named,
) -> Result<Vec<(String, Vec<u8>)>, String> {
    let case = &named.case;
    let slab_rows = named.slab_rows as u32;
    let elements = Element::bf16();
    {
        macro_rules! dense {
            ($module:ident) => {{
                let kernel = $module::native_for_device_with(
                    device,
                    $module::Elements { A: elements },
                    specialization,
                )
                .map_err(|error| format!("prepare {}: {error}", stringify!($module)))?;
                let [history_key, history_value] = &mut *history else {
                    panic!("dense history has two planes")
                };
                kernel
                    .call($module::Args {
                        query: &family.query,
                        gate: &family.gate,
                        key: &family.key,
                        value: &family.value,
                        query_norm: &family.query_norm,
                        key_norm: &family.key_norm,
                        value_norm: &family.value_norm,
                        rotary_components: &inputs.components,
                        rotary_frequencies: &inputs.frequencies,
                        rotary_amplitudes: &family.amplitudes,
                        coordinates: &inputs.coordinates,
                        visible: &inputs.visible,
                        fresh: &inputs.fresh,
                        destinations: &inputs.destinations,
                        history_key,
                        history_value,
                        epsilon: case.epsilon,
                        scale: case.scale,
                        gate_function: 0,
                        slab_rows,
                    })
                    .unwrap()
                    .value
            }};
        }
        macro_rules! affine {
            ($module:ident $(, $extra:ident: $value:expr)?) => {{
                let kernel = $module::native_for_device_with(
                    device,
                    $module::Elements { A: elements },
                    specialization,
                )
                .map_err(|error| format!("prepare {}: {error}", stringify!($module)))?;
                let [key_codes, key_coefficients, value_codes, value_coefficients] = &mut *history
                else {
                    panic!("affine history has four planes")
                };
                kernel
                    .call($module::Args {
                        query: &family.query,
                        gate: &family.gate,
                        key: &family.key,
                        value: &family.value,
                        query_norm: &family.query_norm,
                        key_norm: &family.key_norm,
                        value_norm: &family.value_norm,
                        rotary_components: &inputs.components,
                        rotary_frequencies: &inputs.frequencies,
                        rotary_amplitudes: &family.amplitudes,
                        coordinates: &inputs.coordinates,
                        visible: &inputs.visible,
                        fresh: &inputs.fresh,
                        destinations: &inputs.destinations,
                        $($extra: $value,)?
                        history_key_codes: key_codes,
                        history_key_coefficients: key_coefficients,
                        history_value_codes: value_codes,
                        history_value_coefficients: value_coefficients,
                        epsilon: case.epsilon,
                        scale: case.scale,
                        gate_function: 0,
                        slab_rows,
                    })
                    .unwrap()
                    .value
            }};
        }
        let result = match kind {
            Kind::Decode => dense!(attention_decode),
            Kind::Prefill => dense!(attention_prefill),
            Kind::DecodeK8V4 => affine!(attention_decode_k8v4),
            Kind::PrefillK8V4 => {
                affine!(attention_prefill_k8v4, history_tiles: &inputs.history_tiles)
            }
        };
        Ok(outputs(&result, history))
    }
}

fn outputs(result: &Tensor, history: &[Tensor]) -> Vec<(String, Vec<u8>)> {
    std::iter::once(("result".to_owned(), result.read_to_host().unwrap()))
        .chain(
            history
                .iter()
                .enumerate()
                .map(|(index, plane)| (format!("history{index}"), plane.read_to_host().unwrap())),
        )
        .collect()
}

fn geometries() -> [(&'static str, Geometry); 2] {
    [("qwen", QWEN), ("small", SMALL)]
}

fn open(backend: &str) -> Device {
    let backend = match backend {
        "metal" => BackendName::Metal,
        "cuda" => BackendName::Cuda,
        "vulkan" => BackendName::Vulkan,
        "cpu" => BackendName::Cpu,
        other => panic!("ATTENTION_GOLDEN_BACKEND={other}: expected metal, cuda, vulkan or cpu"),
    };
    DeviceCatalog::discover()
        .unwrap()
        .open_backend(backend)
        .unwrap()
}

#[test]
#[ignore = "golden capture/compare; run explicitly with ATTENTION_GOLDEN_MODE and ATTENTION_GOLDEN_DIR"]
fn attention_golden() {
    let record = match std::env::var("ATTENTION_GOLDEN_MODE").as_deref() {
        Ok("record") => true,
        Ok("compare") => false,
        other => panic!("ATTENTION_GOLDEN_MODE must be record or compare, got {other:?}"),
    };
    let backend = std::env::var("ATTENTION_GOLDEN_BACKEND").expect("ATTENTION_GOLDEN_BACKEND");
    let device = open(&backend);
    let path = std::path::PathBuf::from(
        std::env::var("ATTENTION_GOLDEN_DIR").expect("ATTENTION_GOLDEN_DIR"),
    )
    .join(format!("{backend}.golden"));
    let started = std::time::Instant::now();
    let mut lines = BTreeMap::new();
    let mut nondeterministic = Vec::new();
    for kind in [
        Kind::Decode,
        Kind::Prefill,
        Kind::DecodeK8V4,
        Kind::PrefillK8V4,
    ] {
        for (geometry_label, geometry) in geometries() {
            for named in cases(kind, geometry) {
                let inputs = Inputs::new(&device, &named.case);
                let family_inputs = Family::new(&device, &inputs, &named.case);
                for config in configurations(device.backend(), kind) {
                    let base = Family::statics(&device, geometry);
                    let mut specialization = config
                        .iter()
                        .fold(base, |s, (name, value)| s.with_param(*name, *value));
                    // Metal's and CUDA's decode and affine decode also slice
                    // the query group; the goldens (recorded
                    // before slicing existed) are one slice's walk.
                    let sliced = match device.backend() {
                        BackendName::Metal => matches!(kind, Kind::Decode | Kind::DecodeK8V4),
                        BackendName::Cuda => matches!(kind, Kind::Decode | Kind::DecodeK8V4),
                        _ => false,
                    };
                    if sliced {
                        specialization = specialization.with_param("SLICES", 1);
                    }
                    // Metal's and CUDA's decodes also have the grouped-query
                    // matrix form; the goldens are the vector form's walk.
                    if matches!(kind, Kind::Decode | Kind::DecodeK8V4) {
                        match device.backend() {
                            BackendName::Metal => {
                                specialization = specialization
                                    .with_param("MATRIX", 0)
                                    .with_param("KEYS", 16)
                                    .with_param("TOKENS", 1);
                            }
                            BackendName::Cuda => {
                                specialization = specialization
                                    .with_param("MATRIX", 0)
                                    .with_param("STAGES", 2)
                                    .with_param("COLUMNS", 1)
                            }
                            BackendName::Vulkan => {
                                specialization = specialization.with_param("MATRIX", 0);
                                if kind == Kind::DecodeK8V4 {
                                    specialization = specialization.with_param("KEYWISE", 0);
                                }
                            }
                            _ => {}
                        }
                    }
                    // Metal's prefill also splits a kv head's query heads
                    // into groups and has the direct form, and the affine
                    // entry takes a call that lists its history row tiles
                    // or not and has the COISSUE form; the goldens are one
                    // group's staged walk of a listing call.
                    if device.backend() == BackendName::Metal && !kind.decode() {
                        specialization = specialization
                            .with_param("HEADS", (geometry.g.next_power_of_two() as u64).min(16))
                            .with_param("DIRECT", 0);
                        if kind == Kind::PrefillK8V4 {
                            specialization =
                                specialization.with_static("L", 1).with_param("COISSUE", 0);
                        }
                    }
                    // CUDA's prefill also splits key tiles; the goldens are
                    // one partition's walk.
                    if device.backend() == BackendName::Cuda && !kind.decode() {
                        specialization = specialization
                            .with_param("SPLIT_GROUPS", 1)
                            .with_param("STAGES", 2)
                            .with_param("COLUMNS", 1)
                            .with_param("QREG", 0);
                        // Neither the producer warps' count nor decoding in
                        // the MMA warps (PRODUCERS = 0) changes the result.
                        if kind == Kind::PrefillK8V4 {
                            specialization = specialization.with_param("PRODUCERS", 4);
                        }
                    }
                    let config_label = config
                        .iter()
                        .map(|(n, v)| format!("{n}={v}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    let key = format!(
                        "{}|{geometry_label}|{config_label}|{}",
                        kind.name(),
                        named.label
                    );
                    let run = || {
                        let (_slabs, mut planes) =
                            history(&device, &named.case, kind.affine(), named.slab_rows);
                        run_family(
                            &device,
                            kind,
                            &specialization,
                            &inputs,
                            &family_inputs,
                            &mut planes,
                            &named,
                        )
                    };
                    let first = match run() {
                        Ok(first) => first,
                        Err(error) => {
                            println!("{key}: {error}");
                            lines.insert(format!("{key}|prepare"), "ERR".to_owned());
                            continue;
                        }
                    };
                    let second = if record { Some(run().unwrap()) } else { None };
                    for (index, (output, bytes)) in first.iter().enumerate() {
                        let digest = hash(bytes);
                        let stable = second
                            .as_ref()
                            .is_none_or(|second| second[index].1 == *bytes);
                        if !stable {
                            nondeterministic.push(format!("{key}|{output}"));
                        }
                        lines.insert(
                            format!("{key}|{output}"),
                            if stable { digest } else { "NONDET".into() },
                        );
                    }
                }
            }
        }
    }
    for line in &nondeterministic {
        println!("NONDETERMINISTIC {line}");
    }
    if record {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut text = String::new();
        for (key, value) in &lines {
            writeln!(text, "{key}\t{value}").unwrap();
        }
        std::fs::write(&path, text).unwrap();
        println!(
            "recorded {} keys to {} in {:.1}s",
            lines.len(),
            path.display(),
            started.elapsed().as_secs_f64()
        );
        return;
    }
    let golden = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(|line| {
            let (key, value) = line.split_once('\t').unwrap();
            (key.to_owned(), value.to_owned())
        })
        .collect::<BTreeMap<_, _>>();
    let mut mismatches = Vec::new();
    for key in golden
        .keys()
        .chain(lines.keys())
        .collect::<std::collections::BTreeSet<_>>()
    {
        let (expected, actual) = (golden.get(key), lines.get(key));
        if expected.map(String::as_str) == Some("NONDET") {
            continue;
        }
        if expected != actual {
            mismatches.push(format!("{key}: golden {expected:?} now {actual:?}"));
        }
    }
    for line in &mismatches {
        println!("MISMATCH {line}");
    }
    println!(
        "compared {} keys, {} mismatches, {:.1}s",
        lines.len(),
        mismatches.len(),
        started.elapsed().as_secs_f64()
    );
    assert!(
        mismatches.is_empty(),
        "{} golden mismatches",
        mismatches.len()
    );
}
