// `gated_attention_decode_k8v4` and `gated_attention_prefill_k8v4` (affine K8/V4
// history) on every accelerator present and the CPU: against their portable body (the
// reference interpreter, small shapes) and against a host model of that body
// (the dense host model over the decoded history, Qwen3.5-4B geometry, long
// contexts). Timings are ignored tests.
#![allow(dead_code)]

use magnitude_model_kernels::{
    gated_attention_decode, gated_attention_decode_k8v4, gated_attention_prefill,
    gated_attention_prefill_k8v4,
};
use seismic::{BackendName, Device, DeviceCatalog, Element, NativeSpecialization, Tensor};
use seismic_lang::registry::{f16_bits, f16_to_f32};

include!("attention_common/fixtures.rs");

const KEY_BITS: u32 = 8;
const VALUE_BITS: u32 = 4;
/// Values per (scale, zero) pair: the codec's group (model-state
/// `AFFINE_GROUP`).
const GROUP: usize = 32;

/// Two affine groups per head vector, for the decode portable-body comparison.
const GROUPED: Geometry = Geometry { kv: 2, g: 2, p: 8, s: 48 };

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
        let inverse = if f16_to_f32(scale) > 0.0 { 1.0 / f16_to_f32(scale) } else { 0.0 };
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
            decode(&self.key_codes[vector * kw..][..kw], &self.key_coefficients[vector * c..][..c], KEY_BITS, w),
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
    let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::u32(), &shape, &bytes).unwrap()
}

fn f16_tensor(device: &Device, shape: &[usize], bits: &[u16]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    let bytes = bits.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
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

/// Device tensors of one case.
struct Bound {
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
    key_codes: Tensor,
    key_coefficients: Tensor,
    value_codes: Tensor,
    value_coefficients: Tensor,
}

impl Bound {
    fn new(device: &Device, encoded: &Encoded) -> Self {
        let case = &encoded.case;
        let Geometry { kv, g, p, .. } = case.geometry;
        let (m, w, t) = (case.rows, case.geometry.w(), case.history_rows);
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
            key_codes: u32_tensor(device, &[t, kv, w / 4], &encoded.planes.key_codes),
            key_coefficients: f16_tensor(
                device,
                &[t, kv, coefficient_elements(w)],
                &encoded.planes.key_coefficients,
            ),
            value_codes: u32_tensor(device, &[t, kv, w / 8], &encoded.planes.value_codes),
            value_coefficients: f16_tensor(
                device,
                &[t, kv, coefficient_elements(w)],
                &encoded.planes.value_coefficients,
            ),
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
    ($module:ident, $bound:expr, $case:expr) => {
        $module::Args {
            query_gate: &$bound.query_gate,
            key: &$bound.key,
            value: &$bound.value,
            query_norm: &$bound.query_norm,
            key_norm: &$bound.key_norm,
            rotary_components: &$bound.components,
            rotary_frequencies: &$bound.frequencies,
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
        }
    };
}

/// The devices with k8v4 natives: every accelerator present (Metal, CUDA,
/// Vulkan) and the CPU.
fn k8v4_devices() -> Vec<Device> {
    let catalog = DeviceCatalog::discover().unwrap();
    [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan]
        .into_iter()
        .filter_map(|backend| catalog.open_backend(backend).ok())
        .chain(std::iter::once(catalog.open_backend(BackendName::Cpu).unwrap()))
        .collect()
}

/// Declared decode configurations of `backend`.
fn decode_configs(backend: BackendName) -> Vec<Vec<(&'static str, u64)>> {
    match backend {
        BackendName::Cuda => [(12, 4), (24, 8), (48, 4), (48, 8)]
            .into_iter()
            .map(|(parts, warps)| vec![("PARTS", parts), ("WARPS", warps)])
            .collect(),
        BackendName::Cpu => [8, 4, 16, 1].into_iter().map(|parts| vec![("PARTS", parts)]).collect(),
        _ => [(32, 16, 4), (64, 32, 8), (256, 8, 8), (32, 16, 8)]
            .into_iter()
            .map(|(span, parts, simds)| vec![("SPAN", span), ("PARTS", parts), ("SIMDS", simds)])
            .collect(),
    }
}

/// Declared prefill configurations of `backend`.
fn prefill_configs(backend: BackendName) -> Vec<Vec<(&'static str, u64)>> {
    match backend {
        BackendName::Cuda => [4, 2].into_iter().map(|warps| vec![("WARPS", warps)]).collect(),
        // One CPU form, without parameters.
        BackendName::Cpu => vec![Vec::new()],
        // ROWS = 64 is admissible at every tested geometry (ROWS / G <= 32,
        // and the W = 256 tile fits the shared bound).
        BackendName::Vulkan => [1, 256, 512]
            .into_iter()
            .map(|split| vec![("ROWS", 64), ("SPLIT_GROUPS", split)])
            .collect(),
        _ => [(16, 1), (8, 1), (16, 256), (8, 256)]
            .into_iter()
            .map(|(qt, split)| vec![("QT", qt), ("SPLIT_GROUPS", split)])
            .collect(),
    }
}

/// The specialization of `params` on `device`: GPU forms take the geometry's
/// statics; CPU forms have no statics.
fn specialization_on(device: &Device, geometry: Geometry, params: &[(&'static str, u64)]) -> NativeSpecialization {
    // The CPU forms fix only the head width, which their codec requires to be
    // whole 32-column groups.
    let base = if is_cpu(device) {
        NativeSpecialization::new().with_static("P", geometry.p as u64).with_static("S", geometry.s as u64)
    } else {
        statics(geometry)
    };
    params.iter().fold(base, |spec, (name, value)| spec.with_param(*name, *value))
}

fn decode_kernel(
    device: &Device,
    geometry: Geometry,
    params: &[(&'static str, u64)],
) -> seismic::NativeKernel<gated_attention_decode_k8v4::Entry> {
    gated_attention_decode_k8v4::native_for_device_with(
        device,
        gated_attention_decode_k8v4::Elements { A: Element::bf16() },
        &specialization_on(device, geometry, params),
    )
    .unwrap()
}

fn prefill_kernel(
    device: &Device,
    geometry: Geometry,
    params: &[(&'static str, u64)],
) -> seismic::NativeKernel<gated_attention_prefill_k8v4::Entry> {
    gated_attention_prefill_k8v4::native_for_device_with(
        device,
        gated_attention_prefill_k8v4::Elements { A: Element::bf16() },
        &specialization_on(device, geometry, params),
    )
    .unwrap()
}

/// Gated outputs agree with the host model within bf16 publication plus
/// reduced-precision staging. Planes: vectors nothing appends to are
/// untouched; appended vectors decode within one code step of the host
/// model's encoding (the key may differ by one bf16 step from transcendental
/// rounding, which can move a code).
fn check(label: &str, encoded: &Encoded, gated: &Tensor, bound: &Bound, expected: &(Vec<f32>, Planes)) {
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
        .flat_map(|destination| (0..geometry.kv).map(move |head| *destination as usize * geometry.kv + head))
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
                let (key_step, value_step) =
                    (step(&expected.1.key_coefficients), step(&expected.1.value_coefficients));
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
            assert_eq!((key, value), (expected_key, expected_value), "{label}: vector {vector} was not appended to");
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
        checked::{SourceFile, check_source},
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
    let halves = |values: &[u16]| values.iter().map(|x| f64::from(f16_to_f32(*x))).collect::<Vec<_>>();
    let planes = &encoded.planes;
    let args = vec![
        tensor(DType::BF16, vec![m, kv * g * 2 * w], floats(&case.query_gate)),
        tensor(DType::BF16, vec![m, kv * w], floats(&case.key)),
        tensor(DType::BF16, vec![m, kv * w], floats(&case.value)),
        tensor(DType::F32, vec![w], floats(&case.query_norm)),
        tensor(DType::F32, vec![w], floats(&case.key_norm)),
        tensor(DType::I32, vec![p], ints(&case.components)),
        tensor(DType::F32, vec![p], floats(&case.frequencies)),
        tensor(DType::I32, vec![m, 4], ints(&case.coordinates)),
        tensor(DType::I32, vec![m, case.spans, 2], ints(&case.visible)),
        tensor(DType::I32, vec![m, 2], ints(&case.fresh)),
        tensor(DType::I32, vec![m], ints(&case.destinations)),
        tensor(DType::U32, vec![t, kv, w / 4], unsigned(&planes.key_codes)),
        tensor(DType::F16, vec![t, kv, coefficient_elements(w)], halves(&planes.key_coefficients)),
        tensor(DType::U32, vec![t, kv, w / 8], unsigned(&planes.value_codes)),
        tensor(DType::F16, vec![t, kv, coefficient_elements(w)], halves(&planes.value_coefficients)),
        Arg::Scalar(ReferenceScalar::F32(case.epsilon.to_bits())),
        Arg::Scalar(ReferenceScalar::F32(case.scale.to_bits())),
    ];
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
        let input = inputs.iter().find(|input| input.ordinal() == ordinal).unwrap();
        let tensor = input.tensor();
        (0..tensor.element_count()).map(|index| tensor.read(index).unwrap()).collect::<Vec<_>>()
    };
    let portable = Planes {
        key_codes: read(11).into_iter().map(|x| x as u32).collect(),
        key_coefficients: read(12).into_iter().map(|x| f16_bits(x as f32)).collect(),
        value_codes: read(13).into_iter().map(|x| x as u32).collect(),
        value_coefficients: read(14).into_iter().map(|x| f16_bits(x as f32)).collect(),
    };
    assert_eq!(portable.value_codes, expected.1.value_codes, "{entry} portable value codes");
    assert_eq!(portable.value_coefficients, expected.1.value_coefficients, "{entry} portable value coefficients");
    for vector in 0..t * kv {
        let (key, _) = portable.decoded(case.geometry, vector);
        let (expected_key, _) = expected.1.decoded(case.geometry, vector);
        for column in 0..w {
            let step =
                f16_to_f32(expected.1.key_coefficients[vector * coefficient_elements(w) + column / GROUP * 2]);
            assert!(
                (key[column] - expected_key[column]).abs() <= 1.01 * step + 8.0e-3 * expected_key[column].abs().max(1.0),
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
    check_host_model_against_portable_body("gated_attention_decode_k8v4", &encoded);
    for device in k8v4_devices() {
        decode_matches_portable_body_on(&device, &encoded);
    }
}

fn decode_matches_portable_body_on(device: &Device, encoded: &Encoded) {
    let backend = device.backend();
    let expected = encoded.expected();
    for config in decode_configs(backend) {
        let kernel = decode_kernel(device, GROUPED, &config);
        let mut bound = Bound::new(device, encoded);
        let gated = kernel.call(args!(gated_attention_decode_k8v4, bound, encoded.case)).unwrap().value;
        check(&format!("{backend:?} grouped decode {config:?}"), encoded, &gated, &bound, &expected);
    }
}

#[test]
fn prefill_matches_portable_body() {
    // 300 history rows are 10 key tiles: the split configurations split the
    // first sequence's history spans over two partitions and merge them.
    // One affine group per vector keeps the interpreter's prefill affordable;
    // the Qwen geometry test covers eight groups.
    let encoded = Encoded::new(Case::new(SMALL, 400, 2, &prefill_rows(20, 300), 23));
    check_host_model_against_portable_body("gated_attention_prefill_k8v4", &encoded);
    for device in k8v4_devices() {
        prefill_matches_portable_body_on(&device, &encoded);
    }
}

fn prefill_matches_portable_body_on(device: &Device, encoded: &Encoded) {
    let backend = device.backend();
    let expected = encoded.expected();
    for config in prefill_configs(backend) {
        let kernel = prefill_kernel(device, SMALL, &config);
        let mut bound = Bound::new(device, encoded);
        let gated = kernel.call(args!(gated_attention_prefill_k8v4, bound, encoded.case)).unwrap().value;
        check(&format!("{backend:?} small prefill {config:?}"), encoded, &gated, &bound, &expected);
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
        let encoded = Encoded::new(Case::new(QWEN, context + 128, 2, &decode_rows(context as i32 - 40), 5));
        let expected = encoded.expected();
        for config in decode_configs(backend) {
            let kernel = decode_kernel(device, QWEN, &config);
            let mut bound = Bound::new(device, &encoded);
            let gated = kernel.call(args!(gated_attention_decode_k8v4, bound, encoded.case)).unwrap().value;
            check(&format!("{backend:?} decode context {context} {config:?}"), &encoded, &gated, &bound, &expected);
        }
        for rows in [1, 8] {
            let encoded = Encoded::new(Case::new(QWEN, context + rows, 1, &speculative_rows(rows, context as i32), 13));
            let expected = encoded.expected();
            let config = &decode_configs(backend)[0];
            let kernel = decode_kernel(device, QWEN, config);
            let mut bound = Bound::new(device, &encoded);
            let gated = kernel.call(args!(gated_attention_decode_k8v4, bound, encoded.case)).unwrap().value;
            check(&format!("{backend:?} speculative decode {rows} rows context {context} {config:?}"),
                &encoded, &gated, &bound, &expected);
        }
    }
    for (rows, history) in [(40, 300), (128, 1000), (64, 4096), (48, 16384)] {
        let encoded = Encoded::new(Case::new(QWEN, history as usize + 256, 2, &prefill_rows(rows, history), 7));
        let expected = encoded.expected();
        for config in prefill_configs(backend) {
            let kernel = prefill_kernel(device, QWEN, &config);
            let mut bound = Bound::new(device, &encoded);
            let gated = kernel.call(args!(gated_attention_prefill_k8v4, bound, encoded.case)).unwrap().value;
            check(&format!("{backend:?} prefill {rows} rows after {history} {config:?}"), &encoded, &gated, &bound, &expected);
        }
    }
}

/// The dense history planes of a case, for timing the dense entries on the
/// same inputs.
struct DenseHistory {
    key: Tensor,
    value: Tensor,
}

impl DenseHistory {
    fn new(device: &Device, case: &Case) -> Self {
        let (kv, w, t) = (case.geometry.kv, case.geometry.w(), case.history_rows);
        Self {
            key: bf16_tensor(device, &[t, kv, w], &case.history_key),
            value: bf16_tensor(device, &[t, kv, w], &case.history_value),
        }
    }
}

macro_rules! dense_args {
    ($module:ident, $bound:expr, $dense:expr, $case:expr) => {
        $module::Args {
            query_gate: &$bound.query_gate,
            key: &$bound.key,
            value: &$bound.value,
            query_norm: &$bound.query_norm,
            key_norm: &$bound.key_norm,
            rotary_components: &$bound.components,
            rotary_frequencies: &$bound.frequencies,
            coordinates: &$bound.coordinates,
            visible: &$bound.visible,
            fresh: &$bound.fresh,
            destinations: &$bound.destinations,
            history_key: &mut $dense.key,
            history_value: &mut $dense.value,
            epsilon: $case.epsilon,
            scale: $case.scale,
        }
    };
}

/// Device time of one decode layer's attention at Qwen3.5-4B geometry over
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

fn decode_timing_on(device: &Device) {
    let backend = device.backend();
    for context in [1usize, 256, 4096, 16384, 65536] {
        let rows = [Row {
            spans: vec![(0, context as i32 - 1)],
            fresh: (0, 1),
            destination: context as i32 - 1,
            position: context as i32 - 1,
        }];
        let encoded = Encoded::new(Case::new(QWEN, context + 64, 1, &rows, 3));
        let mut dense = DenseHistory::new(device, &encoded.case);
        for config in decode_configs(backend) {
            let kernel = decode_kernel(device, QWEN, &config);
            let mut bound = Bound::new(device, &encoded);
            let affine = kernel
                .measure(vec![args!(gated_attention_decode_k8v4, bound, encoded.case)], &TIMING)
                .unwrap()
                .median;
            let affine_bytes =
                (context * QWEN.kv * (QWEN.w() + QWEN.w() / 2 + 4 * coefficient_elements(QWEN.w()))) as f64;
            let dense_bytes = (context * QWEN.kv * QWEN.w() * 4) as f64;
            // The dense entry's domain may not admit this configuration.
            let dense_time = gated_attention_decode::native_for_device_with(
                device,
                gated_attention_decode::Elements { A: Element::bf16() },
                &specialization_on(device, QWEN, &config),
            )
            .ok()
            .map(|dense_kernel| {
                dense_kernel
                    .measure(vec![dense_args!(gated_attention_decode, bound, dense, encoded.case)], &TIMING)
                    .unwrap()
                    .median
            });
            let dense_report = dense_time.map_or("dense not admitted".to_owned(), |dense_time| {
                format!(
                    "dense {:.1} us ({:.0} GB/s), k8v4/dense {:.2}",
                    dense_time * 1e6,
                    dense_bytes / dense_time / 1e9,
                    affine / dense_time
                )
            });
            eprintln!(
                "{backend:?} decode context {context} {config:?}: k8v4 {:.1} us ({:.0} GB/s), {dense_report}",
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
                (rows.parse::<usize>().unwrap(), history.parse::<i32>().unwrap())
            })
            .collect::<Vec<_>>(),
        Err(_) => vec![(128usize, 0i32), (512, 0), (512, 4096), (512, 16384), (512, 65536)],
    };
    for (rows, history) in rows_history {
        let spans = if history > 0 { vec![(0, history)] } else { vec![] };
        let rows = (0..rows)
            .map(|row| Row {
                spans: spans.clone(),
                fresh: (0, row as i32 + 1),
                destination: history + row as i32,
                position: history + row as i32,
            })
            .collect::<Vec<_>>();
        let pairs = rows.iter().map(|row| (history + row.fresh.1 - row.fresh.0) as f64).sum::<f64>();
        let flop = pairs * (QWEN.kv * QWEN.g * QWEN.w() * 4) as f64;
        let encoded = Encoded::new(Case::new(QWEN, history as usize + rows.len(), 1, &rows, 9));
        let mut dense = DenseHistory::new(device, &encoded.case);
        for config in prefill_configs(backend) {
            let kernel = prefill_kernel(device, QWEN, &config);
            let mut bound = Bound::new(device, &encoded);
            let affine = kernel
                .measure(vec![args!(gated_attention_prefill_k8v4, bound, encoded.case)], &TIMING)
                .unwrap()
                .median;
            let dense_kernel = gated_attention_prefill::native_for_device_with(
                device,
                gated_attention_prefill::Elements { A: Element::bf16() },
                &specialization_on(device, QWEN, &config),
            )
            .unwrap();
            let dense_time = dense_kernel
                .measure(vec![dense_args!(gated_attention_prefill, bound, dense, encoded.case)], &TIMING)
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
