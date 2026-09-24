//! K1/K2 projection entries on Metal (macOS) and Vulkan (elsewhere). Weights
//! are built host-side in the registry's `rows16` layout; every entry is
//! checked against its portable body (the Seismic interpreter) on small
//! shapes, and against a host reference over the row classes M in {1, 2, 3,
//! 8, 16, 64, 128}, across weight representations and mapping parameters.

use magnitude_model_kernels::{
    attention_output, qwen_attention_project, qwen_dense_expand, qwen_dense_output,
    embedding_rows, readout_features_rows, readout_head_rows, qwen_recurrent_output,
    qwen_recurrent_project, readout_selected_rows,
};
use seismic::{Device, Element, NativeSpecialization, Tensor};
use seismic_lang::{
    checked::{check_source, CheckedModule, SourceFile},
    entry::ElementBindings,
    failure::SourceTermination,
    ids::RepresentationId,
    interp::{Arg, Interpreter, OracleOutcome, OutcomeValue, TensorData},
    reference_math::ReferenceScalar,
    registry::{self, RepresentationKind},
    types::DType,
};

// ---------------------------------------------------------------------------
// Host values.

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits >> 15 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    sign * if exponent == 0 {
        mantissa * 2f32.powi(-24)
    } else {
        (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
    }
}

fn f16_bits(value: f32) -> u16 {
    registry::f16_bits(value)
}

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Act {
    Bf16,
    F16,
}

impl Act {
    fn round(self, value: f32) -> f32 {
        match self {
            Act::Bf16 => bf16_round(value),
            Act::F16 => f16_to_f32(f16_bits(value)),
        }
    }
    fn element(self) -> Element {
        match self {
            Act::Bf16 => Element::bf16(),
            Act::F16 => Element::f16(),
        }
    }
    fn dtype(self) -> DType {
        match self {
            Act::Bf16 => DType::BF16,
            Act::F16 => DType::F16,
        }
    }
    fn bytes(self, values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| match self {
                Act::Bf16 => ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes(),
                Act::F16 => f16_bits(*value).to_le_bytes(),
            })
            .collect()
    }
    fn decode(self, bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|pair| {
                let bits = u16::from_le_bytes([pair[0], pair[1]]);
                match self {
                    Act::Bf16 => f32::from_bits(u32::from(bits) << 16),
                    Act::F16 => f16_to_f32(bits),
                }
            })
            .collect()
    }
    /// Relative size of one unit in the last place.
    fn ulp(self) -> f32 {
        match self {
            Act::Bf16 => 1.0 / 128.0,
            Act::F16 => 1.0 / 1024.0,
        }
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407))
    }
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
    fn uniform(&mut self) -> f32 {
        (self.next() >> 8) as f32 / (1u32 << 24) as f32
    }
    fn symmetric(&mut self) -> f32 {
        self.uniform() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// rows16 weights.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Repr {
    Q4k,
    Q5k,
    Q6k,
    Q8,
    Bf16,
}

impl Repr {
    fn name(self) -> &'static str {
        match self {
            Repr::Q4k => "q4k",
            Repr::Q5k => "q5k",
            Repr::Q6k => "q6k",
            Repr::Q8 => "q8g32s",
            Repr::Bf16 => "bf16",
        }
    }
    fn storage(self) -> RepresentationId {
        match self {
            Repr::Bf16 => registry::dense(DType::BF16),
            _ => registry::storage(self.name(), registry::Layout::Rows16).expect("rows16 storage"),
        }
    }
    fn element(self) -> Element {
        match self {
            Repr::Bf16 => Element::bf16(),
            _ => Element::stored(self.name(), registry::Layout::Rows16).expect("rows16 element"),
        }
    }
}

/// Plane byte offsets of one row, in the frozen rows16 order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Planes {
    stride: usize,
    codes: usize,
    high: usize,
    scales: usize,
    supers: usize,
}

fn planes(repr: Repr, k: usize) -> Planes {
    let blocks = k / 256;
    let mut planes = Planes::default();
    let mut at = 0;
    let mut place = |bytes: usize| {
        let offset = at;
        at += bytes.div_ceil(16) * 16;
        offset
    };
    match repr {
        Repr::Q4k => {
            planes.codes = place(k / 2);
            planes.scales = place(12 * blocks);
            planes.supers = place(4 * blocks);
        }
        Repr::Q5k => {
            planes.codes = place(k / 2);
            planes.high = place(k / 8);
            planes.scales = place(12 * blocks);
            planes.supers = place(4 * blocks);
        }
        Repr::Q6k => {
            planes.codes = place(k / 2);
            planes.high = place(k / 4);
            planes.scales = place(k / 16);
            planes.supers = place(2 * blocks);
        }
        Repr::Q8 => {
            planes.codes = place(k);
            planes.supers = place(2 * (k / 32));
        }
        Repr::Bf16 => at = 2 * k,
    }
    planes.stride = at;
    planes
}

struct Weight {
    repr: Repr,
    rows: usize,
    k: usize,
    bytes: Vec<u8>,
    /// Decoded logical values [rows, k].
    values: Vec<f32>,
}

impl Weight {
    fn tensor(&self, device: &Device) -> Tensor {
        Tensor::from_host(device, self.repr.element(), &[self.rows as u64, self.k as u64], &self.bytes)
            .unwrap()
    }
    fn oracle(&self) -> TensorData {
        match self.repr {
            Repr::Bf16 => TensorData::dense(
                DType::BF16,
                vec![self.rows, self.k],
                self.values.iter().map(|value| f64::from(*value)).collect(),
            ),
            _ => TensorData::encoded(self.repr.storage(), vec![self.rows, self.k], self.bytes.clone())
                .unwrap(),
        }
    }
    fn row(&self, n: usize) -> &[f32] {
        &self.values[n * self.k..(n + 1) * self.k]
    }
}

fn put_bits(bytes: &mut [u8], bit: usize, width: usize, value: u32) {
    for i in 0..width {
        if (value >> i) & 1 != 0 {
            bytes[(bit + i) / 8] |= 1 << ((bit + i) % 8);
        }
    }
}

/// A random weight whose rows have dot products of order `scale` with unit
/// activations.
fn weight(repr: Repr, rows: usize, k: usize, seed: u64, scale: f32) -> Weight {
    let scale = scale * 2.0 / (k as f32).sqrt();
    let mut rng = Rng::new(seed);
    let p = planes(repr, k);
    let mut bytes = vec![0u8; p.stride * rows];
    let mut values = vec![0f32; rows * k];
    for r in 0..rows {
        let row = &mut bytes[r * p.stride..(r + 1) * p.stride];
        let out = &mut values[r * k..(r + 1) * k];
        match repr {
            Repr::Q4k | Repr::Q5k => {
                let bits = if repr == Repr::Q4k { 4 } else { 5 };
                let center = if bits == 5 { 15.5 } else { 7.5 };
                for block in 0..k / 256 {
                    let d_value = scale / (center * 32.0) * (0.5 + rng.uniform());
                    let d = f16_bits(d_value);
                    let dmin = f16_bits(d_value * center * (0.75 + 0.5 * rng.uniform()));
                    row[p.supers + 4 * block..p.supers + 4 * block + 2].copy_from_slice(&d.to_le_bytes());
                    row[p.supers + 4 * block + 2..p.supers + 4 * block + 4]
                        .copy_from_slice(&dmin.to_le_bytes());
                    for group in 0..8 {
                        let local = rng.below(64);
                        let minimum = rng.below(64);
                        let fields = &mut row[p.scales + 12 * block..p.scales + 12 * block + 12];
                        put_bits(fields, 12 * group, 6, local);
                        put_bits(fields, 12 * group + 6, 6, minimum);
                        let group_scale = f16_to_f32(d) * local as f32;
                        let group_bias = -(f16_to_f32(dmin) * minimum as f32);
                        for i in 0..32 {
                            let column = 256 * block + 32 * group + i;
                            let code = rng.below(1 << bits);
                            put_bits(&mut row[p.codes..], 4 * column, 4, code & 15);
                            if bits == 5 {
                                put_bits(&mut row[p.high..], column, 1, code >> 4);
                            }
                            out[column] = group_scale.mul_add(code as f32, group_bias);
                        }
                    }
                }
            }
            Repr::Q6k => {
                for block in 0..k / 256 {
                    let d = f16_bits(scale / 32.0 / 48.0 * (0.5 + rng.uniform()));
                    row[p.supers + 2 * block..p.supers + 2 * block + 2].copy_from_slice(&d.to_le_bytes());
                    for group in 0..16 {
                        let local = (rng.below(255) as i32 - 127) as i8;
                        row[p.scales + 16 * block + group] = local as u8;
                        let group_scale = f16_to_f32(d) * f32::from(local);
                        for i in 0..16 {
                            let column = 256 * block + 16 * group + i;
                            let code = rng.below(64);
                            put_bits(&mut row[p.codes..], 4 * column, 4, code & 15);
                            put_bits(&mut row[p.high..], 2 * column, 2, code >> 4);
                            out[column] = group_scale * (code as i32 - 32) as f32;
                        }
                    }
                }
            }
            Repr::Q8 => {
                for group in 0..k / 32 {
                    let factor = f16_bits(scale / 96.0 * (0.5 + rng.uniform()));
                    row[p.supers + 2 * group..p.supers + 2 * group + 2].copy_from_slice(&factor.to_le_bytes());
                    for i in 0..32 {
                        let column = 32 * group + i;
                        let code = (rng.below(255) as i32 - 127) as i8;
                        row[p.codes + column] = code as u8;
                        out[column] = f16_to_f32(factor) * f32::from(code);
                    }
                }
            }
            Repr::Bf16 => {
                for column in 0..k {
                    let value = bf16_round(rng.symmetric() * scale);
                    row[2 * column..2 * column + 2]
                        .copy_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
                    out[column] = value;
                }
            }
        }
    }
    Weight { repr, rows, k, bytes, values }
}

// ---------------------------------------------------------------------------
// References.

/// Ordered F32 FMA dot and the magnitude sum bounding its reassociation error.
fn dot(x: &[f32], w: &[f32]) -> (f32, f32) {
    let mut acc = 0f32;
    let mut magnitude = 0f32;
    for (a, b) in x.iter().zip(w) {
        acc = a.mul_add(*b, acc);
        magnitude += (a * b).abs();
    }
    (acc, magnitude)
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

/// RMS normalization of one row rounded to the activation, with, per element,
/// the rounding flip a device may legitimately take because its square sum is
/// ordered differently (zero away from rounding boundaries).
fn rms_row(act: Act, x: &[f32], norm: &[f32], epsilon: f32) -> (Vec<f32>, Vec<f32>) {
    let squares = x.iter().fold(0f32, |sum, value| value.mul_add(*value, sum));
    let inverse = 1.0 / (squares / x.len() as f32 + epsilon).sqrt();
    let values = x.iter().zip(norm).map(|(v, w)| act.round(v * inverse * w)).collect();
    let slack = x
        .iter()
        .zip(norm)
        .map(|(v, w)| {
            let exact = v * inverse * w;
            (act.round(exact * (1.0 + 1e-5)) - act.round(exact * (1.0 - 1e-5))).abs()
        })
        .collect();
    (values, slack)
}

fn slack_dot(slack: &[f32], w: &[f32]) -> f32 {
    slack.iter().zip(w).map(|(s, w)| s * w.abs()).sum()
}

/// Bound for an A-rounded projection: reassociation plus one rounding flip.
fn rounded_bound(act: Act, value: f32, magnitude: f32) -> f32 {
    reassociation_bound(magnitude) + value.abs() * act.ulp() + 1e-7
}

fn assert_within(label: &str, actual: &[f32], expected: &[f32], bound: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let failures = (0..expected.len())
        .filter(|i| !((actual[*i] - expected[*i]).abs() <= bound[*i]))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty(),
        "{label}: {} of {} outside tolerance; first {:?}",
        failures.len(),
        expected.len(),
        failures
            .iter()
            .take(4)
            .map(|i| (*i, actual[*i], expected[*i], bound[*i]))
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Device helpers.

/// Metal on macOS; elsewhere Vulkan (rows16 is the resident layout of both).
/// Hosts without either skip.
fn device() -> Option<Device> {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    if cfg!(target_os = "macos") {
        Some(catalog.open_backend(seismic::BackendName::Metal).unwrap())
    } else {
        catalog.open_backend(seismic::BackendName::Vulkan).ok()
    }
}

fn f32_tensor(device: &Device, shape: &[usize], values: &[f32]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::f32(), &shape, &bytes).unwrap()
}

fn i32_tensor(device: &Device, shape: &[usize], values: &[i32]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::i32(), &shape, &bytes).unwrap()
}

fn act_tensor(device: &Device, act: Act, shape: &[usize], values: &[f32]) -> Tensor {
    let shape = shape.iter().map(|x| *x as u64).collect::<Vec<_>>();
    Tensor::from_host(device, act.element(), &shape, &act.bytes(values)).unwrap()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn read_act(act: Act, tensor: &Tensor) -> Vec<f32> {
    act.decode(&tensor.read_to_host().unwrap())
}

/// A K1 mapping: the GEMV threadgroup (SIMDGROUPS x ROWS lane groups of
/// LANES lanes), the first row count served by the batched GEMV (BATCH_FROM;
/// smaller classes run the GEMV), the batched threadgroup (BATCH_SIMDGROUPS x
/// BATCH_ROWS blocks of 8 weight rows), the GEMM tile (TILE_M x TILE_N) and,
/// for the small-N output projections, the split-K factor.
#[derive(Clone, Copy, Debug)]
struct Mapping {
    simdgroups: u64,
    rows: u64,
    lanes: u64,
    batch_from: u64,
    batch_simdgroups: u64,
    batch_rows: u64,
    tile_m: u64,
    tile_n: u64,
    split: u64,
}

/// Decode defaults with the given GEMM tile and split.
const fn gemm_mapping(tile_m: u64, tile_n: u64, split: u64) -> Mapping {
    Mapping {
        simdgroups: 16,
        rows: 1,
        lanes: 16,
        batch_from: 5,
        batch_simdgroups: 8,
        batch_rows: 2,
        tile_m,
        tile_n,
        split,
    }
}

const MAPPINGS: [Mapping; 4] = [
    Mapping { simdgroups: 4, rows: 1, lanes: 32, batch_from: 3, batch_simdgroups: 4, batch_rows: 1, tile_m: 64, tile_n: 64, split: 1 },
    Mapping { simdgroups: 2, rows: 4, lanes: 16, batch_from: 9, batch_simdgroups: 16, batch_rows: 2, tile_m: 128, tile_n: 64, split: 2 },
    Mapping { simdgroups: 32, rows: 2, lanes: 32, batch_from: 5, batch_simdgroups: 8, batch_rows: 4, tile_m: 32, tile_n: 128, split: 4 },
    Mapping { simdgroups: 16, rows: 2, lanes: 16, batch_from: 3, batch_simdgroups: 32, batch_rows: 1, tile_m: 64, tile_n: 128, split: 2 },
];

/// Relative bound on the reassociation error of a dot product over
/// `outputs` output rows: F32 accumulation in the GEMV (outputs <= 2), plus
/// half-rounded operands on the matrix units (batched GEMV and GEMM).
fn reassociation(outputs: usize) -> f32 {
    if outputs > 2 { 7e-4 } else { 3e-5 }
}

thread_local! {
    static REASSOCIATION: std::cell::Cell<f32> = const { std::cell::Cell::new(3e-5) };
}

fn reassociation_bound(magnitude: f32) -> f32 {
    REASSOCIATION.with(|factor| factor.get()) * magnitude
}

/// Evaluate a reference under the reassociation bound of `outputs` rows.
fn under<T>(outputs: usize, reference: impl FnOnce() -> T) -> T {
    REASSOCIATION.with(|factor| factor.set(reassociation(outputs)));
    let value = reference();
    REASSOCIATION.with(|factor| factor.set(3e-5));
    value
}

fn specialization(statics: &[(&str, usize)], mapping: Mapping) -> NativeSpecialization {
    let specialization = statics
        .iter()
        .fold(NativeSpecialization::new(), |spec, (name, value)| spec.with_static(*name, *value as u64))
        .with_param("SIMDGROUPS", mapping.simdgroups)
        .with_param("ROWS", mapping.rows)
        .with_param("LANES", mapping.lanes)
        .with_param("TILE_M", mapping.tile_m)
        .with_param("TILE_N", mapping.tile_n);
    // Vulkan has no batched GEMV class (its GEMV serves M <= 8).
    if cfg!(target_os = "macos") {
        specialization
            .with_param("BATCH_FROM", mapping.batch_from)
            .with_param("BATCH_SIMDGROUPS", mapping.batch_simdgroups)
            .with_param("BATCH_ROWS", mapping.batch_rows)
    } else {
        specialization
    }
}

/// The specialization of an output projection with split-K.
fn split_specialization(statics: &[(&str, usize)], mapping: Mapping) -> NativeSpecialization {
    specialization(statics, mapping).with_param("SPLIT", mapping.split)
}

const ROW_CLASSES: [usize; 9] = [1, 2, 3, 8, 9, 16, 17, 64, 128];

/// The decode and verify rows checked at the pinned Qwen3.5 4B geometry: the
/// GEMV (1, 2, 4), the batched class (5, 16) and the first GEMM class (17).
const FOUR_B_ROWS: [usize; 6] = [1, 2, 4, 5, 16, 17];

/// The row classes and mappings a geometry is checked over: every mapping at
/// the small geometry, the decode mapping over the decode and verify rows at
/// the 4B geometry.
fn checked_classes(four_b: bool) -> (Vec<usize>, Vec<Mapping>) {
    if four_b {
        (FOUR_B_ROWS.to_vec(), vec![gemm_mapping(64, 64, 1)])
    } else {
        (ROW_CLASSES.to_vec(), MAPPINGS.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Portable bodies.

fn module() -> CheckedModule {
    let mut sources = seismic_std::sources();
    for (path, text) in [
        ("dense_rows.seismic", include_str!("../kernels/dense_rows.seismic")),
        ("readout.seismic", include_str!("../kernels/readout.seismic")),
        ("target.seismic", include_str!("../kernels/target.seismic")),
        ("recurrent.seismic", include_str!("../kernels/recurrent.seismic")),
        ("attention.seismic", include_str!("../kernels/attention.seismic")),
    ] {
        sources.push(SourceFile { path: path.into(), text: text.into() });
    }
    check_source(sources).unwrap()
}

enum Input {
    Data(TensorData),
    F32(f32),
}

fn floats(shape: &[usize], values: &[f32]) -> Input {
    Input::Data(TensorData::dense(DType::F32, shape.to_vec(), values.iter().map(|v| f64::from(*v)).collect()))
}

fn ints(shape: &[usize], values: &[i32]) -> Input {
    Input::Data(TensorData::dense(DType::I32, shape.to_vec(), values.iter().map(|v| f64::from(*v)).collect()))
}

fn activations(act: Act, shape: &[usize], values: &[f32]) -> Input {
    Input::Data(TensorData::dense(act.dtype(), shape.to_vec(), values.iter().map(|v| f64::from(*v)).collect()))
}

fn interpret(module: &CheckedModule, name: &str, bindings: &[(&str, RepresentationId)], inputs: Vec<Input>) -> OracleOutcome {
    let elements = bindings
        .iter()
        .fold(ElementBindings::new(), |elements, (name, id)| elements.bind(name, *id));
    let logical = module.entry(module.entry_named(name).unwrap(), &elements).unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let arguments = inputs
        .into_iter()
        .map(|input| match input {
            Input::Data(data) => Arg::Tensor(interpreter.add_tensor(data)),
            Input::F32(value) => Arg::Scalar(ReferenceScalar::F32(value.to_bits())),
        })
        .collect::<Vec<_>>();
    let outcome = interpreter.run(&arguments).unwrap_or_else(|error| panic!("{name}: {error}"));
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("{name} failed: {failure}");
    }
    outcome
}

fn result(outcome: &OracleOutcome, index: usize) -> Vec<f32> {
    let result = outcome.results().nth(index).unwrap();
    let OutcomeValue::Tensor(tensor) = result.value() else { panic!("tensor result") };
    (0..tensor.element_count()).map(|i| tensor.read(i).unwrap() as f32).collect()
}

fn norm_values(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..n).map(|_| bf16_round(0.5 + rng.uniform())).collect()
}

fn bf16_norm(device: &Device, values: &[f32]) -> Tensor {
    act_tensor(device, Act::Bf16, &[values.len()], values)
}

// ---------------------------------------------------------------------------
// Tests.

#[test]
fn rows16_builder_matches_the_registry_layout_and_decode() {
    for repr in [Repr::Q4k, Repr::Q5k, Repr::Q6k, Repr::Q8] {
        let k = 512;
        let w = weight(repr, 3, k, 1, 1.0);
        let info = registry::representation_info(repr.storage());
        let RepresentationKind::PackedRows(layout) = &info.kind else { panic!("{repr:?} rows16") };
        let expected = planes(repr, k);
        assert_eq!(layout.row_stride_bytes(k as u64).unwrap() as usize, expected.stride, "{repr:?} stride");
        for (ordinal, plane) in layout.planes.iter().enumerate() {
            let offset = layout.plane_row_offset(ordinal, k as u64).unwrap() as usize;
            let mine = match plane.name {
                "codes_lo" | "codes" => expected.codes,
                "codes_hi" => expected.high,
                "scales" => expected.scales,
                "supers" => expected.supers,
                other => panic!("unexpected plane {other}"),
            };
            assert_eq!(offset, mine, "{repr:?} plane {}", plane.name);
        }
        let decoded = w.oracle().values().unwrap();
        for (i, (oracle, host)) in decoded.iter().zip(&w.values).enumerate() {
            assert_eq!(*oracle as f32, *host, "{repr:?} value {i}");
        }
    }
}

fn dense_expand_native(device: &Device, act: Act, gate: &Weight, up: &Weight, residual: &[f32], rows: usize,
    norm: &[f32], out_rows: &[i32], eps: f32, mapping: Mapping) -> Vec<f32> {
    let (h, f) = (gate.k, gate.rows);
    let kernel = qwen_dense_expand::native_for_device_with(
        device,
        qwen_dense_expand::Elements { NW: Element::bf16(), GW: gate.repr.element(), UW: up.repr.element(), A: act.element() },
        &specialization(&[("H", h), ("F", f)], mapping),
    )
    .unwrap();
    let out = kernel
        .call(qwen_dense_expand::Args {
            residual: &f32_tensor(device, &[rows, h], residual),
            norm: &bf16_norm(device, norm),
            gate_weight: &gate.tensor(device),
            up_weight: &up.tensor(device),
            out_rows: &i32_tensor(device, &[out_rows.len()], out_rows),
            eps,
        })
        .unwrap()
        .value;
    read_act(act, &out)
}

fn dense_expand_reference(act: Act, gate: &Weight, up: &Weight, residual: &[f32], norm: &[f32], out_rows: &[i32],
    eps: f32) -> (Vec<f32>, Vec<f32>) {
    let (h, f) = (gate.k, gate.rows);
    let mut expected = Vec::new();
    let mut bound = Vec::new();
    for row in out_rows {
        let source = *row as usize;
        let (x, slack) = rms_row(act, &residual[source * h..(source + 1) * h], norm, eps);
        for n in 0..f {
            let (g, gm) = dot(&x, gate.row(n));
            let (u, um) = dot(&x, up.row(n));
            let (g, u) = (act.round(g), act.round(u));
            let a = act.round(silu(g));
            let value = act.round(a * u);
            expected.push(value);
            let gate_error = reassociation_bound(gm) + g.abs() * act.ulp() + slack_dot(&slack, gate.row(n));
            let up_error = reassociation_bound(um) + u.abs() * act.ulp() + slack_dot(&slack, up.row(n));
            bound.push(1.2 * (gate_error * u.abs() + up_error * a.abs()) + value.abs() * act.ulp() + 1e-6);
        }
    }
    (expected, bound)
}

#[test]
fn dense_expand_matches_its_portable_body() {
    let Some(device) = device() else { return };
    let module = module();
    let act = Act::Bf16;
    let (h, f) = (256, 40);
    for (gate_repr, up_repr) in [(Repr::Q4k, Repr::Q6k), (Repr::Q5k, Repr::Q8)] {
        let gate = weight(gate_repr, f, h, 11, 1.0);
        let up = weight(up_repr, f, h, 12, 1.0);
        let norm = norm_values(h, 3);
        for (rows, out_rows) in [(3usize, vec![2, 0]), (18, (0..16).map(|i| (i * 5 + 1) % 18).collect::<Vec<i32>>())] {
            let mut rng = Rng::new(rows as u64);
            let residual = (0..rows * h).map(|_| rng.symmetric() * 3.0).collect::<Vec<_>>();
            let outcome = interpret(
                &module,
                "qwen_dense_expand",
                &[("NW", registry::dense(DType::BF16)), ("GW", gate_repr.storage()), ("UW", up_repr.storage()), ("A", registry::dense(DType::BF16))],
                vec![
                    floats(&[rows, h], &residual),
                    activations(Act::Bf16, &[h], &norm),
                    Input::Data(gate.oracle()),
                    Input::Data(up.oracle()),
                    ints(&[out_rows.len()], &out_rows),
                    Input::F32(1e-6),
                ],
            );
            let oracle = result(&outcome, 0);
            for mapping in MAPPINGS {
                let (_, bound) = under(out_rows.len(), || dense_expand_reference(act, &gate, &up, &residual, &norm, &out_rows, 1e-6));
                let native = dense_expand_native(&device, act, &gate, &up, &residual, rows, &norm, &out_rows, 1e-6, mapping);
                assert_within(&format!("dense_expand {gate_repr:?}/{up_repr:?} rows {rows} {mapping:?}"), &native, &oracle, &bound);
            }
        }
    }
}

#[test]
fn dense_expand_matches_the_host_reference_across_row_classes() {
    let Some(device) = device() else { return };
    let (h, f) = (512, 1000);
    for (act, gate_repr, up_repr) in [
        (Act::Bf16, Repr::Q4k, Repr::Q4k),
        (Act::Bf16, Repr::Q5k, Repr::Q6k),
        (Act::Bf16, Repr::Q8, Repr::Bf16),
        (Act::F16, Repr::Q4k, Repr::Q4k),
    ] {
        let gate = weight(gate_repr, f, h, 21, 1.0);
        let up = weight(up_repr, f, h, 22, 1.0);
        let norm = norm_values(h, 4);
        for rows in ROW_CLASSES {
            let mut rng = Rng::new(rows as u64 + 7);
            let residual = (0..rows * h).map(|_| rng.symmetric() * 3.0).collect::<Vec<_>>();
            let out_rows = (0..rows).map(|i| ((i * 7 + 3) % rows) as i32).collect::<Vec<_>>();
            for mapping in MAPPINGS {
                let (expected, bound) = under(out_rows.len(), || dense_expand_reference(act, &gate, &up, &residual, &norm, &out_rows, 1e-6));
                let native = dense_expand_native(&device, act, &gate, &up, &residual, rows, &norm, &out_rows, 1e-6, mapping);
                assert_within(&format!("dense_expand {act:?} {gate_repr:?}/{up_repr:?} M {rows} {mapping:?}"), &native, &expected, &bound);
            }
        }
    }
}

fn dense_output_native(device: &Device, act: Act, down: &Weight, residual: &[f32], rows: usize, product: &[f32],
    out_rows: &[i32], mapping: Mapping) -> Vec<f32> {
    let (h, f) = (down.rows, down.k);
    let kernel = qwen_dense_output::native_for_device_with(
        device,
        qwen_dense_output::Elements { A: act.element(), DW: down.repr.element() },
        &split_specialization(&[("H", h), ("F", f)], mapping),
    )
    .unwrap();
    let out = kernel
        .call(qwen_dense_output::Args {
            residual: &f32_tensor(device, &[rows, h], residual),
            product: &act_tensor(device, act, &[out_rows.len(), f], product),
            down_weight: &down.tensor(device),
            out_rows: &i32_tensor(device, &[out_rows.len()], out_rows),
        })
        .unwrap()
        .value;
    read_f32(&out)
}

fn dense_output_reference(act: Act, down: &Weight, residual: &[f32], product: &[f32], out_rows: &[i32]) -> (Vec<f32>, Vec<f32>) {
    let (h, f) = (down.rows, down.k);
    let mut expected = Vec::new();
    let mut bound = Vec::new();
    for (r, row) in out_rows.iter().enumerate() {
        for n in 0..h {
            let (acc, magnitude) = dot(&product[r * f..(r + 1) * f], down.row(n));
            expected.push(residual[*row as usize * h + n] + act.round(acc));
            bound.push(rounded_bound(act, acc, magnitude));
        }
    }
    (expected, bound)
}

#[test]
fn dense_output_matches_its_portable_body_and_the_host_reference() {
    let Some(device) = device() else { return };
    let module = module();
    let act = Act::Bf16;
    for repr in [Repr::Q4k, Repr::Q5k, Repr::Q6k, Repr::Q8, Repr::Bf16] {
        // Portable body at small shapes.
        let (h, f) = (40, 256);
        let down = weight(repr, h, f, 31, 1.0);
        for (rows, out_rows) in [(3usize, vec![1, 2]), (20, (0..16).map(|i| (i * 3 + 2) % 20).collect::<Vec<i32>>())] {
            let mut rng = Rng::new(rows as u64);
            let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let product = (0..out_rows.len() * f).map(|_| act.round(rng.symmetric())).collect::<Vec<_>>();
            let outcome = interpret(
                &module,
                "qwen_dense_output",
                &[("A", registry::dense(DType::BF16)), ("DW", repr.storage())],
                vec![floats(&[rows, h], &residual), activations(act, &[out_rows.len(), f], &product), Input::Data(down.oracle()), ints(&[out_rows.len()], &out_rows)],
            );
            let oracle = result(&outcome, 0);
            for mapping in MAPPINGS {
                let (_, bound) = under(out_rows.len(), || dense_output_reference(act, &down, &residual, &product, &out_rows));
                let native = dense_output_native(&device, act, &down, &residual, rows, &product, &out_rows, mapping);
                assert_within(&format!("dense_output portable {repr:?} rows {rows} {mapping:?}"), &native, &oracle, &bound);
            }
        }
        // Host reference over the row classes, with a row tail (H = 300).
        let (h, f) = (300, 1024);
        let down = weight(repr, h, f, 32, 1.0);
        for rows in ROW_CLASSES {
            let mut rng = Rng::new(rows as u64 + 100);
            let residual = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let out_rows = (0..rows).map(|i| ((i * 5 + 1) % rows) as i32).collect::<Vec<_>>();
            let product = (0..rows * f).map(|_| act.round(rng.symmetric())).collect::<Vec<_>>();
            for mapping in MAPPINGS {
                let (expected, bound) = under(out_rows.len(), || dense_output_reference(act, &down, &residual, &product, &out_rows));
                let native = dense_output_native(&device, act, &down, &residual, rows, &product, &out_rows, mapping);
                assert_within(&format!("dense_output {repr:?} M {rows} {mapping:?}"), &native, &expected, &bound);
            }
        }
    }
}

/// The decode and verify row classes at the pinned Qwen3.5 4B geometry
/// (H 2560, F 9216) with the decode mapping: every M from 1 to 16 and the
/// first GEMM class 17, against the host reference. The GEMV class (M below
/// BATCH_FROM) must reproduce each row's M = 1 result bit for bit, so a
/// speculative verify row computes exactly what plain decode computes.
#[test]
fn dense_entries_at_4b_geometry_match_the_host_reference_and_single_rows() {
    let Some(device) = device() else { return };
    let act = Act::Bf16;
    let (h, f) = (2560, 9216);
    let mapping = gemm_mapping(64, 64, 1);
    let gate = weight(Repr::Q4k, f, h, 71, 1.0);
    let up = weight(Repr::Q4k, f, h, 72, 1.0);
    let norm = norm_values(h, 8);
    let down6 = weight(Repr::Q6k, h, f, 73, 1.0);
    let down4 = weight(Repr::Q4k, h, f, 74, 1.0);
    let rows_max = 17usize;
    let mut rng = Rng::new(75);
    let residual = (0..rows_max * h).map(|_| rng.symmetric() * 3.0).collect::<Vec<_>>();
    let product = (0..rows_max * f).map(|_| act.round(rng.symmetric())).collect::<Vec<_>>();
    let expand_single = (0..rows_max)
        .map(|r| dense_expand_native(&device, act, &gate, &up, &residual, rows_max, &norm, &[r as i32], 1e-6, mapping))
        .collect::<Vec<_>>();
    let output_single = |down: &Weight| {
        (0..rows_max)
            .map(|r| dense_output_native(&device, act, down, &residual, rows_max, &product[r * f..(r + 1) * f], &[r as i32], mapping))
            .collect::<Vec<_>>()
    };
    let (single6, single4) = (output_single(&down6), output_single(&down4));
    for rows in 1..=rows_max {
        let out_rows = (0..rows as i32).collect::<Vec<_>>();
        let (expected, bound) = under(rows, || dense_expand_reference(act, &gate, &up, &residual, &norm, &out_rows, 1e-6));
        let native = dense_expand_native(&device, act, &gate, &up, &residual, rows_max, &norm, &out_rows, 1e-6, mapping);
        assert_within(&format!("dense_expand 4B M {rows}"), &native, &expected, &bound);
        for (label, down, single) in [("q6k", &down6, &single6), ("q4k", &down4, &single4)] {
            let (expected, bound) = under(rows, || dense_output_reference(act, down, &residual, &product[..rows * f], &out_rows));
            let output = dense_output_native(&device, act, down, &residual, rows_max, &product[..rows * f], &out_rows, mapping);
            assert_within(&format!("dense_output {label} 4B M {rows}"), &output, &expected, &bound);
            if (rows as u64) < mapping.batch_from {
                for r in 0..rows {
                    assert!(output[r * h..(r + 1) * h].iter().zip(&single[r]).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "dense_output {label} 4B M {rows} row {r} differs from its M = 1 result");
                }
            }
        }
        if (rows as u64) < mapping.batch_from {
            for r in 0..rows {
                assert!(native[r * f..(r + 1) * f].iter().zip(&expand_single[r]).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "dense_expand 4B M {rows} row {r} differs from its M = 1 result");
            }
        }
    }
}

#[test]
fn recurrent_project_matches_its_portable_body_and_the_host_reference() {
    let Some(device) = device() else { return };
    let module = module();
    let act = Act::Bf16;
    // A small geometry over every mapping, then the pinned 4B geometry over
    // the decode and verify rows with the decode mapping.
    for (h, nk, nv, w, four_b) in [(512usize, 2usize, 4usize, 32usize, false), (2560, 16, 32, 128, true)] {
    let rows_of = [(2 * nk + nv) * w, nv * w, nv, nv];
    let total: usize = rows_of.iter().sum();
    let reprs = [Repr::Q5k, Repr::Q4k, Repr::Q8, Repr::Q8];
    let weights = (0..4).map(|i| weight(reprs[i], rows_of[i], h, 40 + i as u64, 1.0)).collect::<Vec<_>>();
    let norm = norm_values(h, 5);
    let reference = |hidden: &[f32], rows: usize| {
        let mut expected = Vec::new();
        let mut bound = Vec::new();
        for r in 0..rows {
            let (x, slack) = rms_row(act, &hidden[r * h..(r + 1) * h], &norm, 1e-6);
            for wt in &weights {
                for n in 0..wt.rows {
                    let (acc, magnitude) = dot(&x, wt.row(n));
                    expected.push(act.round(acc));
                    bound.push(rounded_bound(act, acc, magnitude) + slack_dot(&slack, wt.row(n)));
                }
            }
        }
        (expected, bound)
    };
    let native = |hidden: &[f32], rows: usize, mapping: Mapping| {
        let kernel = qwen_recurrent_project::native_for_device_with(
            &device,
            qwen_recurrent_project::Elements {
                NW: Element::bf16(),
                QW: reprs[0].element(),
                GW: reprs[1].element(),
                AW: reprs[2].element(),
                BW: reprs[3].element(),
                A: act.element(),
            },
            &specialization(&[("H", h), ("NK", nk), ("NV", nv), ("W", w)], mapping),
        )
        .unwrap();
        let out = kernel
            .call(qwen_recurrent_project::Args {
                hidden: &f32_tensor(&device, &[rows, h], hidden),
                input_norm: &bf16_norm(&device, &norm),
                qkv_weight: &weights[0].tensor(&device),
                gate_weight: &weights[1].tensor(&device),
                alpha_weight: &weights[2].tensor(&device),
                beta_weight: &weights[3].tensor(&device),
                epsilon: 1e-6,
            })
            .unwrap()
            .value;
        read_act(act, &out)
    };
    let (row_classes, mappings) = checked_classes(four_b);
    let portable_rows: &[usize] = if four_b { &[] } else { &[3, 17] };
    for &rows in portable_rows {
        let mut rng = Rng::new(rows as u64 + 3);
        let hidden = (0..rows * h).map(|_| rng.symmetric() * 2.0).collect::<Vec<_>>();
        let outcome = interpret(
            &module,
            "qwen_recurrent_project",
            &[("NW", registry::dense(DType::BF16)), ("QW", reprs[0].storage()), ("GW", reprs[1].storage()), ("AW", reprs[2].storage()), ("BW", reprs[3].storage()), ("A", registry::dense(DType::BF16))],
            vec![
                floats(&[rows, h], &hidden),
                activations(Act::Bf16, &[h], &norm),
                Input::Data(weights[0].oracle()),
                Input::Data(weights[1].oracle()),
                Input::Data(weights[2].oracle()),
                Input::Data(weights[3].oracle()),
                Input::F32(1e-6),
            ],
        );
        let oracle = result(&outcome, 0);
        assert_eq!(oracle.len(), rows * total);
        for mapping in MAPPINGS {
            let (_, bound) = under(rows, || reference(&hidden, rows));
            assert_within(&format!("recurrent_project portable rows {rows} {mapping:?}"), &native(&hidden, rows, mapping), &oracle, &bound);
        }
    }
    for &rows in &row_classes {
        let mut rng = Rng::new(rows as u64 + 9);
        let hidden = (0..rows * h).map(|_| rng.symmetric() * 2.0).collect::<Vec<_>>();
        for &mapping in &mappings {
            let (expected, bound) = under(rows, || reference(&hidden, rows));
            assert_within(&format!("recurrent_project H {h} M {rows} {mapping:?}"), &native(&hidden, rows, mapping), &expected, &bound);
        }
    }
    }
}

#[test]
fn recurrent_output_matches_its_portable_body_and_the_host_reference() {
    let Some(device) = device() else { return };
    let module = module();
    let act = Act::Bf16;
    // A small geometry over every representation and mapping, then the pinned
    // 4B geometry (q5k) over the decode and verify rows.
    for (h, nk, nv, w, four_b) in [(300usize, 1usize, 2usize, 128usize, false), (2560, 16, 32, 128, true)] {
    let k = nv * w;
    let total = (2 * nk + nv) * w + nv * w + 2 * nv;
    let z0 = (2 * nk + nv) * w;
    let (row_classes, mappings) = checked_classes(four_b);
    let reprs: &[Repr] = if four_b { &[Repr::Q5k] } else { &[Repr::Q5k, Repr::Q6k, Repr::Bf16] };
    for &repr in reprs {
        let out_weight = weight(repr, h, k, 50, 1.0);
        let norm = norm_values(w, 6);
        let case = |rows: usize| {
            let mut rng = Rng::new(rows as u64 + 11);
            let hidden = (0..rows * h).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let projection = (0..rows * total).map(|_| act.round(rng.symmetric() * 3.0)).collect::<Vec<_>>();
            let mixed = (0..rows * k).map(|_| act.round(rng.symmetric() * 0.3)).collect::<Vec<_>>();
            (hidden, projection, mixed)
        };
        let reference = |hidden: &[f32], projection: &[f32], mixed: &[f32], rows: usize| {
            let mut expected = Vec::new();
            let mut bound = Vec::new();
            for r in 0..rows {
                let mut gated = vec![0f32; k];
                let mut slack = vec![0f32; k];
                for head in 0..nv {
                    let values = &mixed[r * k + head * w..r * k + (head + 1) * w];
                    let squares = values.iter().fold(0f32, |s, v| v.mul_add(*v, s));
                    let inverse = 1.0 / (squares / w as f32 + 1e-6).sqrt();
                    for c in 0..w {
                        let exact = values[c] * inverse * norm[c];
                        let normalized = act.round(exact);
                        let activated = act.round(silu(projection[r * total + z0 + head * w + c]));
                        gated[head * w + c] = act.round(normalized * activated);
                        let flip = (act.round(exact * (1.0 + 1e-5)) - act.round(exact * (1.0 - 1e-5))).abs();
                        slack[head * w + c] = (flip * activated.abs() + gated[head * w + c].abs() * act.ulp()) * 1.01;
                    }
                }
                for n in 0..h {
                    let (acc, magnitude) = dot(&gated, out_weight.row(n));
                    expected.push(hidden[r * h + n] + act.round(acc));
                    bound.push(rounded_bound(act, acc, magnitude) + slack_dot(&slack, out_weight.row(n)) * 0.25);
                }
            }
            (expected, bound)
        };
        let native = |hidden: &[f32], projection: &[f32], mixed: &[f32], rows: usize, mapping: Mapping| {
            let kernel = qwen_recurrent_output::native_for_device_with(
                &device,
                qwen_recurrent_output::Elements { RN: Element::bf16(), OW: repr.element(), A: act.element() },
                &split_specialization(&[("H", h), ("NK", nk), ("NV", nv), ("W", w)], mapping),
            )
            .unwrap();
            let out = kernel
                .call(qwen_recurrent_output::Args {
                    hidden: &f32_tensor(&device, &[rows, h], hidden),
                    mixed: &act_tensor(&device, act, &[rows, nv, w], mixed),
                    projection: &act_tensor(&device, act, &[rows, total], projection),
                    recurrent_norm: &bf16_norm(&device, &norm),
                    output_weight: &out_weight.tensor(&device),
                    epsilon: 1e-6,
                })
                .unwrap()
                .value;
            read_f32(&out)
        };
        let portable_rows: &[usize] = if four_b { &[] } else { &[2, 12] };
        for &rows in portable_rows {
            let (hidden, projection, mixed) = case(rows);
            let outcome = interpret(
                &module,
                "qwen_recurrent_output",
                &[("RN", registry::dense(DType::BF16)), ("OW", repr.storage()), ("A", registry::dense(DType::BF16))],
                vec![
                    floats(&[rows, h], &hidden),
                    activations(act, &[rows, nv, w], &mixed),
                    activations(act, &[rows, total], &projection),
                    activations(Act::Bf16, &[w], &norm),
                    Input::Data(out_weight.oracle()),
                    Input::F32(1e-6),
                ],
            );
            let oracle = result(&outcome, 0);
            for mapping in MAPPINGS {
                let (_, bound) = under(rows, || reference(&hidden, &projection, &mixed, rows));
                assert_within(&format!("recurrent_output portable {repr:?} rows {rows} {mapping:?}"),
                    &native(&hidden, &projection, &mixed, rows, mapping), &oracle, &bound);
            }
        }
        for &rows in &row_classes {
            let (hidden, projection, mixed) = case(rows);
            for &mapping in &mappings {
                let (expected, bound) = under(rows, || reference(&hidden, &projection, &mixed, rows));
                assert_within(&format!("recurrent_output {repr:?} H {h} M {rows} {mapping:?}"),
                    &native(&hidden, &projection, &mixed, rows, mapping), &expected, &bound);
            }
        }
    }
    }
}

#[test]
fn attention_projections_match_their_portable_bodies_and_the_host_reference() {
    let Some(device) = device() else { return };
    let module = module();
    let act = Act::Bf16;
    // A small geometry over every mapping, then the pinned 4B geometry over
    // the decode and verify rows with the decode mapping.
    for (d, kv, g, w, four_b) in [(512usize, 2usize, 2usize, 64usize, false), (2560, 4, 4, 256, true)] {
    let (row_classes, mappings) = checked_classes(four_b);
    let rows_of = [kv * g * 2 * w, kv * w, kv * w];
    let reprs = [Repr::Q4k, Repr::Q4k, Repr::Q6k];
    let weights = (0..3).map(|i| weight(reprs[i], rows_of[i], d, 60 + i as u64, 1.0)).collect::<Vec<_>>();
    let norm = norm_values(d, 7);
    let query_norm = vec![1.0f32; w];
    let project = |hidden: &[f32], rows: usize, mapping: Mapping| {
        let kernel = qwen_attention_project::native_for_device_with(
            &device,
            qwen_attention_project::Elements { NW: Element::bf16(), QW: reprs[0].element(), KW: reprs[1].element(), VW: reprs[2].element(), A: act.element() },
            &specialization(&[("D", d), ("KV", kv), ("G", g), ("W", w)], mapping),
        )
        .unwrap();
        let out = kernel
            .call(qwen_attention_project::Args {
                hidden: &f32_tensor(&device, &[rows, d], hidden),
                input_norm: &bf16_norm(&device, &norm),
                query_norm: &f32_tensor(&device, &[w], &query_norm),
                query_gate_weight: &weights[0].tensor(&device),
                key_weight: &weights[1].tensor(&device),
                value_weight: &weights[2].tensor(&device),
                epsilon: 1e-6,
            })
            .unwrap();
        [read_act(act, &out.r0), read_act(act, &out.r1), read_act(act, &out.r2)]
    };
    let reference = |hidden: &[f32], rows: usize| {
        let mut out = [(Vec::new(), Vec::new()), (Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
        for r in 0..rows {
            let (x, slack) = rms_row(act, &hidden[r * d..(r + 1) * d], &norm, 1e-6);
            for (i, wt) in weights.iter().enumerate() {
                for n in 0..wt.rows {
                    let (acc, magnitude) = dot(&x, wt.row(n));
                    out[i].0.push(act.round(acc));
                    out[i].1.push(rounded_bound(act, acc, magnitude) + slack_dot(&slack, wt.row(n)));
                }
            }
        }
        out
    };
    let portable_rows: &[usize] = if four_b { &[] } else { &[2, 16] };
    for &rows in portable_rows {
        let mut rng = Rng::new(rows as u64 + 5);
        let hidden = (0..rows * d).map(|_| rng.symmetric() * 2.0).collect::<Vec<_>>();
        let outcome = interpret(
            &module,
            "qwen_attention_project",
            &[("NW", registry::dense(DType::BF16)), ("QW", reprs[0].storage()), ("KW", reprs[1].storage()), ("VW", reprs[2].storage()), ("A", registry::dense(DType::BF16))],
            vec![
                floats(&[rows, d], &hidden),
                activations(Act::Bf16, &[d], &norm),
                floats(&[w], &query_norm),
                Input::Data(weights[0].oracle()),
                Input::Data(weights[1].oracle()),
                Input::Data(weights[2].oracle()),
                Input::F32(1e-6),
            ],
        );
        for mapping in MAPPINGS {
            let expected = under(rows, || reference(&hidden, rows));
            let native = project(&hidden, rows, mapping);
            for i in 0..3 {
                assert_within(&format!("attention_project[{i}] portable rows {rows} {mapping:?}"), &native[i], &result(&outcome, i), &expected[i].1);
            }
        }
    }
    for &rows in &row_classes {
        let mut rng = Rng::new(rows as u64 + 55);
        let hidden = (0..rows * d).map(|_| rng.symmetric() * 2.0).collect::<Vec<_>>();
        for &mapping in &mappings {
            let expected = under(rows, || reference(&hidden, rows));
            let native = project(&hidden, rows, mapping);
            for i in 0..3 {
                assert_within(&format!("attention_project[{i}] D {d} M {rows} {mapping:?}"), &native[i], &expected[i].0, &expected[i].1);
            }
        }
    }
    // Output projection over Q = KV * G gated heads.
    let q = kv * g;
    let reprs: &[Repr] = if four_b { &[Repr::Q4k] } else { &[Repr::Q4k, Repr::Q8, Repr::Bf16] };
    for &repr in reprs {
        let (d_out, k) = (if four_b { d } else { 300usize }, q * w);
        let out_weight = weight(repr, d_out, k, 70, 1.0);
        let native = |hidden: &[f32], gated: &[f32], rows: usize, mapping: Mapping| {
            let kernel = attention_output::native_for_device_with(
                &device,
                attention_output::Elements { A: act.element(), OW: repr.element() },
                &split_specialization(&[("D", d_out), ("Q", q), ("W", w)], mapping),
            )
            .unwrap();
            let out = kernel
                .call(attention_output::Args {
                    hidden: &f32_tensor(&device, &[rows, d_out], hidden),
                    gated: &act_tensor(&device, act, &[rows, q, w], gated),
                    output_weight: &out_weight.tensor(&device),
                })
                .unwrap()
                .value;
            read_f32(&out)
        };
        let reference = |hidden: &[f32], gated: &[f32], rows: usize| {
            let mut expected = Vec::new();
            let mut bound = Vec::new();
            for r in 0..rows {
                for n in 0..d_out {
                    let (acc, magnitude) = dot(&gated[r * k..(r + 1) * k], out_weight.row(n));
                    expected.push(hidden[r * d_out + n] + act.round(acc));
                    bound.push(rounded_bound(act, acc, magnitude));
                }
            }
            (expected, bound)
        };
        for &rows in &row_classes {
            let mut rng = Rng::new(rows as u64 + 77);
            let hidden = (0..rows * d_out).map(|_| rng.symmetric()).collect::<Vec<_>>();
            let gated = (0..rows * k).map(|_| act.round(rng.symmetric())).collect::<Vec<_>>();
            if !four_b && (rows == 3 || rows == 16) {
                let (_, bound) = under(rows, || reference(&hidden, &gated, rows));
                let outcome = interpret(
                    &module,
                    "attention_output",
                    &[("A", registry::dense(DType::BF16)), ("OW", repr.storage())],
                    vec![floats(&[rows, d_out], &hidden), activations(act, &[rows, q, w], &gated), Input::Data(out_weight.oracle())],
                );
                let oracle = result(&outcome, 0);
                assert_within(&format!("attention_output portable {repr:?} rows {rows}"), &native(&hidden, &gated, rows, MAPPINGS[0]), &oracle, &bound);
            }
            for &mapping in &mappings {
                let (expected, bound) = under(rows, || reference(&hidden, &gated, rows));
                assert_within(&format!("attention_output {repr:?} D {d_out} M {rows} {mapping:?}"), &native(&hidden, &gated, rows, mapping), &expected, &bound);
            }
        }
    }
    }
}

#[test]
fn readout_entries_match_their_portable_bodies_and_the_host_reference() {
    let Some(device) = device() else { return };
    let module = module();
    let act = Act::Bf16;
    let (v, d) = (1000usize, 512usize);
    for repr in [Repr::Q6k, Repr::Q4k, Repr::Q8] {
        let head = weight(repr, v, d, 80, 1.0);
        let norm = norm_values(d, 8);
        let selected = (0..37).map(|i| ((i * 37 + 11) % v) as i32).collect::<Vec<_>>();
        for outputs in ROW_CLASSES {
            let rows = outputs.max(2) + 1;
            let mut rng = Rng::new(outputs as u64 + 13);
            let hidden = (0..rows * d).map(|_| rng.symmetric() * 2.0).collect::<Vec<_>>();
            let out_rows = (0..outputs).map(|i| ((i * 3 + 2) % rows) as i32).collect::<Vec<_>>();
            let mut features = Vec::new();
            let mut feature_bound = Vec::new();
            let mut logits = Vec::new();
            let mut logit_bound = Vec::new();
            let mut chosen = Vec::new();
            let mut chosen_bound = Vec::new();
            for row in &out_rows {
                let source = *row as usize;
                let (x, slack) = rms_row(act, &hidden[source * d..(source + 1) * d], &norm, 1e-6);
                features.extend_from_slice(&x);
                feature_bound.extend_from_slice(&slack);
                for n in 0..v {
                    let (acc, magnitude) = dot(&x, head.row(n));
                    logits.push(acc);
                    logit_bound.push((magnitude, 1e-7 + slack_dot(&slack, head.row(n))));
                }
                for n in &selected {
                    let (acc, magnitude) = dot(&x, head.row(*n as usize));
                    chosen.push(acc);
                    chosen_bound.push((magnitude, 1e-7 + slack_dot(&slack, head.row(*n as usize))));
                }
            }
            let hidden_tensor = f32_tensor(&device, &[rows, d], &hidden);
            let rows_tensor = i32_tensor(&device, &[outputs], &out_rows);
            let features_native = readout_features_rows::native_for_device_with(
                &device,
                readout_features_rows::Elements { NW: Element::bf16(), A: act.element() },
                &NativeSpecialization::new().with_static("D", d as u64),
            )
            .unwrap()
            .call(readout_features_rows::Args { hidden: &hidden_tensor, norm: &bf16_norm(&device, &norm), out_rows: &rows_tensor, epsilon: 1e-6 })
            .unwrap()
            .value;
            assert_within(&format!("features_rows O {outputs}"), &read_act(act, &features_native), &features, &feature_bound);
            let bounds = |terms: &[(f32, f32)]| {
                terms.iter().map(|(magnitude, rest)| reassociation(outputs) * magnitude + rest).collect::<Vec<_>>()
            };
            for mapping in MAPPINGS {
                let head_native = readout_head_rows::native_for_device_with(
                    &device,
                    readout_head_rows::Elements { NW: Element::bf16(), OW: repr.element(), A: act.element() },
                    &specialization(&[("V", v), ("D", d)], mapping),
                )
                .unwrap()
                .call(readout_head_rows::Args {
                    hidden: &hidden_tensor,
                    norm: &bf16_norm(&device, &norm),
                    weight: &head.tensor(&device),
                    out_rows: &rows_tensor,
                    epsilon: 1e-6,
                })
                .unwrap()
                .value;
                assert_within(&format!("head_rows {repr:?} O {outputs} {mapping:?}"), &read_f32(&head_native), &logits, &bounds(&logit_bound));
                let selected_native = readout_selected_rows::native_for_device_with(
                    &device,
                    readout_selected_rows::Elements { NW: Element::bf16(), OW: repr.element(), A: act.element() },
                    &specialization(&[("V", v), ("D", d)], mapping),
                )
                .unwrap()
                .call(readout_selected_rows::Args {
                    hidden: &hidden_tensor,
                    norm: &bf16_norm(&device, &norm),
                    weight: &head.tensor(&device),
                    out_rows: &rows_tensor,
                    selected: &i32_tensor(&device, &[selected.len()], &selected),
                    epsilon: 1e-6,
                })
                .unwrap()
                .value;
                assert_within(&format!("selected_rows {repr:?} O {outputs} {mapping:?}"), &read_f32(&selected_native), &chosen, &bounds(&chosen_bound));
            }
            if outputs == 2 && repr == Repr::Q6k {
                let bindings = [("NW", registry::dense(DType::BF16)), ("OW", repr.storage()), ("A", registry::dense(DType::BF16))];
                let inputs = || {
                    vec![
                        floats(&[rows, d], &hidden),
                        activations(Act::Bf16, &[d], &norm),
                        Input::Data(head.oracle()),
                        ints(&[outputs], &out_rows),
                    ]
                };
                let mut head_inputs = inputs();
                head_inputs.push(Input::F32(1e-6));
                let oracle = result(&interpret(&module, "readout_head_rows", &bindings, head_inputs), 0);
                assert_within("head_rows portable", &oracle, &logits, &bounds(&logit_bound));
                let mut selected_inputs = inputs();
                selected_inputs.push(ints(&[selected.len()], &selected));
                selected_inputs.push(Input::F32(1e-6));
                let oracle = result(&interpret(&module, "readout_selected_rows", &bindings, selected_inputs), 0);
                assert_within("selected_rows portable", &oracle, &chosen, &bounds(&chosen_bound));
                let oracle = result(
                    &interpret(
                        &module,
                        "readout_features_rows",
                        &[("NW", registry::dense(DType::BF16)), ("A", registry::dense(DType::BF16))],
                        vec![floats(&[rows, d], &hidden), activations(Act::Bf16, &[d], &norm), ints(&[outputs], &out_rows), Input::F32(1e-6)],
                    ),
                    0,
                );
                assert_within("features_rows portable", &oracle, &features, &feature_bound);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Timing (indicative on a shared GPU; evidence only on a quiet M4 Pro).

/// Random bytes in `repr`'s rows16 layout: timing does not depend on the
/// values, so no host decode is built.
fn noise_weight(device: &Device, repr: Repr, rows: usize, k: usize, seed: u64) -> Tensor {
    let stride = planes(repr, k).stride;
    let mut bytes = vec![0u8; stride * rows];
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    for chunk in bytes.chunks_exact_mut(8) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        chunk.copy_from_slice(&state.to_le_bytes());
    }
    Tensor::from_host(device, repr.element(), &[rows as u64, k as u64], &bytes).unwrap()
}

fn weight_bytes(repr: Repr, rows: usize, k: usize) -> usize {
    planes(repr, k).stride * rows
}

/// Enough copies of a weight set that one rotation streams from DRAM rather
/// than the system-level cache.
fn copies(bytes: usize) -> usize {
    (256usize << 20).div_ceil(bytes).clamp(1, 24)
}

fn uniform_values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut rng = Rng::new(seed);
    (0..count).map(|_| rng.symmetric() * scale).collect()
}

/// One timed configuration: the per-call device time, the weight bytes one
/// call streams and its multiply-add work.
fn report(label: &str, m: usize, mapping: Mapping, seconds: f64, bytes: usize, macs: usize) {
    eprintln!(
        "timing {label} M {m} (SG {}, R {}, LANES {}, BATCH_FROM {}, BATCH SG {}, BATCH R {}, TILE_M {}, TILE_N {}, SPLIT {}): {:.1} us, {:.1} GB/s, {:.2} TFLOP/s",
        mapping.simdgroups,
        mapping.rows,
        mapping.lanes,
        mapping.batch_from,
        mapping.batch_simdgroups,
        mapping.batch_rows,
        mapping.tile_m,
        mapping.tile_n,
        mapping.split,
        seconds * 1e6,
        bytes as f64 / seconds / 1e9,
        2.0 * macs as f64 / seconds / 1e12
    );
}

const TIMING_OPTIONS: seismic::MeasureOptions = seismic::MeasureOptions { samples: 9, min_sample_seconds: 0.02 };

fn timing_rows() -> Vec<usize> {
    std::env::var("PROJECTION_TIMING_ROWS")
        .map(|rows| rows.split(',').map(|m| m.trim().parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![1, 4, 8, 32, 128, 512])
}

fn timing_mappings(m: usize) -> Vec<Mapping> {
    if m <= 2 {
        let mut all = Vec::new();
        for simdgroups in [4, 8, 16, 32] {
            for rows in [1, 2, 4] {
                for lanes in [32, 16] {
                    all.push(Mapping { simdgroups, rows, lanes, ..gemm_mapping(64, 64, 1) });
                }
            }
        }
        all
    } else if m <= 16 {
        // The batched GEMV, and up to 8 rows the GEMV serving the class (BATCH_FROM 9).
        let batched = [(4, 1), (4, 2), (8, 1), (8, 2), (16, 1), (32, 1), (4, 4)].map(|(batch_simdgroups, batch_rows)| {
            Mapping { batch_from: 3, batch_simdgroups, batch_rows, ..gemm_mapping(64, 64, 1) }
        });
        let gemv = [(8, 1, 16), (16, 1, 16), (32, 1, 16), (8, 1, 32), (16, 1, 32), (8, 2, 16), (16, 2, 16)].map(
            |(simdgroups, rows, lanes)| Mapping { simdgroups, rows, lanes, batch_from: 9, ..gemm_mapping(64, 64, 1) },
        );
        batched.into_iter().chain(gemv.into_iter().filter(|_| m <= 8)).collect()
    } else {
        [(64, 64), (128, 64), (64, 128), (32, 128), (32, 64), (128, 128)]
            .map(|(tile_m, tile_n)| gemm_mapping(tile_m, tile_n, 1))
            .to_vec()
    }
}

/// The mappings of the split-K output projections: every GEMM tile with each
/// split factor.
fn split_mappings(m: usize) -> Vec<Mapping> {
    timing_mappings(m)
        .into_iter()
        .flat_map(|mapping| {
            let splits: &[u64] = if m <= 16 { &[1] } else { &[1, 2, 4] };
            splits.iter().map(move |split| Mapping { split: *split, ..mapping })
        })
        .collect()
}

fn timing_selected(name: &str) -> bool {
    std::env::var("PROJECTION_TIMING_ENTRIES").map_or(true, |list| list.split(',').any(|entry| entry == name))
}

/// Device time of every K1 entry at Qwen3.5 4B and 35B-A3B shapes over the
/// row classes: `cargo test --release -p magnitude-model-kernels --test
/// projection -- --ignored --nocapture timing`. `PROJECTION_TIMING_ROWS`
/// (comma-separated M) and `PROJECTION_TIMING_ENTRIES` (comma-separated case
/// labels) narrow the run. Weights rotate over copies so every call streams
/// them from DRAM.
#[test]
#[ignore]
fn timing() {
    let Some(device) = device() else { return };
    let act = Act::Bf16;
    let a = act.element();
    let rows_list = timing_rows();
    let max_m = *rows_list.iter().max().unwrap();

    // Plain prologue, residual epilogue: the down projection.
    for (label, repr, h, f) in [
        ("dense_output 4b q6k 2560x9216", Repr::Q6k, 2560usize, 9216usize),
        ("dense_output 4b q4k 2560x9216", Repr::Q4k, 2560, 9216),
    ] {
        if !timing_selected(label.split(' ').next().unwrap()) {
            continue;
        }
        let bytes = weight_bytes(repr, h, f);
        let weights = (0..copies(bytes)).map(|i| noise_weight(&device, repr, h, f, i as u64 + 1)).collect::<Vec<_>>();
        let residual = f32_tensor(&device, &[max_m, h], &uniform_values(max_m * h, 1, 1.0));
        for &m in &rows_list {
            let product = act_tensor(&device, act, &[m, f], &uniform_values(m * f, 2, 1.0));
            let out_rows = i32_tensor(&device, &[m], &(0..m as i32).collect::<Vec<_>>());
            for mapping in split_mappings(m) {
                let kernel = qwen_dense_output::native_for_device_with(
                    &device,
                    qwen_dense_output::Elements { A: a, DW: repr.element() },
                    &split_specialization(&[("H", h), ("F", f)], mapping),
                )
                .unwrap();
                let rotation = weights
                    .iter()
                    .map(|weight| qwen_dense_output::Args { residual: &residual, product: &product, down_weight: weight, out_rows: &out_rows })
                    .collect();
                let measured = kernel.measure(rotation, &TIMING_OPTIONS).unwrap();
                report(label, m, mapping, measured.median, bytes, m * h * f);
            }
        }
    }

    // RMS prologue, paired SiLU epilogue: gate+up.
    for (label, repr, h, f) in [("dense_expand 4b q4k 2x9216x2560", Repr::Q4k, 2560usize, 9216usize)] {
        if !timing_selected(label.split(' ').next().unwrap()) {
            continue;
        }
        let bytes = 2 * weight_bytes(repr, f, h);
        let weights = (0..copies(bytes))
            .map(|i| (noise_weight(&device, repr, f, h, 2 * i as u64 + 1), noise_weight(&device, repr, f, h, 2 * i as u64 + 2)))
            .collect::<Vec<_>>();
        let residual = f32_tensor(&device, &[max_m, h], &uniform_values(max_m * h, 3, 1.0));
        let norm = bf16_norm(&device, &norm_values(h, 3));
        for &m in &rows_list {
            let out_rows = i32_tensor(&device, &[m], &(0..m as i32).collect::<Vec<_>>());
            for mapping in timing_mappings(m) {
                let kernel = qwen_dense_expand::native_for_device_with(
                    &device,
                    qwen_dense_expand::Elements { NW: Element::bf16(), GW: repr.element(), UW: repr.element(), A: a },
                    &specialization(&[("H", h), ("F", f)], mapping),
                )
                .unwrap();
                let rotation = weights
                    .iter()
                    .map(|(gate, up)| qwen_dense_expand::Args {
                        residual: &residual,
                        norm: &norm,
                        gate_weight: gate,
                        up_weight: up,
                        out_rows: &out_rows,
                        eps: 1e-6,
                    })
                    .collect();
                let measured = kernel.measure(rotation, &TIMING_OPTIONS).unwrap();
                report(label, m, mapping, measured.median, bytes, m * 2 * f * h);
            }
        }
    }

    // Segmented RMS projection: qkv | z | alpha | beta.
    for (label, reprs, h, nk, nv, w) in [
        ("recurrent_project 4b q5k|q4k|q8|q8 2560", [Repr::Q5k, Repr::Q4k, Repr::Q8, Repr::Q8], 2560usize, 16usize, 32usize, 128usize),
        ("recurrent_project 35b q8 2048", [Repr::Q8; 4], 2048, 16, 32, 128),
    ] {
        if !timing_selected(label.split(' ').next().unwrap()) {
            continue;
        }
        let rows_of = [(2 * nk + nv) * w, nv * w, nv, nv];
        let bytes = (0..4).map(|i| weight_bytes(reprs[i], rows_of[i], h)).sum::<usize>();
        let weights = (0..copies(bytes))
            .map(|c| (0..4).map(|i| noise_weight(&device, reprs[i], rows_of[i], h, 10 * c as u64 + i as u64)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let norm = bf16_norm(&device, &norm_values(h, 4));
        let total = rows_of.iter().sum::<usize>();
        for &m in &rows_list {
            let hidden = f32_tensor(&device, &[m, h], &uniform_values(m * h, 4, 1.0));
            for mapping in timing_mappings(m) {
                let kernel = qwen_recurrent_project::native_for_device_with(
                    &device,
                    qwen_recurrent_project::Elements {
                        NW: Element::bf16(),
                        QW: reprs[0].element(),
                        GW: reprs[1].element(),
                        AW: reprs[2].element(),
                        BW: reprs[3].element(),
                        A: a,
                    },
                    &specialization(&[("H", h), ("NK", nk), ("NV", nv), ("W", w)], mapping),
                )
                .unwrap();
                let rotation = weights
                    .iter()
                    .map(|set| qwen_recurrent_project::Args {
                        hidden: &hidden,
                        input_norm: &norm,
                        qkv_weight: &set[0],
                        gate_weight: &set[1],
                        alpha_weight: &set[2],
                        beta_weight: &set[3],
                        epsilon: 1e-6,
                    })
                    .collect();
                let measured = kernel.measure(rotation, &TIMING_OPTIONS).unwrap();
                report(label, m, mapping, measured.median, bytes, m * total * h);
            }
        }
    }

    // Gated per-head RMS·SiLU(z) prologue, residual epilogue.
    for (label, repr, h, nk, nv, w) in [
        ("recurrent_output 4b q5k 2560x4096", Repr::Q5k, 2560usize, 16usize, 32usize, 128usize),
        ("recurrent_output 35b q8 2048x4096", Repr::Q8, 2048, 16, 32, 128),
    ] {
        if !timing_selected(label.split(' ').next().unwrap()) {
            continue;
        }
        let k = nv * w;
        let total = (2 * nk + nv) * w + nv * w + 2 * nv;
        let bytes = weight_bytes(repr, h, k);
        let weights = (0..copies(bytes)).map(|i| noise_weight(&device, repr, h, k, i as u64 + 5)).collect::<Vec<_>>();
        let norm = bf16_norm(&device, &norm_values(w, 5));
        for &m in &rows_list {
            let hidden = f32_tensor(&device, &[m, h], &uniform_values(m * h, 5, 1.0));
            let mixed = act_tensor(&device, act, &[m, nv, w], &uniform_values(m * k, 6, 0.3));
            let projection = act_tensor(&device, act, &[m, total], &uniform_values(m * total, 7, 2.0));
            for mapping in split_mappings(m) {
                let kernel = qwen_recurrent_output::native_for_device_with(
                    &device,
                    qwen_recurrent_output::Elements { RN: Element::bf16(), OW: repr.element(), A: a },
                    &split_specialization(&[("H", h), ("NK", nk), ("NV", nv), ("W", w)], mapping),
                )
                .unwrap();
                let rotation = weights
                    .iter()
                    .map(|weight| qwen_recurrent_output::Args {
                        hidden: &hidden,
                        mixed: &mixed,
                        projection: &projection,
                        recurrent_norm: &norm,
                        output_weight: weight,
                        epsilon: 1e-6,
                    })
                    .collect();
                let measured = kernel.measure(rotation, &TIMING_OPTIONS).unwrap();
                report(label, m, mapping, measured.median, bytes, m * h * k);
            }
        }
    }

    // Segmented RMS projection: q+gate | k | v.
    for (label, reprs, d, kv, g, w) in [
        ("attention_project 4b q4k|q4k|q6k 2560", [Repr::Q4k, Repr::Q4k, Repr::Q6k], 2560usize, 4usize, 4usize, 256usize),
        ("attention_project 35b q8 2048", [Repr::Q8; 3], 2048, 2, 8, 256),
    ] {
        if !timing_selected(label.split(' ').next().unwrap()) {
            continue;
        }
        let rows_of = [kv * g * 2 * w, kv * w, kv * w];
        let bytes = (0..3).map(|i| weight_bytes(reprs[i], rows_of[i], d)).sum::<usize>();
        let weights = (0..copies(bytes))
            .map(|c| (0..3).map(|i| noise_weight(&device, reprs[i], rows_of[i], d, 10 * c as u64 + i as u64 + 3)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let norm = bf16_norm(&device, &norm_values(d, 6));
        let query_norm = f32_tensor(&device, &[w], &vec![1.0; w]);
        let total = rows_of.iter().sum::<usize>();
        for &m in &rows_list {
            let hidden = f32_tensor(&device, &[m, d], &uniform_values(m * d, 8, 1.0));
            for mapping in timing_mappings(m) {
                let kernel = qwen_attention_project::native_for_device_with(
                    &device,
                    qwen_attention_project::Elements {
                        NW: Element::bf16(),
                        QW: reprs[0].element(),
                        KW: reprs[1].element(),
                        VW: reprs[2].element(),
                        A: a,
                    },
                    &specialization(&[("D", d), ("KV", kv), ("G", g), ("W", w)], mapping),
                )
                .unwrap();
                let rotation = weights
                    .iter()
                    .map(|set| qwen_attention_project::Args {
                        hidden: &hidden,
                        input_norm: &norm,
                        query_norm: &query_norm,
                        query_gate_weight: &set[0],
                        key_weight: &set[1],
                        value_weight: &set[2],
                        epsilon: 1e-6,
                    })
                    .collect();
                let measured = kernel.measure(rotation, &TIMING_OPTIONS).unwrap();
                report(label, m, mapping, measured.median, bytes, m * total * d);
            }
        }
    }

    // Plain prologue over gated heads, residual epilogue: o_proj.
    for (label, repr, d, q, w) in [
        ("attention_output 4b q4k 2560x4096", Repr::Q4k, 2560usize, 16usize, 256usize),
        ("attention_output 35b q8 2048x4096", Repr::Q8, 2048, 16, 256),
    ] {
        if !timing_selected(label.split(' ').next().unwrap()) {
            continue;
        }
        let k = q * w;
        let bytes = weight_bytes(repr, d, k);
        let weights = (0..copies(bytes)).map(|i| noise_weight(&device, repr, d, k, i as u64 + 9)).collect::<Vec<_>>();
        for &m in &rows_list {
            let hidden = f32_tensor(&device, &[m, d], &uniform_values(m * d, 9, 1.0));
            let gated = act_tensor(&device, act, &[m, q, w], &uniform_values(m * k, 10, 1.0));
            for mapping in split_mappings(m) {
                let kernel = attention_output::native_for_device_with(
                    &device,
                    attention_output::Elements { A: a, OW: repr.element() },
                    &split_specialization(&[("D", d), ("Q", q), ("W", w)], mapping),
                )
                .unwrap();
                let rotation = weights
                    .iter()
                    .map(|weight| attention_output::Args { hidden: &hidden, gated: &gated, output_weight: weight })
                    .collect();
                let measured = kernel.measure(rotation, &TIMING_OPTIONS).unwrap();
                report(label, m, mapping, measured.median, bytes, m * d * k);
            }
        }
    }

    // The launch floor: one threadgroup normalizing one 2560-column row (10 KB
    // read), timed like every entry above.
    if timing_selected("launch_floor") {
        let d = 2560usize;
        let hidden = f32_tensor(&device, &[1, d], &uniform_values(d, 12, 1.0));
        let norm = bf16_norm(&device, &norm_values(d, 12));
        let out_rows = i32_tensor(&device, &[1], &[0]);
        let kernel = readout_features_rows::native_for_device_with(
            &device,
            readout_features_rows::Elements { NW: Element::bf16(), A: a },
            &NativeSpecialization::new().with_static("D", d as u64),
        )
        .unwrap();
        let rotation = vec![readout_features_rows::Args { hidden: &hidden, norm: &norm, out_rows: &out_rows, epsilon: 1e-6 }];
        let measured = kernel.measure(rotation, &TIMING_OPTIONS).unwrap();
        eprintln!("timing launch_floor features_rows 1x2560: {:.1} us", measured.median * 1e6);
    }

    // The vocabulary head: RMS prologue, F32 logits.
    for (label, repr, v, d) in [
        ("head_rows 4b q6k 248320x2560", Repr::Q6k, 248_320usize, 2560usize),
        ("head_rows 35b q6k 248320x2048", Repr::Q6k, 248_320, 2048),
    ] {
        if !timing_selected(label.split(' ').next().unwrap()) {
            continue;
        }
        let bytes = weight_bytes(repr, v, d);
        let weight = noise_weight(&device, repr, v, d, 11);
        let norm = bf16_norm(&device, &norm_values(d, 11));
        for &m in rows_list.iter().filter(|m| **m <= 128) {
            let hidden = f32_tensor(&device, &[m, d], &uniform_values(m * d, 11, 1.0));
            let out_rows = i32_tensor(&device, &[m], &(0..m as i32).collect::<Vec<_>>());
            for mapping in timing_mappings(m) {
                let kernel = readout_head_rows::native_for_device_with(
                    &device,
                    readout_head_rows::Elements { NW: Element::bf16(), OW: repr.element(), A: a },
                    &specialization(&[("V", v), ("D", d)], mapping),
                )
                .unwrap();
                let rotation = vec![readout_head_rows::Args { hidden: &hidden, norm: &norm, weight: &weight, out_rows: &out_rows, epsilon: 1e-6 }];
                let measured = kernel.measure(rotation, &TIMING_OPTIONS).unwrap();
                report(label, m, mapping, measured.median, bytes, m * v * d);
            }
        }
    }
}

#[test]
fn embedding_rows_decode_exactly_as_the_portable_body() {
    let Some(device) = device() else { return };
    let module = module();
    let act = Act::Bf16;
    let (v, d) = (300usize, 2560usize);
    for repr in [Repr::Q6k, Repr::Q4k, Repr::Q5k, Repr::Q8, Repr::Bf16] {
        let table = weight(repr, v, d, 90, 8.0);
        for rows in [1usize, 3, 8, 64] {
            let tokens = (0..rows).map(|i| ((i * 97 + 5) % v) as i32).collect::<Vec<_>>();
            // (token, status) rows, the `sample_rows` result layout.
            let pairs = tokens.iter().flat_map(|token| [*token, 0]).collect::<Vec<_>>();
            let out = embedding_rows::native_for_device_with(
                &device,
                embedding_rows::Elements { EW: repr.element(), A: act.element() },
                &NativeSpecialization::new().with_static("D", d as u64),
            )
            .unwrap()
            .call(embedding_rows::Args {
                table: &table.tensor(&device),
                tokens: &i32_tensor(&device, &[rows, 2], &pairs),
            })
            .unwrap();
            let expected = tokens
                .iter()
                .flat_map(|t| table.row(*t as usize).iter().map(|value| act.round(*value)))
                .collect::<Vec<_>>();
            let exact = vec![0.0; expected.len()];
            assert_within(&format!("embedding {repr:?} rows {rows} (A)"), &read_act(act, &out.r0), &expected, &exact);
            assert_within(&format!("embedding {repr:?} rows {rows} (F32)"), &read_f32(&out.r1), &expected, &exact);
            if rows == 3 {
                let outcome = interpret(
                    &module,
                    "embedding_rows",
                    &[("EW", repr.storage()), ("A", registry::dense(DType::BF16))],
                    vec![Input::Data(table.oracle()), ints(&[rows, 2], &pairs)],
                );
                assert_within(&format!("embedding {repr:?} portable"), &result(&outcome, 1), &expected, &exact);
            }
        }
    }
}
