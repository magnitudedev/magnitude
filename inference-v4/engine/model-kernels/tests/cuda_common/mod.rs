//! Shared host models and fixtures of the CUDA K1 entry tests: random GGUF
//! blocks repacked into mma16 by the registry's host conversion, tensors,
//! RMS/projection host models, mappings and tolerance checks.
#![allow(dead_code)]

use seismic::{BackendName, Device, DeviceCatalog, Element, Layout, Tensor};
use seismic_lang::registry::{bf16_round, f16_bits};

pub fn cuda() -> Option<Device> {
    DeviceCatalog::discover().ok()?.open_backend(BackendName::Cuda).ok()
}

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    pub fn byte(&mut self) -> u8 {
        self.next() as u8
    }
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * (self.next() as f32 / (1u64 << 31) as f32)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Format {
    Q4K,
    Q5K,
    Q6K,
    Q8,
}

impl Format {
    pub const ALL: [Format; 4] = [Format::Q4K, Format::Q5K, Format::Q6K, Format::Q8];
    pub fn external(self) -> &'static str {
        match self {
            Format::Q4K => "gguf_q4_k",
            Format::Q5K => "gguf_q5_k",
            Format::Q6K => "gguf_q6_k",
            Format::Q8 => "gguf_q8_0",
        }
    }
    pub fn representation(self) -> &'static str {
        match self {
            Format::Q4K => "q4k",
            Format::Q5K => "q5k",
            Format::Q6K => "q6k",
            Format::Q8 => "q8g32s",
        }
    }
    pub fn resident(self) -> Element {
        Element::stored(self.representation(), Layout::Mma16).unwrap()
    }
    /// One random GGUF block (with small positive super factors).
    pub fn block(self, rng: &mut Rng) -> Vec<u8> {
        let f16 = |value: f32| f16_bits(value).to_le_bytes();
        let mut block = Vec::new();
        match self {
            Format::Q4K | Format::Q5K => {
                block.extend(f16(rng.uniform(0.0005, 0.003)));
                block.extend(f16(rng.uniform(0.0, 0.002)));
                let bytes = if matches!(self, Format::Q4K) { 12 + 128 } else { 12 + 32 + 128 };
                block.extend((0..bytes).map(|_| rng.byte()));
            }
            Format::Q6K => {
                block.extend((0..128 + 64).map(|_| rng.byte()));
                block.extend((0..16).map(|_| (rng.next() % 128) as u8 ^ 0xC0));
                block.extend(f16(rng.uniform(0.0002, 0.001)));
            }
            Format::Q8 => {
                block.extend(f16(rng.uniform(0.001, 0.005)));
                block.extend((0..32).map(|_| rng.byte()));
            }
        }
        block
    }
    pub fn block_values(self) -> usize {
        match self {
            Format::Q8 => 32,
            _ => 256,
        }
    }
}

/// A resident mma16 weight [rows, k] and its decoded values.
pub struct Weight {
    pub tensor: Tensor,
    pub values: Vec<f64>,
    pub bytes: Vec<u8>,
}

pub fn weight(device: &Device, format: Format, rows: usize, k: usize, rng: &mut Rng) -> Weight {
    let mut external = Vec::new();
    for _ in 0..rows * k / format.block_values() {
        external.extend(format.block(rng));
    }
    let shape = [rows as u64, k as u64];
    let resident = format.resident();
    let bytes = resident
        .repack_host(Element::named(format.external()).unwrap(), &shape, &external)
        .expect("registered GGUF -> mma16 conversion");
    let values = resident.decode_host(&shape, &bytes).unwrap();
    let tensor = Tensor::from_host(device, resident, &shape, &bytes).unwrap();
    Weight { tensor, values, bytes }
}

pub fn f32_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::f32(), shape, &bytes).unwrap()
}
pub fn bf16_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|v| ((bf16_round(*v).to_bits() >> 16) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::bf16(), shape, &bytes).unwrap()
}
pub fn i32_tensor(device: &Device, shape: &[u64], values: &[i32]) -> Tensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::i32(), shape, &bytes).unwrap()
}
pub fn u32_tensor(device: &Device, shape: &[u64], values: &[u32]) -> Tensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    Tensor::from_host(device, Element::u32(), shape, &bytes).unwrap()
}
pub fn read_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}
pub fn read_bf16(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(2)
        .map(|w| f32::from_bits((u16::from_le_bytes(w.try_into().unwrap()) as u32) << 16))
        .collect()
}
pub fn read_i32(tensor: &Tensor) -> Vec<i32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

pub const EPSILON: f32 = 1e-6;

/// RMS rows rounded to bf16, as the prologue forms them.
pub fn normalized(residual: &[f32], norm: &[f32], m: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * k];
    for row in 0..m {
        let line = &residual[row * k..(row + 1) * k];
        let squares: f32 = line.iter().map(|v| v * v).sum();
        let inverse = 1.0 / (squares / k as f32 + EPSILON).sqrt();
        for c in 0..k {
            out[row * k + c] = bf16_round(line[c] * inverse * norm[c]);
        }
    }
    out
}

/// sum_k x[m, k] * w[n, k] and sum_k |x[m, k] * w[n, k]| in f64.
pub fn project(x: &[f32], w: &[f64], m: usize, n: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0; m * n];
    let mut magnitude = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            let mut abs = 0.0;
            for c in 0..k {
                let p = f64::from(x[row * k + c]) * w[col * k + c];
                sum += p;
                abs += p.abs();
            }
            out[row * n + col] = sum;
            magnitude[row * n + col] = abs;
        }
    }
    (out, magnitude)
}

pub fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}


/// A native mapping: GEMV K split, GEMM BM, the INT8 operand path and the
/// GEMM split-K (entries without SPLIT ignore it).
#[derive(Clone, Copy, Debug)]
pub struct Mapping {
    pub ksplit: u64,
    pub bm: u64,
    pub int8: u64,
    pub split: u64,
}

impl Mapping {
    /// The specialization params of an entry; `split` only for entries
    /// declaring SPLIT.
    pub fn params(self, spec: seismic::NativeSpecialization, split: bool) -> seismic::NativeSpecialization {
        let spec = spec.with_param("KSPLIT", self.ksplit).with_param("BM", self.bm).with_param("INT8", self.int8);
        if split { spec.with_param("SPLIT", self.split) } else { spec }
    }
}

pub const GEMV_MAPPINGS: [Mapping; 4] = [
    Mapping { ksplit: 4, bm: 64, int8: 0, split: 1 },
    Mapping { ksplit: 2, bm: 64, int8: 0, split: 1 },
    Mapping { ksplit: 8, bm: 64, int8: 1, split: 1 },
    Mapping { ksplit: 4, bm: 64, int8: 1, split: 1 },
];
pub const GEMM_MAPPINGS: [Mapping; 4] = [
    Mapping { ksplit: 4, bm: 64, int8: 0, split: 1 },
    Mapping { ksplit: 4, bm: 128, int8: 0, split: 2 },
    Mapping { ksplit: 4, bm: 64, int8: 1, split: 2 },
    Mapping { ksplit: 4, bm: 128, int8: 1, split: 1 },
];

pub fn mappings(m: usize) -> &'static [Mapping] {
    if m <= 8 { &GEMV_MAPPINGS } else { &GEMM_MAPPINGS }
}

/// q8_1 quantization of rows of `k` as the INT8 path forms them: per 32
/// values d = max|x| / 127 and x ~= d * rint(x / d). Returns the dequantized
/// rows and each value's half quantization step (d / 2).
pub fn quantize(x: &[f32], k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut values = vec![0.0; x.len()];
    let mut half_steps = vec![0.0; x.len()];
    for group in (0..x.len()).step_by(32) {
        debug_assert_eq!(group % k % 32, 0);
        let maximum = x[group..group + 32].iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let d = maximum / 127.0;
        for i in group..group + 32 {
            values[i] = if d == 0.0 { 0.0 } else { d * (x[i] / d).round_ties_even() };
            half_steps[i] = d / 2.0;
        }
    }
    (values, half_steps)
}

/// sum_k |w[n, k]| * slack[m, k]: the bound of a projection whose inputs may
/// each differ by `slack`.
pub fn slack_bound(slack: &[f32], w: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            out[row * n + col] =
                (0..k).map(|c| f64::from(slack[row * k + c]) * w[col * k + c].abs()).sum::<f64>();
        }
    }
    out
}

/// The bound of the 16-bit GEMM's weight dequantization (rows > 8, INT8 off):
/// each weight becomes round_A(code * round_A(scale) - round_A(bias)), off
/// by at most 2^-9 (|code * scale| + |bias| + |value|) <= 2^-7 max_row |w|
/// in bf16; zero on every other path.
pub fn dequant_bound(x: &[f32], w: &[f64], m: usize, n: usize, k: usize, mapping: Mapping) -> Vec<f64> {
    if m <= 8 || mapping.int8 == 1 {
        return vec![0.0; m * n];
    }
    let maximum: Vec<f64> =
        (0..n).map(|col| w[col * k..(col + 1) * k].iter().fold(0.0f64, |a, v| a.max(v.abs()))).collect();
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        let magnitude: f64 = x[row * k..(row + 1) * k].iter().map(|v| f64::from(v.abs())).sum();
        for col in 0..n {
            out[row * n + col] = magnitude * maximum[col] / 128.0;
        }
    }
    out
}

/// The activation rows a mapping's operand path multiplies, and the slack of
/// each (zero on the 16-bit path; half a quantization step on the INT8 path,
/// where a tie resolved differently before quantization moves a value by up
/// to one step).
pub fn operand_rows(x: &[f32], k: usize, mapping: Mapping) -> (Vec<f32>, Vec<f32>) {
    if mapping.int8 == 1 { quantize(x, k) } else { (x.to_vec(), vec![0.0; x.len()]) }
}

pub fn check(label: &str, actual: &[f32], expected: &[f64], tolerance: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let mut worst = (0.0f64, 0usize);
    for i in 0..actual.len() {
        let ratio = (f64::from(actual[i]) - expected[i]).abs() / tolerance[i];
        assert!(ratio.is_finite(), "{label}[{i}]: {} vs {}", actual[i], expected[i]);
        if ratio > worst.0 {
            worst = (ratio, i);
        }
    }
    assert!(
        worst.0 <= 1.0,
        "{label}[{}]: {} vs {} (tolerance {}, ratio {:.3})",
        worst.1,
        actual[worst.1],
        expected[worst.1],
        tolerance[worst.1],
        worst.0
    );
}


/// A zero-filled resident weight for timings: device time does not depend on
/// weight values, and the host conversion of 4B-sized weights is slow.
pub fn timing_weight(device: &Device, format: Format, rows: usize, k: usize) -> Tensor {
    Tensor::zeros(device, format.resident(), &[rows as u64, k as u64]).unwrap()
}
