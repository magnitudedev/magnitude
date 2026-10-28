//! Shared host models and fixtures of the CUDA K1 entry tests: random GGUF
//! blocks repacked into mma16 by the registry's host conversion, tensors,
//! RMS/projection host models, mappings and tolerance checks.
#![allow(dead_code)]

use seismic::{BackendName, Device, Element, Layout, Tensor};
use seismic_lang::registry::{bf16_round, f16_bits};

pub fn cuda() -> Option<Device> {
    seismic::Devices::discover(&[])
        .ok()?
        .open_backend(BackendName::Cuda)
        .ok()
}

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    pub fn byte(&mut self) -> u8 {
        self.next() as u8
    }
    pub fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * (self.next() as f32 / (1u64 << 31) as f32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Q4K,
    Q5K,
    Q6K,
    Q8,
    /// GGUF IQ4_XS, resident as iq4xs.
    Iq4,
    /// GGUF Q4_0, Q4_1, Q5_0, Q5_1, MXFP4, NVFP4 (representations of their
    /// own).
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Mxfp4,
    Nvfp4,
    /// GGUF NVFP4 calibrated for FP4 activations (W4A4): nvfp4g16's storage,
    /// NVFP4 operands.
    Nvfp4A4,
    /// GGUF IQ2_XS, IQ3_XXS and IQ2_XXS (vector codes, representations of
    /// their own).
    Iq2Xs,
    Iq3Xxs,
    Iq2Xxs,
    /// GGUF Q2_K, resident as q2k.
    Q2K,
    /// GGUF Q3_K, resident as q3k.
    Q3K,
    /// GGUF IQ2_S, resident as iq2s (vector codes with a high part).
    Iq2S,
    /// GGUF IQ3_S, resident as iq3s (vector codes with a high part).
    Iq3S,
}

impl Format {
    pub const ALL: [Format; 19] = [
        Format::Q4K,
        Format::Q5K,
        Format::Q6K,
        Format::Q8,
        Format::Iq4,
        Format::Q4_0,
        Format::Q4_1,
        Format::Q5_0,
        Format::Q5_1,
        Format::Mxfp4,
        Format::Nvfp4,
        Format::Nvfp4A4,
        Format::Iq2Xs,
        Format::Iq3Xxs,
        Format::Iq2Xxs,
        Format::Q2K,
        Format::Q3K,
        Format::Iq2S,
        Format::Iq3S,
    ];
    pub fn external(self) -> &'static str {
        match self {
            Format::Q4K => "gguf_q4_k",
            Format::Q5K => "gguf_q5_k",
            Format::Q6K => "gguf_q6_k",
            Format::Q8 => "gguf_q8_0",
            Format::Iq4 => "gguf_iq4_xs",
            Format::Q4_0 => "gguf_q4_0",
            Format::Q4_1 => "gguf_q4_1",
            Format::Q5_0 => "gguf_q5_0",
            Format::Q5_1 => "gguf_q5_1",
            Format::Mxfp4 => "gguf_mxfp4",
            Format::Nvfp4 | Format::Nvfp4A4 => "gguf_nvfp4",
            Format::Iq2Xs => "gguf_iq2_xs",
            Format::Iq3Xxs => "gguf_iq3_xxs",
            Format::Iq2Xxs => "gguf_iq2_xxs",
            Format::Q2K => "gguf_q2_k",
            Format::Q3K => "gguf_q3_k",
            Format::Iq2S => "gguf_iq2_s",
            Format::Iq3S => "gguf_iq3_s",
        }
    }
    pub fn representation(self) -> &'static str {
        match self {
            Format::Q4K => "q4k",
            Format::Q5K => "q5k",
            Format::Q6K => "q6k",
            Format::Q8 => "q8g32s",
            Format::Iq4 => "iq4xs",
            Format::Q4_0 => "q4g32s",
            Format::Q4_1 => "q4g32",
            Format::Q5_0 => "q5g32s",
            Format::Q5_1 => "q5g32",
            Format::Mxfp4 => "mxfp4g32",
            Format::Nvfp4 => "nvfp4g16",
            Format::Nvfp4A4 => "nvfp4g16a4",
            Format::Iq2Xs => "iq2xs",
            Format::Iq3Xxs => "iq3xxs",
            Format::Iq2Xxs => "iq2xxs",
            Format::Q2K => "q2k",
            Format::Q3K => "q3k",
            Format::Iq2S => "iq2s",
            Format::Iq3S => "iq3s",
        }
    }
    /// Whether the CUDA GEMM keeps this format's codes exact and scales its
    /// F32 partial products (`SCALED_GEMM`): its weights are never rounded
    /// to the activation type.
    pub fn scaled_gemm(self) -> bool {
        matches!(
            self,
            Format::Q4_0
                | Format::Q4_1
                | Format::Q5_0
                | Format::Q5_1
                | Format::Mxfp4
                | Format::Nvfp4
                | Format::Nvfp4A4
                | Format::Iq2Xs
                | Format::Iq3Xxs
                | Format::Iq2Xxs
                | Format::Iq2S
                | Format::Iq3S
        )
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
                let bytes = if matches!(self, Format::Q4K) {
                    12 + 128
                } else {
                    12 + 32 + 128
                };
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
            Format::Iq4 => {
                // d, the eight six-bit sub-block scales (each less 32) as
                // 2 high-bit + 4 low-nibble bytes, and 128 bytes of table codes.
                block.extend(f16(rng.uniform(0.00005, 0.0003)));
                block.extend((0..2 + 4 + 128).map(|_| rng.byte()));
            }
            Format::Q4_0 => {
                block.extend(f16(rng.uniform(0.0005, 0.003)));
                block.extend((0..16).map(|_| rng.byte()));
            }
            Format::Q4_1 => {
                block.extend(f16(rng.uniform(0.0005, 0.003)));
                block.extend(f16(-rng.uniform(0.005, 0.02)));
                block.extend((0..16).map(|_| rng.byte()));
            }
            Format::Q5_0 => {
                block.extend(f16(rng.uniform(0.0002, 0.0015)));
                block.extend((0..4 + 16).map(|_| rng.byte()));
            }
            Format::Q5_1 => {
                block.extend(f16(rng.uniform(0.0002, 0.0015)));
                block.extend(f16(-rng.uniform(0.005, 0.02)));
                block.extend((0..4 + 16).map(|_| rng.byte()));
            }
            Format::Mxfp4 => {
                // E8M0 2^-10 .. 2^-8 (values up to 6 of them).
                block.push(117 + (rng.next() % 3) as u8);
                block.extend((0..16).map(|_| rng.byte()));
            }
            Format::Nvfp4 | Format::Nvfp4A4 => {
                // UE4M3 subnormal and smallest-normal scales (<= 2^-5).
                block.extend((0..4).map(|_| (rng.next() % 16) as u8));
                block.extend((0..32).map(|_| rng.byte()));
            }
            Format::Iq2Xs => {
                // d (values reach d * 31 * 43 / 8), 32 grid entries, eight
                // scale bytes.
                block.extend(f16(rng.uniform(0.0002, 0.001)));
                block.extend((0..64 + 8).map(|_| rng.byte()));
            }
            Format::Iq3Xxs => {
                // d (values reach d * 31 * 62 / 4), 64 grid indices, eight
                // sign-and-scale words.
                block.extend(f16(rng.uniform(0.00005, 0.0003)));
                block.extend((0..64 + 32).map(|_| rng.byte()));
            }
            Format::Iq2Xxs => {
                // d (values reach d * 31 * 43 / 8), then eight pairs of four
                // grid indices and a sign-and-scale word.
                block.extend(f16(rng.uniform(0.0002, 0.001)));
                block.extend((0..64).map(|_| rng.byte()));
            }
            Format::Q2K => {
                // Scale and minimum nibbles, codes, then d and dmin.
                block.extend((0..16 + 64).map(|_| rng.byte()));
                block.extend(f16(rng.uniform(0.0005, 0.003)));
                block.extend(f16(rng.uniform(0.0, 0.002)));
            }
            Format::Q3K => {
                // High bits, low codes and scale bytes, then d (values
                // reach d * 32 * 4).
                block.extend((0..32 + 64 + 12).map(|_| rng.byte()));
                block.extend(f16(rng.uniform(0.0001, 0.0005)));
            }
            Format::Iq2S => {
                // d (values reach d * 31 * 43 / 8), 32 low index bytes, 32
                // sign bytes, eight high index bytes, eight scale bytes.
                block.extend(f16(rng.uniform(0.0002, 0.001)));
                block.extend((0..64 + 8 + 8).map(|_| rng.byte()));
            }
            Format::Iq3S => {
                // d (values reach d * 31 * 15), low grid indices, their high
                // bits, signs and scale nibbles.
                block.extend(f16(rng.uniform(0.00005, 0.0003)));
                block.extend((0..64 + 8 + 32 + 4).map(|_| rng.byte()));
            }
        }
        block
    }
    pub fn block_values(self) -> usize {
        match self {
            Format::Q8
            | Format::Q4_0
            | Format::Q4_1
            | Format::Q5_0
            | Format::Q5_1
            | Format::Mxfp4 => 32,
            Format::Nvfp4 | Format::Nvfp4A4 => 64,
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
        .repack_host(
            Element::named(format.external()).unwrap(),
            &shape,
            &external,
        )
        .expect("registered GGUF -> mma16 conversion");
    let values = resident.decode_host(&shape, &bytes).unwrap();
    let tensor = Tensor::from_host(device, resident, &shape, &bytes).unwrap();
    Weight {
        tensor,
        values,
        bytes,
    }
}

/// A dense bf16 weight [rows, k] (row-major, as dense GGUF weights are
/// resident) and its values.
pub fn dense_weight(device: &Device, rows: usize, k: usize, rng: &mut Rng) -> Weight {
    let values: Vec<f32> = (0..rows * k)
        .map(|_| bf16_round(rng.uniform(-0.05, 0.05)))
        .collect();
    let tensor = bf16_tensor(device, &[rows as u64, k as u64], &values);
    let bytes = tensor.read_to_host().unwrap();
    Weight {
        tensor,
        values: values.iter().map(|v| f64::from(*v)).collect(),
        bytes,
    }
}

pub fn f32_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
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
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::i32(), shape, &bytes).unwrap()
}
pub fn u32_tensor(device: &Device, shape: &[u64], values: &[u32]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
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

/// The largest row count the CUDA K1 GEMV serves (`projection::GEMV_ROWS`);
/// larger counts run the GEMM.
pub const GEMV_ROWS: usize = 16;

/// The largest row count the small-band GEMM serves
/// (`projection::SMALL_GEMM_ROWS`); larger counts run the large-band GEMM.
pub const SMALL_GEMM_ROWS: usize = 64;

/// A native mapping: the GEMV K split, the small-band GEMM's INT8 operand
/// path (the large-band GEMM is the 16-bit path) and the large-band GEMM's
/// loader rotation.
#[derive(Clone, Copy, Debug)]
pub struct Mapping {
    pub ksplit: u64,
    pub int8: u64,
    pub rotate: u64,
}

impl Mapping {
    /// Entry-wide K1 projection parameters outside the dense pair.
    pub fn params(self, spec: seismic::NativeSpecialization) -> seismic::NativeSpecialization {
        spec.with_param("KSPLIT", self.ksplit)
            .with_param("INT8", self.int8)
    }
    /// Shared producer/GEMV/small-GEMM/large-GEMM schedule: arithmetic
    /// representation is independent of the row tile chosen by the schedule.
    pub fn projection_params(
        self,
        spec: seismic::NativeSpecialization,
    ) -> seismic::NativeSpecialization {
        spec.with_param("INT8", self.int8)
            .with_param("OPERAND", 0)
            .with_launch_param(1, "KSPLIT", self.ksplit)
            .with_launch_param(2, "KSPLIT", self.ksplit)
            .with_launch_param(4, "ROTATE", self.rotate)
    }
    /// [`Mapping::projection_params`] of the dense pair (`dense_up`,
    /// `dense_output`), whose GEMV launches share one entry-wide KSPLIT.
    pub fn dense_pair_params(
        self,
        spec: seismic::NativeSpecialization,
    ) -> seismic::NativeSpecialization {
        spec.with_param("INT8", self.int8)
            .with_param("OPERAND", 0)
            .with_param("KSPLIT", self.ksplit)
            .with_launch_param(4, "ROTATE", self.rotate)
    }
    /// Readout head GEMVs own KSPLIT on launches 1 and 2, its GEMM ROTATE.
    pub fn head_params(self, spec: seismic::NativeSpecialization) -> seismic::NativeSpecialization {
        spec.with_param("OPERAND", 0)
            .with_launch_param(1, "KSPLIT", self.ksplit)
            .with_launch_param(2, "KSPLIT", self.ksplit)
            .with_launch_param(4, "ROTATE", self.rotate)
    }
    /// Whether `rows` rows use quantized activation operands in a matrix band.
    pub fn quantizes(self, rows: usize) -> bool {
        self.int8 == 1 && rows > GEMV_ROWS
    }
}

/// GEMV mappings. INT8 1 is included to show the GEMV ignores it.
pub const GEMV_MAPPINGS: [Mapping; 4] = [
    Mapping {
        ksplit: 4,
        int8: 0,
        rotate: 2,
    },
    Mapping {
        ksplit: 2,
        int8: 0,
        rotate: 2,
    },
    Mapping {
        ksplit: 8,
        int8: 1,
        rotate: 2,
    },
    Mapping {
        ksplit: 4,
        int8: 1,
        rotate: 2,
    },
];
/// GEMM mappings: both operand paths (the large band ignores INT8) and every
/// loader rotation of the large band (which ignores INT8 and keeps the
/// arithmetic).
pub const GEMM_MAPPINGS: [Mapping; 4] = [
    Mapping {
        ksplit: 4,
        int8: 0,
        rotate: 2,
    },
    Mapping {
        ksplit: 4,
        int8: 1,
        rotate: 2,
    },
    Mapping {
        ksplit: 4,
        int8: 0,
        rotate: 1,
    },
    Mapping {
        ksplit: 4,
        int8: 0,
        rotate: 4,
    },
];

pub fn mappings(m: usize) -> &'static [Mapping] {
    if m <= GEMV_ROWS {
        &GEMV_MAPPINGS
    } else {
        &GEMM_MAPPINGS
    }
}

/// q8_1 quantization of rows of `k` as the INT8 path forms them: per 32
/// values d = max|x| / 127 and x ~= d * rint(x / d). Returns the dequantized
/// rows and each value's half quantization step (d / 2).
pub fn quantize(x: &[f32], k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut values = vec![0.0; x.len()];
    let mut half_steps = vec![0.0; x.len()];
    for group in (0..x.len()).step_by(32) {
        debug_assert_eq!(group % k % 32, 0);
        let maximum = x[group..group + 32]
            .iter()
            .fold(0.0f32, |acc, v| acc.max(v.abs()));
        let d = maximum / 127.0;
        for i in group..group + 32 {
            values[i] = if d == 0.0 {
                0.0
            } else {
                d * (x[i] / d).round_ties_even()
            };
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
            out[row * n + col] = (0..k)
                .map(|c| f64::from(slack[row * k + c]) * w[col * k + c].abs())
                .sum::<f64>();
        }
    }
    out
}

/// The bound of the 16-bit GEMM's weight dequantization (rows > GEMV_ROWS,
/// not on the INT8 path): each weight becomes round_A(code * round_A(scale)
/// - round_A(bias)), off by at most 2^-9 (|code * scale| + |bias| + |value|)
/// <= 2^-7 max_row |w| in bf16; zero on every other path.
pub fn dequant_bound(
    x: &[f32],
    w: &[f64],
    m: usize,
    n: usize,
    k: usize,
    mapping: Mapping,
) -> Vec<f64> {
    if m <= GEMV_ROWS || mapping.quantizes(m) {
        return vec![0.0; m * n];
    }
    let maximum: Vec<f64> = (0..n)
        .map(|col| {
            w[col * k..(col + 1) * k]
                .iter()
                .fold(0.0f64, |a, v| a.max(v.abs()))
        })
        .collect();
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        let magnitude: f64 = x[row * k..(row + 1) * k]
            .iter()
            .map(|v| f64::from(v.abs()))
            .sum();
        for col in 0..n {
            out[row * n + col] = magnitude * maximum[col] / 128.0;
        }
    }
    out
}

/// The activation rows (x holds rows of `k`) a mapping's operand path
/// multiplies over weights of `format` (`None`: dense), and the slack of
/// each: the recipe's rows of NVFP4 weights (`nvfp4_rows`), exact, which
/// activation quantization never re-forms; half a quantization step on the
/// INT8 path, where a tie resolved differently before quantization moves a
/// value by up to one step; else the rows as they are, exact.
pub fn operand_rows(
    x: &[f32],
    k: usize,
    mapping: Mapping,
    format: Option<Format>,
) -> (Vec<f32>, Vec<f32>) {
    if format == Some(Format::Nvfp4A4) {
        (nvfp4_rows(x, k), vec![0.0; x.len()])
    } else if mapping.quantizes(x.len() / k) {
        quantize(x, k)
    } else {
        (x.to_vec(), vec![0.0; x.len()])
    }
}

#[path = "../operands/mod.rs"]
mod operands;
pub use operands::nvfp4_rows;

pub fn check(label: &str, actual: &[f32], expected: &[f64], tolerance: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let mut worst = (0.0f64, 0usize);
    for i in 0..actual.len() {
        let ratio = (f64::from(actual[i]) - expected[i]).abs() / tolerance[i];
        assert!(
            ratio.is_finite(),
            "{label}[{i}]: {} vs {}",
            actual[i],
            expected[i]
        );
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
