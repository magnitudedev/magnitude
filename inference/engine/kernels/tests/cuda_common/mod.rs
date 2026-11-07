//! Shared host models and fixtures of the CUDA K1 entry tests: random GGUF
//! blocks repacked into mma16 by the registry's host conversion, tensors,
//! RMS/projection host models, mappings and tolerance checks.
#![allow(dead_code)]

use magnitude_kernels::dense_expand;
use seismic::{BackendName, Device, Element, Layout, NativeSpecialization, OperandRecipe, Tensor};
use seismic_lang::registry::bf16_round;

/// The CUDA device, opened with the test binary's one store of compiled
/// programs ([`seismic_native_tests::devices`]): a form a test prepares again,
/// for another row count, mode or mapping, compiles once in the process.
pub fn cuda() -> Option<Device> {
    seismic_native_tests::devices(&[BackendName::Cuda]).into_iter().next()
}

#[path = "../gguf_formats/mod.rs"]
mod gguf_formats;
#[path = "../mapped/mod.rs"]
mod mapped;
pub use gguf_formats::{Format, Rng};
pub use mapped::Mapped;
// The suites with a module of tests a format generate them.
#[allow(unused_imports)]
pub(crate) use gguf_formats::gguf_format_tests;

/// The magnitude of the weights' values, packed and dense.
pub const WEIGHT_SCALE: f32 = 0.05;

impl Format {
    /// The mma16 element CUDA's kernels read.
    pub fn resident(self) -> Element {
        self.element(Layout::Mma16)
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
        external.extend(format.block(rng, WEIGHT_SCALE));
    }
    let shape = [rows as u64, k as u64];
    let resident = format.resident();
    let bytes = resident
        .repack_host(format.source(), &shape, &external)
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
        .map(|_| bf16_round(rng.uniform(-WEIGHT_SCALE, WEIGHT_SCALE)))
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
/// path (the large-band GEMM is the 16-bit path), the large-band GEMM's
/// loader rotation and the dense pair's GEMV residency (blocks of streaming
/// weights per SM).
#[derive(Clone, Copy, Debug)]
pub struct Mapping {
    pub ksplit: u64,
    pub int8: u64,
    pub rotate: u64,
    pub residency: u64,
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
            .with_launch_param(1, "RESIDENCY", self.residency)
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
        residency: 1,
    },
    Mapping {
        ksplit: 2,
        int8: 0,
        rotate: 2,
        residency: 2,
    },
    Mapping {
        ksplit: 8,
        int8: 1,
        rotate: 2,
        residency: 2,
    },
    Mapping {
        ksplit: 4,
        int8: 1,
        rotate: 2,
        residency: 1,
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
        residency: 1,
    },
    Mapping {
        ksplit: 4,
        int8: 1,
        rotate: 2,
        residency: 1,
    },
    Mapping {
        ksplit: 4,
        int8: 0,
        rotate: 1,
        residency: 1,
    },
    Mapping {
        ksplit: 4,
        int8: 0,
        rotate: 4,
        residency: 1,
    },
];

/// `dense_expand` of [h]-wide rows through `gate` and `up` weights of `f`
/// rows.
pub fn expansion(
    gate: Element,
    up: Element,
    h: usize,
    f: usize,
) -> Mapped<dense_expand::Entry, impl Fn(&Device, Mapping) -> NativeSpecialization> {
    Mapped::new(
        dense_expand::native_entry_with(dense_expand::Elements {
            NW: Element::f32(),
            GW: gate,
            UW: up,
            A: Element::bf16(),
        }),
        move |_: &Device, mapping: Mapping| {
            mapping.projection_params(
                NativeSpecialization::new()
                    .with_static("H", h as u64)
                    .with_static("F", f as u64)
                    .with_static("GS", 0)
                    .with_static("US", 0),
            )
        },
    )
}

/// The mappings of `m` rows that `admits` holds for ([`admitted`]).
pub fn mappings(m: usize, admits: impl Fn(Mapping) -> bool) -> Vec<Mapping> {
    admitted(
        if m <= GEMV_ROWS {
            &GEMV_MAPPINGS
        } else {
            &GEMM_MAPPINGS
        },
        admits,
    )
}

/// The `mappings` that `admits` holds for: those every entry a test
/// prepares under each admits ([`Mapped::admits`]).
pub fn admitted(mappings: &[Mapping], admits: impl Fn(Mapping) -> bool) -> Vec<Mapping> {
    mappings
        .iter()
        .copied()
        .filter(|mapping| admits(*mapping))
        .collect()
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
    if format.is_some_and(|format| format.operands == OperandRecipe::Nvfp4Rows) {
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
