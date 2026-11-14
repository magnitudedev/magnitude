//! Shared host models and fixtures of the CUDA K1 entry tests: random GGUF
//! blocks repacked into mma16 by the registry's host conversion, tensors,
//! RMS/projection host models and their bounds (`common::bound`), and
//! mappings.
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
#[path = "../int8_rule/mod.rs"]
pub mod int8_rule;
#[path = "../mapped/mod.rs"]
mod mapped;
#[path = "../common/bound.rs"]
pub mod bound;
// Not every suite reads every item.
#[allow(unused_imports)]
pub use bound::{Float, Form, Library, Products, assert_within, silu_product_from_exact};
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
    let shape = [rows as u64, k as u64];
    let external = format.external(rng, &shape, WEIGHT_SCALE);
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

/// RMS rows rounded to bf16, as the prologue forms them, with each value's
/// rounding flip: CUDA's prologue and this F32 host each land within
/// [`Library::rms`] of the exact normalized value (zero away from a rounding
/// boundary).
pub fn normalized(residual: &[f32], norm: &[f32], m: usize, k: usize) -> (Vec<f32>, Vec<f64>) {
    let native = Library::of(BackendName::Cuda).rms(k);
    let host = Library::of(BackendName::Cpu).rms(k);
    let window = (native + host) / (1.0 - host);
    let mut out = vec![0.0; m * k];
    let mut slack = vec![0.0; m * k];
    for row in 0..m {
        let line = &residual[row * k..(row + 1) * k];
        let squares: f32 = line.iter().map(|v| v * v).sum();
        let inverse = 1.0 / (squares / k as f32 + EPSILON).sqrt();
        for c in 0..k {
            let value = line[c] * inverse * norm[c];
            out[row * k + c] = bf16_round(value);
            slack[row * k + c] = Float::Bf16.flip(f64::from(value), window);
        }
    }
    (out, slack)
}

/// `Σ_k x[m, k] w[n, k]` in F64 for weight values `w` [n, k] of resident
/// `element`, and how far CUDA's F32 sum of each on `mapping` may be from it
/// ([`form`], [`Form::exact`]).
pub fn project(
    x: &[f32],
    w: &[f64],
    element: Element,
    m: usize,
    n: usize,
    k: usize,
    mapping: Mapping,
) -> (Vec<f64>, Vec<f64>) {
    let form = form(element, m, mapping);
    (0..m * n)
        .map(|i| {
            let (x, w) = (&x[i / n * k..][..k], &w[i % n * k..][..k]);
            let sum = x.iter().zip(w).map(|(a, b)| f64::from(*a) * b).sum::<f64>();
            (sum, form.exact(&Products::of(x, w)))
        })
        .unzip()
}

/// [`project`] of operand rows `x` [m, k] through `weight` [n, k], each
/// value's tolerance widened by its input `slack` through the weight
/// ([`slack_bound`]).
pub fn project_with_slack(
    x: &[f32],
    slack: &[f64],
    weight: &Weight,
    m: usize,
    n: usize,
    k: usize,
    mapping: Mapping,
) -> (Vec<f64>, Vec<f64>) {
    let (values, sums) = project(x, &weight.values, weight.tensor.element(), m, n, k, mapping);
    let carried = slack_bound(slack, &weight.values, m, n, k);
    (values, sum_slack(&sums, &carried))
}

/// The arithmetic of CUDA's projection of `rows` rows over a weight resident
/// as `element` on `mapping`: the INT8 path's past the GEMV band
/// ([`Mapping::quantizes`], [`Form::cuda_int8`]), else the 16-bit path's
/// ([`Form::projection`]).
pub fn form(element: Element, rows: usize, mapping: Mapping) -> Form {
    if mapping.quantizes(rows) {
        Form::cuda_int8()
    } else {
        Form::projection(BackendName::Cuda, rows, Float::Bf16, element)
    }
}

pub fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Each value's slack from two independent sources, summed.
pub fn sum_slack(first: &[f64], second: &[f64]) -> Vec<f64> {
    first.iter().zip(second).map(|(a, b)| a + b).collect()
}

/// How far a published residual output `residual + round_A(ŝ)` (F32) may be
/// from the exact `residual + sum`, the native sum `ŝ` within `sums` of
/// `sum`.
pub fn residual_bound(residual: f64, sum: f64, sums: f64) -> f64 {
    Float::F32.rounded_from_exact(Float::Bf16.rounded_from_exact(sums, sum), residual + sum)
}

/// How far an A-published projection may be from the exact `sum`, its native
/// sum within `sums` of it.
pub fn published_bound(sum: f64, sums: f64) -> f64 {
    Float::Bf16.rounded_from_exact(sums, sum)
}

/// The largest slope of SiLU, `sup x σ(x)' + σ(x)` ≈ 1.0998 (at x ≈ 2.4).
/// `SiLU(gate) · up` of operand rows `x` [o, h] through `gate` and `up`
/// [f, h] in F64, and each output's tolerance on `mapping`: each sum within
/// its form's bound plus the rows' prologue `flips` and `operand` slack
/// through its weight; through SiLU's slope and the other factor; and the
/// epilogue's own model, its three bf16 roundings and SiLU's evaluation
/// (`4 / 256` of the result, 1e-6 absolute).
pub fn expansion_bound(
    x: &[f32],
    flips: &[f64],
    operand: &[f64],
    gate: &Weight,
    up: &Weight,
    o: usize,
    f: usize,
    h: usize,
    mapping: Mapping,
) -> (Vec<f64>, Vec<f64>) {
    let (g, g_sums) = project(x, &gate.values, gate.tensor.element(), o, f, h, mapping);
    let (u, u_sums) = project(x, &up.values, up.tensor.element(), o, f, h, mapping);
    let slack = sum_slack(flips, operand);
    let g_slack = slack_bound(&slack, &gate.values, o, f, h);
    let u_slack = slack_bound(&slack, &up.values, o, f, h);
    (0..o * f)
        .map(|i| {
            let expected = silu(g[i]) * u[i];
            let (dg, du) = (g_sums[i] + g_slack[i], u_sums[i] + u_slack[i]);
            (expected, silu_product_from_exact(Float::Bf16, &Library::of(BackendName::Cuda), (g[i], dg), (u[i], du)))
        })
        .unzip()
}

/// The largest row count the CUDA K1 GEMV serves (`projection::GEMV_ROWS`);
/// larger counts run the GEMM.
pub const GEMV_ROWS: usize = bound::gemv_rows(BackendName::Cuda);

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
                    .with_static("US", 0)
                    .with_static("LB", 0),
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

/// INT8 quantization of rows of `k` as the INT8 path forms them, per 32
/// values by the rule every backend shares (`int8_rule`). Returns the
/// dequantized rows and each value's half quantization step (d / 2).
pub fn quantize(x: &[f32], k: usize) -> (Vec<f32>, Vec<f64>) {
    let mut values = vec![0.0; x.len()];
    let mut half_steps = vec![0.0; x.len()];
    for group in (0..x.len()).step_by(int8_rule::GROUP) {
        debug_assert_eq!(group % k % int8_rule::GROUP, 0);
        let (d, codes) = int8_rule::cpu(x[group..group + int8_rule::GROUP].try_into().unwrap());
        for (i, code) in codes.iter().enumerate() {
            values[group + i] = d * f32::from(*code);
            half_steps[group + i] = f64::from(d) / 2.0;
        }
    }
    (values, half_steps)
}

/// sum_k |w[n, k]| * slack[m, k]: the bound of a projection whose inputs may
/// each differ by `slack`.
pub fn slack_bound(slack: &[f64], w: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            out[row * n + col] = (0..k)
                .map(|c| slack[row * k + c] * w[col * k + c].abs())
                .sum::<f64>();
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
) -> (Vec<f32>, Vec<f64>) {
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

