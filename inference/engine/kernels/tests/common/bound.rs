//! How far a native form may stray from its host model of the same math:
//! a rounding analysis of the arithmetic both perform, never a tolerance.
//! (It is not the precision system's r <= α admission of a tuned
//! alternative, docs/precision.md.)
//!
//! A host model sums the K products `x_k w_k` of one output in F32, in its
//! own order. A native [`Form`] sums the same products in F32 in another
//! order, of operands it may first stage in a narrower format on a matrix
//! unit. Both round the sum once to the output format ([`Float::rounded`]).
//! Underflow of F32 products and sums and overflow anywhere are outside the
//! analysis; underflow of a staged operand is in it.

#![allow(dead_code)]

use seismic::{BackendName, Element};

/// A floating format a value is rounded to, to nearest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Float {
    F32,
    Bf16,
    F16,
}

impl Float {
    /// The unit roundoff `u`: rounding moves a value in the normal range by
    /// at most `u` times its magnitude.
    pub fn unit(self) -> f64 {
        2f64.powi(-match self {
            Float::F32 => 24,
            Float::Bf16 => 8,
            Float::F16 => 11,
        })
    }

    /// The most rounding moves a value below the normal range: half the
    /// subnormal step.
    pub fn underflow(self) -> f64 {
        2f64.powi(-match self {
            Float::F32 => 150,
            Float::Bf16 => 134,
            Float::F16 => 25,
        })
    }

    /// The format of a dense element's values; `None` for packed storage.
    pub fn dense(element: Element) -> Option<Float> {
        [
            (Element::bf16(), Float::Bf16),
            (Element::f16(), Float::F16),
            (Element::f32(), Float::F32),
        ]
        .into_iter()
        .find_map(|(dense, float)| (dense == element).then_some(float))
    }

    /// How a value of this format is held once converted to `to`.
    pub fn staged_as(self, to: Float) -> Staging {
        match (self, to) {
            _ if self == to => Staging::Exact,
            (_, Float::F32) => Staging::Exact,
            (Float::Bf16, Float::F16) => Staging::IntoF16 { scaled: false },
            _ => Staging::Rounded(to),
        }
    }

    /// How far apart two values within `difference` of each other may be
    /// once each is rounded to this format, `y` the host's rounded value.
    /// For `|a - b| <= Δ` and `|r(v) - v| <= u|v| + η`:
    /// `|r(a) - r(b)| <= Δ + u(|a| + |b|) + 2η <= (1 + u)Δ + 2u|b| + 2η`,
    /// with `|b| <= (|y| + η) / (1 - u)`.
    pub fn rounded(self, difference: f64, y: f32) -> f64 {
        let (u, eta) = (self.unit(), self.underflow());
        (1.0 + u) * difference + 2.0 * u * (f64::from(y).abs() + eta) / (1.0 - u) + 2.0 * eta
    }

    /// How far a value within `difference` of an exact `b` may be from `b`
    /// once rounded to this format: `|r(a) - b| <= Δ + u|a| + η <=
    /// (1 + u)Δ + u|b| + η`.
    pub fn rounded_from_exact(self, difference: f64, b: f64) -> f64 {
        let (u, eta) = (self.unit(), self.underflow());
        (1.0 + u) * difference + u * b.abs() + eta
    }
}

/// `γ_n = n 2^-24 / (1 - n 2^-24)`: an F32 sum of `n` products, in any order,
/// with or without fused multiply-adds, is within `γ_n` times the sum of
/// their magnitudes of the exact sum (Higham, Accuracy and Stability of
/// Numerical Algorithms, §3.1).
pub fn gamma(n: usize) -> f64 {
    let nu = n as f64 * Float::F32.unit();
    assert!(nu < 1.0, "γ_{n} is unbounded");
    nu / (1.0 - nu)
}

/// The most rows a backend's `project_rows` serves with its GEMV launches,
/// where it sums in F32 over unstaged operands (the GEMM runs past it): the
/// rows its first GEMM launch starts past in `dense.seismic`, which
/// `tests/projection_rows.rs` pins.
pub const fn gemv_rows(backend: BackendName) -> usize {
    match backend {
        BackendName::Cpu => usize::MAX,
        BackendName::Metal | BackendName::Cuda => 16,
        BackendName::Vulkan => 8,
    }
}

/// The supremum of `|d/dx x / (1 + e^-x)|`: 1.09984 at `x = 2.3993`.
pub const SILU_SLOPE: f64 = 1.1;

/// `x / (1 + e^-x)` in F64.
pub fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// How far, relative, an evaluation of `x / (1 + exp(-x))` is from its exact
/// value in a library: the exponential, the sum with 1 and the division.
pub fn silu_evaluation(library: &Library) -> f64 {
    (1.0 + library.exp) * (1.0 + Float::F32.unit()) * (1.0 + library.divide) - 1.0
}

/// How far `silu(x)` may move when its input is within `dx` of `x` and each
/// of `libraries` evaluates it: the slope times `dx`, and each evaluation's
/// relative error of the (moved) value.
fn silu_moved(libraries: &[&Library], x: f64, dx: f64) -> f64 {
    let carried = SILU_SLOPE * dx;
    let evaluation = libraries.iter().map(|library| silu_evaluation(library)).sum::<f64>();
    carried + evaluation * (silu_f64(x).abs() + carried)
}

/// How far `R(R(silu(g)) u)` may differ between a native and a host that
/// each evaluate SiLU and multiply in F32, `R` the rounding to `output`, for
/// a host-rounded gate `g` and up `u` the native's are within `dg` and `du`
/// of: `a = R(silu(g))` and `value = R(a u)` are the host's.
pub fn silu_product_between(
    output: Float,
    (native, host): (&Library, &Library),
    (g, dg): (f32, f64),
    (u, du): (f32, f64),
    (a, value): (f32, f32),
) -> f64 {
    let da = output.rounded(silu_moved(&[native, host], f64::from(g), dg), a);
    let (a, u) = (f64::from(a).abs(), f64::from(u).abs());
    let product = da * (u + du) + a * du + 2.0 * Float::F32.unit() * (a + da) * (u + du);
    output.rounded(product, value)
}

/// How far a native's `R(R(silu(R(g̃))) R(ũ))`, `R` the rounding to `output`,
/// the F32 product and `g̃`, `ũ` the sums within `dg` and `du` of the exact
/// `g` and `u`, may be from the exact `silu(g) u`.
pub fn silu_product_from_exact(output: Float, native: &Library, (g, dg): (f64, f64), (u, du): (f64, f64)) -> f64 {
    let gate = output.rounded_from_exact(dg, g);
    let activated = silu_f64(g);
    let a = output.rounded_from_exact(silu_moved(&[native], g, gate), activated);
    let up = output.rounded_from_exact(du, u);
    let product = a * (u.abs() + up) + activated.abs() * up + Float::F32.unit() * (activated.abs() + a) * (u.abs() + up);
    output.rounded_from_exact(product, activated * u)
}

/// How a form holds an operand where it multiplies it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Staging {
    /// As the host does.
    Exact,
    /// Rounded to a format.
    Rounded(Float),
    /// A value F16's significand holds (BF16, F16 or an NVFP4 operand), as
    /// F16: exact within F16's normal range, moved by at most F16's
    /// underflow below it. `scaled`: the row is first scaled by `2^-e`
    /// for the least `e >= 0` that keeps its largest magnitude below 2^15
    /// (Vulkan's BF16 activations), which scales that move by `2^e`.
    IntoF16 { scaled: bool },
}

impl Staging {
    /// How weight `element`'s values are held staged in `to`: a dense
    /// weight's values are of their own format, a packed weight's decoded
    /// to F32.
    pub fn weight(element: Element, to: Float) -> Staging {
        match Float::dense(element) {
            Some(float) => float.staged_as(to),
            None => Staging::Rounded(to),
        }
    }

    /// `(u, η)` with `|staged(v) - v| <= u|v| + η` for each value `v` of a
    /// row whose largest magnitude is `largest`.
    fn error(self, largest: f64) -> (f64, f64) {
        match self {
            Staging::Exact => (0.0, 0.0),
            Staging::Rounded(float) => (float.unit(), float.underflow()),
            Staging::IntoF16 { scaled } => {
                // frexp(largest) = (m, e'), 2^e' <= 2 largest, e = e' - 15.
                let scale = if scaled && largest >= 2f64.powi(15) {
                    largest * 2f64.powi(-14)
                } else {
                    1.0
                };
                (0.0, Float::F16.underflow() * scale)
            }
        }
    }
}

/// The K products of one output, as the bound reads them.
#[derive(Clone, Copy, Debug)]
pub struct Products {
    count: usize,
    /// `Σ|x||w|`.
    magnitude: f64,
    /// `Σ|x|` and `Σ|w|`.
    activations: f64,
    weights: f64,
    /// `max |x|` and `max |w|`: a row's largest magnitude, which a scaled
    /// staging scales it by.
    largest_activation: f64,
    largest_weight: f64,
}

impl Products {
    /// The products of activation row `x` and weight row `w`, each held as
    /// F32 or F64.
    pub fn of<X: Copy + Into<f64>, W: Copy + Into<f64>>(x: &[X], w: &[W]) -> Products {
        assert_eq!(x.len(), w.len(), "one weight per activation");
        x.iter().zip(w).fold(
            Products {
                count: x.len(),
                magnitude: 0.0,
                activations: 0.0,
                weights: 0.0,
                largest_activation: 0.0,
                largest_weight: 0.0,
            },
            |p, (a, b)| {
                let (a, b) = ((*a).into().abs(), (*b).into().abs());
                Products {
                    magnitude: p.magnitude + a * b,
                    activations: p.activations + a,
                    weights: p.weights + b,
                    largest_activation: p.largest_activation.max(a),
                    largest_weight: p.largest_weight.max(b),
                    ..p
                }
            },
        )
    }
}

/// A native form's arithmetic for one output: its K products of operands
/// held as `activation` and `weight` stage them, summed in F32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Form {
    pub activation: Staging,
    pub weight: Staging,
}

impl Form {
    /// Both operands as the host holds them.
    pub const F32: Form = Form {
        activation: Staging::Exact,
        weight: Staging::Exact,
    };

    /// The projection family's form (`project_rows`, `dense_*`,
    /// `attention_*`, `gated_delta_project`, the readouts) on `backend` at
    /// `rows` activation rows of `act`, over a weight resident as `weight`,
    /// at the declarations' default arithmetic (no INT8, PACK or
    /// F16-operand tuning). Paths are under `kernels/kernels/`; a kernel that
    /// changes what it stages changes this:
    /// - CPU (`cpu/lib/projection/stage.rs`, `projection.rs`): both
    ///   operands widened exactly to F32.
    /// - Metal: the GEMVs and batched GEMVs (`M <= 16`) hold both in F32
    ///   (`metal/lib/projection/gemv.h`, `projection.h` `matrix_value`).
    ///   Past 16 rows (`dense.seismic` `project_rows_gemm_small when
    ///   M > 16`) every GEMM keeps the A words unchanged
    ///   (`metal/lib/projection/projection.h` above `gemm_tile`) and rounds
    ///   the weight's F32 values to half (`gemm_store_step`, and the tall
    ///   form's staging `half4(half(even.x), …)`).
    /// - Vulkan: the GEMVs (`M <= 8`) hold both in F32
    ///   (`vulkan/lib/projection/projection.glsl` `element_split8`). The
    ///   GEMM past 8 rows (`dense.seismic` `project_rows_gemm when
    ///   M > 8`) packs both as F16 (`projection_gemm_load_a`,
    ///   `projection_gemm_store_b_of`: `element_pack8(ELEMENT_F16, …)`), a
    ///   BF16 row first scaled by `2^-(e'-15)` when its largest magnitude
    ///   `2^e'·m` reaches 2^15 (`projection_gemm_row_exponent`).
    /// - CUDA: the MMA operands are A's type. A dense weight of A's own type
    ///   is exact and any other is converted to A
    ///   (`seismic/lang/native-library/cuda/packets.cuh` `Dense`). A packed
    ///   weight's codes or table values are exact in A, its scales folded
    ///   in F32, in the GEMVs (`M <= 16`, `cuda/lib/projection/projection.cuh`)
    ///   and in the GEMMs past 16 rows over the formats CUDA's GEMM folds
    ///   ([`Element::cuda_gemm_fold`]); over the rest the GEMM rounds each
    ///   value `code · scale - bias` to A (`projection.cuh`
    ///   `dequantize_pair`).
    pub fn projection(backend: BackendName, rows: usize, act: Float, weight: Element) -> Form {
        let dense = Float::dense(weight);
        let staged = |to: Float| Staging::weight(weight, to);
        match backend {
            BackendName::Cpu => Form::F32,
            BackendName::Metal if rows <= gemv_rows(backend) => Form::F32,
            BackendName::Metal => Form {
                activation: Staging::Exact,
                weight: staged(Float::F16),
            },
            BackendName::Vulkan if rows <= gemv_rows(backend) => Form::F32,
            BackendName::Vulkan => Form {
                activation: match act {
                    Float::Bf16 => Staging::IntoF16 { scaled: true },
                    float => float.staged_as(Float::F16),
                },
                weight: staged(Float::F16),
            },
            BackendName::Cuda => Form {
                activation: Staging::Exact,
                weight: match dense {
                    Some(float) => float.staged_as(act),
                    None if rows <= gemv_rows(backend) || weight.cuda_gemm_fold() => Staging::Exact,
                    None => Staging::Rounded(act),
                },
            },
        }
    }

    /// How far this form's F32 sum may be from the exact `Σ x w`. With
    /// staged operands `|x̂ - x| <= u_x|x| + η_x`, `|ŵ - w| <= u_w|w| + η_w`,
    /// the products move by at most
    /// `D = c Σ|x||w| + η_x(1 + u_w) Σ|w| + η_w(1 + u_x) Σ|x| + K η_x η_w`,
    /// `c = (1 + u_x)(1 + u_w) - 1`, and `Σ|x̂||ŵ| <= Σ|x||w| + D`, so the
    /// native sum is within `γ_K (Σ|x||w| + D) + D` of the exact one.
    pub fn exact(&self, p: &Products) -> f64 {
        let (ux, ex) = self.activation.error(p.largest_activation);
        let (uw, ew) = self.weight.error(p.largest_weight);
        let staged = ((1.0 + ux) * (1.0 + uw) - 1.0) * p.magnitude
            + ex * (1.0 + uw) * p.weights
            + ew * (1.0 + ux) * p.activations
            + p.count as f64 * ex * ew;
        gamma(p.count) * (p.magnitude + staged) + staged
    }

    /// How far this form's F32 sum may be from a host's F32 sum of the same
    /// products in another order, itself within `γ_K Σ|x||w|` of the exact
    /// one.
    pub fn sums(&self, p: &Products) -> f64 {
        self.exact(p) + gamma(p.count) * p.magnitude
    }

    /// How far this form's output may be from the host's `y` when both
    /// round their sums once to `output`.
    pub fn rounded(&self, p: &Products, output: Float, y: f32) -> f64 {
        output.rounded(self.sums(p), y)
    }
}

/// `actual` within `bound` of `expected`, element by element.
/// `expected` is a host model's, held as F32 or F64.
pub fn assert_within<E: Copy + Into<f64> + std::fmt::Display>(
    label: &str,
    actual: &[f32],
    expected: &[E],
    bound: &[f64],
) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    assert_eq!(bound.len(), expected.len(), "{label}: bounds");
    let outside = (0..expected.len())
        .filter(|&i| !((f64::from(actual[i]) - expected[i].into()).abs() <= bound[i]))
        .collect::<Vec<_>>();
    assert!(
        outside.is_empty(),
        "{label}: {} of {} outside the bound; first {}",
        outside.len(),
        expected.len(),
        outside
            .iter()
            .take(8)
            .map(|&i| format!(
                "[{i}] {} vs {} (bound {:.2e})",
                actual[i], expected[i], bound[i]
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

impl Form {
    /// The projection family's matrix-unit GEMM at any row count: the form
    /// [`Form::projection`] gives past every GEMV band. The routed grouped
    /// and combine entries run it for every block, whatever its rows
    /// (`metal/lib/routed/routed.h` `expert_paired` and
    /// `metal/routed_combine.metal` `projection::gemm`;
    /// `vulkan/routed_experts.comp` and `vulkan/routed_combine.comp`
    /// `projection_gemm`; `cuda/lib/routed/routed.cuh` `routed_gemm`).
    pub fn matrix(backend: BackendName, act: Float, weight: Element) -> Form {
        Form::projection(backend, usize::MAX, act, weight)
    }

    /// CUDA's F16-operand GEMM (`OPERAND = 1`,
    /// `cuda/lib/projection/projection.cuh` `F16GemmOperands`): the
    /// activation staged as F16 with no range scaling, and every weight
    /// value as F16 (`EXACT_CODES = false`: packed values rounded, dense
    /// ones converted).
    pub fn cuda_f16_operands(act: Float, weight: Element) -> Form {
        Form {
            activation: act.staged_as(Float::F16),
            weight: Staging::weight(weight, Float::F16),
        }
    }
}

impl Float {
    /// `v` rounded to nearest, ties to even, in this format (overflow is
    /// outside the analysis).
    pub fn round(self, v: f64) -> f64 {
        let (precision, lowest) = match self {
            Float::F32 => (24, -126),
            Float::Bf16 => (8, -126),
            Float::F16 => (11, -14),
        };
        if v == 0.0 {
            return 0.0;
        }
        let exponent = (v.abs().log2().floor() as i32).max(lowest);
        let quantum = 2f64.powi(exponent - (precision - 1));
        (v / quantum).round_ties_even() * quantum
    }

    /// The most two values within `window` (relative) of `v` may differ
    /// once each is rounded to this format: rounding is monotone, so both
    /// lie between the roundings of `v (1 - δ)` and `v (1 + δ)`. Zero away
    /// from a rounding boundary: the rounding flips an order or a library
    /// approximation can cause.
    pub fn flip(self, v: f64, window: f64) -> f64 {
        (self.round(v * (1.0 + window)) - self.round(v * (1.0 - window))).abs()
    }
}

/// The worst-case relative error of a backend's F32 division, reciprocal
/// square root and exponential as its natives are compiled, from each
/// platform's documented accuracy (an ulp of `v` is at most `2u |v|`):
/// - CPU: Rust's correctly rounded `/` and `sqrt`, `rsqrt` as `1 / sqrt`
///   (`seismic/native-cpu/src/math.rs`), and the platform `expf` within
///   1 ulp.
/// - Metal (precise, no fast math: `seismic/backends/metal/src/lib.rs`):
///   the Metal Shading Language's accuracy table, `x / y` within 2.5 ulp,
///   `rsqrt` within 2 ulp, `exp` within 4 ulp.
/// - CUDA (`--prec-div=true --prec-sqrt=true`,
///   `seismic/backends/cuda/src/nvrtc.rs`): `/` correctly rounded, `rsqrtf`
///   and `expf` within 2 ulp (CUDA C Programming Guide, mathematical
///   functions).
/// - Vulkan: `seismic_div_rn` correctly rounded, GLSL's `inversesqrt`
///   within 2 ulp, and the GELU's `precise_exp`
///   (`vulkan/lib/core/precise.glsl`): nine fused Horner steps on
///   `|r| <= ln 2 / 2`, `γ_9 e^{2|r|}`, the degree-8 truncation
///   `|r|^9 / 9! e^{2|r|}`, the reduction's rounding `u`, and `|k| <= 128`
///   times the split ln 2's error.
pub struct Library {
    pub divide: f64,
    pub rsqrt: f64,
    pub exp: f64,
}

impl Library {
    pub fn of(backend: BackendName) -> Library {
        let (u, ulp) = (Float::F32.unit(), 2.0 * Float::F32.unit());
        match backend {
            BackendName::Cpu => Library {
                divide: u,
                rsqrt: (1.0 + u) / (1.0 - u) - 1.0,
                exp: ulp,
            },
            BackendName::Metal => Library {
                divide: 2.5 * ulp,
                rsqrt: 2.0 * ulp,
                exp: 4.0 * ulp,
            },
            BackendName::Cuda => Library {
                divide: u,
                rsqrt: 2.0 * ulp,
                exp: 2.0 * ulp,
            },
            BackendName::Vulkan => {
                let reach = std::f64::consts::LN_2 / 2.0;
                let growth = (2.0 * reach).exp();
                let split =
                    (std::f64::consts::LN_2 - (0.693145751953125 + 1.4286068203094172e-06)).abs();
                let truncation = reach.powi(9) / 362_880.0;
                Library {
                    divide: u,
                    rsqrt: 2.0 * ulp,
                    exp: (gamma(9) + truncation) * growth + u + 128.0 * split,
                }
            }
        }
    }

    /// How far, relative, a native RMS prologue's normalized value
    /// `x rsqrt(Σx² / K + ε) w` over `width` columns may be from its exact
    /// value before its rounding to A: the F32 square sum (`γ_K`, every term
    /// positive), the division, the addition of ε > 0, `rsqrt` (an argument
    /// `t (1 + e)` moves it by `(1 - e)^-1/2`), and the two products.
    pub fn rms(&self, width: usize) -> f64 {
        let u = Float::F32.unit();
        let argument = (1.0 + gamma(width)) * (1.0 + self.divide) * (1.0 + u) - 1.0;
        (1.0 - argument).powf(-0.5) * (1.0 + self.rsqrt) * (1.0 + u).powi(2) - 1.0
    }
}

/// How far apart, per column, a native's and a host's F32 LayerNorm
/// `(x - mean) rsqrt(var + ε) n + b` of one row `x` may be, each in its own
/// library: two F32 sums in different orders for the mean and the variance
/// (`γ_2W`), the centered values' roundings, the division, the reciprocal
/// square root and the products.
pub fn layer_norm_between(
    (native, host): (&Library, &Library),
    x: &[f32],
    norm: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Vec<f64> {
    let w = x.len();
    let (unit, sums) = (Float::F32.unit(), gamma(2 * w));
    let x = x.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
    let mean = x.iter().sum::<f64>() / w as f64;
    let mean_error = sums * x.iter().map(|v| v.abs()).sum::<f64>() / w as f64 + (native.divide + host.divide) * mean.abs();
    let centered = x.iter().map(|v| v - mean).collect::<Vec<_>>();
    let errors = centered.iter().map(|c| mean_error + 2.0 * unit * c.abs()).collect::<Vec<_>>();
    let largest = errors.iter().copied().fold(0.0, f64::max);
    let squares = centered.iter().map(|c| c * c).sum::<f64>();
    let variance = (2.0 * largest * centered.iter().map(|c| c.abs()).sum::<f64>() + w as f64 * largest * largest) / squares
        + sums;
    let argument = (1.0 + variance) * (1.0 + native.divide + host.divide) * (1.0 + 2.0 * unit) - 1.0;
    assert!(argument < 0.5, "the row's variance is too small to bound");
    let reciprocal = (1.0 - argument).powf(-0.5) * (1.0 + native.rsqrt) * (1.0 + host.rsqrt) - 1.0;
    let inverse = 1.0 / (squares / w as f64 + f64::from(epsilon)).sqrt();
    (0..w)
        .map(|c| {
            let weight = f64::from(norm[c]).abs();
            let out = centered[c].abs() * inverse * weight + f64::from(bias[c]).abs();
            (centered[c].abs() + errors[c]) * (1.0 + reciprocal) * (1.0 + unit).powi(2) * inverse * weight
                - centered[c].abs() * inverse * weight
                + 2.0 * unit * out
        })
        .collect()
}

impl Form {
    /// CUDA's INT8 GEMM (`INT8 = 1` past the GEMV band): `stage_row_s8`
    /// (`cuda/lib/projection/stage.cuh`) quantizes each 32 activations to
    /// `d q`, `d = max|x| / 127` and `q = rint(x / d)`, which the host holds as
    /// the F32 product `fl(d q)`; `gemm_tile_s8` (`projection.cuh`)
    /// multiplies the exact s8 codes into exact int32 sums and folds them as
    /// `acc += scale (d P) - bias (d Σq)`. The native's operand `d q` is so
    /// within one F32 rounding of the host's, the weight's codes exact.
    pub fn cuda_int8() -> Form {
        Form {
            activation: Staging::Rounded(Float::F32),
            weight: Staging::Exact,
        }
    }
}
