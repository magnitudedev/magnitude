// Shared pieces of the Qwen3-VL vision entries on CPU (`qwen_vision_stem`,
// `qwen_vision_block`, `qwen_vision_merger`; contracts and portable bodies in
// vision.seismic). The counterpart of `metal/lib/vision/vision.h`,
// `cuda/lib/vision/vision.cuh` and `vulkan/lib/vision/vision.glsl`: the
// projections with their bias epilogues run on the projection library's
// row-block driver; the layer norm, the 2D rotary embedding, the full
// (non-causal) attention and the two GELU forms are the vision-specific
// pieces.
//
// Numerics: the residual stream is F32; the published intermediates
// (normalized rows, projections feeding a projection or the attention,
// rotated queries and keys, the attention output, activations) are the F32
// computation rounded once to the activation element A, held as F32 in
// scratch. The scalar functions follow the portable bodies operation by
// operation (unfused, as the portable arithmetic is written).

use super::super::core::{activation, reduce};
use super::super::projection::projection;
use seismic::cpu::{Dense, Weights};

// ---------------------------------------------------------------------------
// Scalar functions.

/// The portable `gelu_tanh`, with `tanh(z) = 2 / (1 + exp(-2z)) - 1`.
#[inline(always)]
pub fn gelu_tanh(value: f32) -> f32 {
    let argument = 0.797_884_560_802_865_4_f32 * (value + 0.044715 * value * value * value);
    let hyperbolic = 2.0 / (1.0 + (-2.0 * argument).exp()) - 1.0;
    0.5 * value * (1.0 + hyperbolic)
}

/// The portable `erf_values` of one value: the rational erf of
/// rust-lang/libm `erff` (FreeBSD msun / SunPro) in F32, saturating at
/// |x| >= 6 and propagating NaN.
#[inline(always)]
pub fn erf(x: f32) -> f32 {
    let magnitude = x.abs();
    if magnitude < 0.84375 {
        if magnitude < 3.725290298461914e-9 {
            return 0.125 * (8.0 * x + 1.0270333290e+00 * x);
        }
        let z = x * x;
        let numerator = 1.2837916613e-01
            + z * (-3.2504209876e-01 + z * (-2.8481749818e-02 + z * (-5.7702702470e-03 + z * (-2.3763017452e-05))));
        let denominator = 1.0
            + z * (3.9791721106e-01
                + z * (6.5022252500e-02
                    + z * (5.0813062117e-03 + z * (1.3249473704e-04 + z * (-3.9602282413e-06)))));
        return x + x * (numerator / denominator);
    }
    let mut positive = 1.0f32;
    if magnitude < 1.25 {
        let s = magnitude - 1.0;
        let numerator = -2.3621185683e-03
            + s * (4.1485610604e-01
                + s * (-3.7220788002e-01
                    + s * (3.1834661961e-01
                        + s * (-1.1089469492e-01 + s * (3.5478305072e-02 + s * (-2.1663755178e-03))))));
        let denominator = 1.0
            + s * (1.0642088205e-01
                + s * (5.4039794207e-01
                    + s * (7.1828655899e-02
                        + s * (1.2617121637e-01 + s * (1.3637083583e-02 + s * (1.1984500103e-02))))));
        let complement = (1.0 - 8.4506291151e-01) - numerator / denominator;
        positive = 1.0 - complement;
    } else if magnitude < 6.0 {
        let s = 1.0 / (magnitude * magnitude);
        let (numerator, denominator) = if magnitude < 2.857142925262451 {
            (
                -9.8649440333e-03
                    + s * (-6.9385856390e-01
                        + s * (-1.0558626175e+01
                            + s * (-6.2375331879e+01
                                + s * (-1.6239666748e+02
                                    + s * (-1.8460508728e+02 + s * (-8.1287437439e+01 + s * (-9.8143291473e+00))))))),
                1.0 + s
                    * (1.9651271820e+01
                        + s * (1.3765776062e+02
                            + s * (4.3456588745e+02
                                + s * (6.4538726807e+02
                                    + s * (4.2900814819e+02
                                        + s * (1.0863500214e+02 + s * (6.5702495575e+00 + s * (-6.0424413532e-02)))))))),
            )
        } else {
            (
                -9.8649431020e-03
                    + s * (-7.9928326607e-01
                        + s * (-1.7757955551e+01
                            + s * (-1.6063638306e+02
                                + s * (-6.3756646729e+02 + s * (-1.0250950928e+03 + s * (-4.8351919556e+02)))))),
                1.0 + s
                    * (3.0338060379e+01
                        + s * (3.2579251099e+02
                            + s * (1.5367296143e+03
                                + s * (3.1998581543e+03
                                    + s * (2.5530502930e+03 + s * (4.7452853394e+02 + s * (-2.2440952301e+01))))))),
            )
        };
        // The truncation has at most 11 significant bits on [1.25, 6), so
        // z * z is exact; the residual restores the discarded fraction.
        let z = (magnitude * 256.0) as i32 as f32 / 256.0;
        let complement =
            (-z * z - 0.5625).exp() * ((z - magnitude) * (z + magnitude) + numerator / denominator).exp() / magnitude;
        positive = 1.0 - complement;
    }
    if x.is_nan() {
        x
    } else if x < 0.0 {
        -positive
    } else {
        positive
    }
}

/// The portable erf-form `gelu`: `0.5 * x * (1 + erf(x / sqrt(2)))`.
#[inline(always)]
pub fn gelu_erf(value: f32) -> f32 {
    0.5 * value * (1.0 + erf(value * 0.707_106_781_186_547_6))
}

// ---------------------------------------------------------------------------
// Layer norm.

/// The portable `layer_norm` of one F32 row with the decoded `weight` and
/// `bias`: two-pass centered F32 statistics, each value published to `A`.
#[inline(always)]
pub fn layer_norm<A: Dense>(x: &[f32], weight: &[f32], bias: &[f32], epsilon: f32, out: &mut [f32]) {
    let width = x.len() as f32;
    let out = &mut out[..x.len()];
    let mean = reduce::sum(x) / width;
    for (target, value) in out.iter_mut().zip(x) {
        *target = value - mean;
    }
    let inverse = 1.0 / (reduce::sum_squares(out) / width + epsilon).sqrt();
    for ((target, w), b) in out.iter_mut().zip(&weight[..x.len()]).zip(&bias[..x.len()]) {
        *target = activation::publish::<A>(*target * inverse * w + b);
    }
}

// ---------------------------------------------------------------------------
// Projections.

/// A decoded bias (or norm) vector: every value of the rank-1 operand.
#[inline(always)]
pub fn decode_vector<'a>(vector: &Weights<'_>, out: &'a mut [f32]) -> &'a [f32] {
    let out = &mut out[..vector.k()];
    vector.decode_row(0, out);
    out
}

/// The projections of weight rows `rows` against each of the staged F32
/// activation rows `x` (`k` values each), with the decoded `bias` (one value
/// per weight row) added in F32: `emit(row, first, values)` receives
/// `x[row] · w + bias` for the weight rows `first..first + values.len()`.
#[inline(always)]
pub fn project_bias(
    weights: &Weights<'_>,
    bias: &[f32],
    rows: std::ops::Range<usize>,
    x: &[f32],
    mut emit: impl FnMut(usize, usize, &[f32]),
) {
    let k = weights.k();
    let mut projected = [0.0f32; projection::MAX_ROWS];
    let projected = &mut projected[..rows.len()];
    for (row, x) in x.chunks_exact(k).enumerate() {
        projection::project(weights, rows.start, x, projected);
        for (value, bias) in projected.iter_mut().zip(&bias[rows.clone()]) {
            *value += bias;
        }
        emit(row, rows.start, projected);
    }
}

// ---------------------------------------------------------------------------
// 2D rotary embedding.

/// The portable `qwen_vision_rotate` of one head row of width 4P, in place,
/// each value published to `A`: column i < 2P pairs with i + 2P; pair
/// p = i % 2P turns by coordinates[p / P] * 10000^(-(p % P) / P).
#[inline(always)]
pub fn rotate<A: Dense>(head: &mut [f32], coordinates: [i32; 2], quarter: usize) {
    let half = 2 * quarter;
    let (first, second) = head[..2 * half].split_at_mut(half);
    for (pair, (x, partner)) in first.iter_mut().zip(second.iter_mut()).enumerate() {
        let frequency = (pair % quarter) as f32;
        let angle = coordinates[pair / quarter] as f32 * (-(10000f32.ln()) * frequency / quarter as f32).exp();
        let (s, c) = seismic::cpu::math::sin_cos(angle);
        let (a, b) = (*x, *partner);
        *x = activation::publish::<A>(a * c - b * s);
        *partner = activation::publish::<A>(b * c + a * s);
    }
}

// ---------------------------------------------------------------------------
// Full attention.

/// The portable `qwen_vision_full_attention` of one query row of one head
/// over every key: `key(j)` and `value(j)` are the head rows of patch row j
/// of `keys` patch rows, `scores` holds `keys` values, and `out` receives
/// the attention output published to `A`. One online-softmax update over the
/// whole history: scores accumulate the A-valued products in F32, the
/// probabilities are exp(s * scale - max(s) * scale), and the value product
/// visits the history in ascending order.
#[inline(always)]
pub fn attend<'k, A: Dense>(
    query: &[f32],
    keys: usize,
    key: impl Fn(usize) -> &'k [f32],
    value: impl Fn(usize) -> &'k [f32],
    scores: &mut [f32],
    out: &mut [f32],
) {
    let width = query.len();
    let scale = 1.0 / (width as f32).sqrt();
    let scores = &mut scores[..keys];
    for (j, score) in scores.iter_mut().enumerate() {
        *score = reduce::dot(query, key(j));
    }
    let maximum = f32::NEG_INFINITY.max(reduce::max(scores) * scale);
    for score in scores.iter_mut() {
        *score = (*score * scale - maximum).exp();
    }
    let denominator = reduce::sum(scores);
    let out = &mut out[..width];
    out.fill(0.0);
    for (j, probability) in scores.iter().enumerate() {
        for (acc, v) in out.iter_mut().zip(value(j)) {
            *acc += probability * v;
        }
    }
    for acc in out.iter_mut() {
        *acc = activation::publish::<A>(*acc / denominator);
    }
}
