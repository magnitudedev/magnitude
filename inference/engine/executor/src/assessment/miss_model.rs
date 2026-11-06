//! The generic miss model: how a bank of a share of a layer's experts misses
//! what the layer routes, from the layer's `E` and `k` alone.
//!
//! A row scores all `E` experts and the top `k` win. A score is what persists
//! over a text and what changes with the token, both standard normal,
//! `z = √c·g + √(1−c)·ε`, so an expert of persistence `g` is chosen with
//! `p(g) = Φ((√c·g − τ)/√(1−c))`, `τ = Φ⁻¹(1 − k/E)`.
//!
//! The ranking holds the highest decayed counts. An expert's count, at half-life
//! `h`, has mean `p·W₁` and variance `p(1−p)·W₂`, with `Wₙ = 1/(2^(n/h) − 1)`,
//! and one threshold `θ`, set by the bank's size, decides what is held: `P` the
//! chance that a Gamma of those moments passes `θ`, with `E_g[P] = s` the bank's
//! share. A bank then misses `m(s; c) = 1 − E_g[p·P]/E_g[p]` of what is routed.
//!
//! `c` is the one thing no header holds, and a text moves it. Knowing nothing of
//! it, the model is the mean over `c` uniform on (0, 1), by Gauss-Legendre: no
//! constant is fitted and no model's data enters. Each `(E, k)`'s table is built
//! once in a process, so its cost does not grow with the number of models.

use std::collections::BTreeMap;
use std::sync::{Arc, LazyLock, Mutex};

/// The nodes and weights of `c`, Gauss-Legendre on (0, 1).
const NODES: [(f64, f64); 4] = [
    (0.069_431_844_202_973_7, 0.173_927_422_568_726_9),
    (0.330_009_478_207_571_9, 0.326_072_577_431_273_1),
    (0.669_990_521_792_428_1, 0.326_072_577_431_273_1),
    (0.930_568_155_797_026_3, 0.173_927_422_568_726_9),
];
/// Equal-mass quantiles of `g`.
const QUANTILES: usize = 65;
/// Thresholds `θ` from next to none to past every count's mean.
const THRESHOLDS: usize = 48;
const SMALLEST_THRESHOLD: f64 = 1e-3;
/// The half-life `h` the model's counts decay at, in steps: of those a ranking
/// can settle on (`experts::reuse`), the one a long decode's reads favour.
const HALF_LIFE: f64 = 64.;
/// How far `p` keeps from 0 and 1, where a count's Gamma has no moments.
const CHOSEN_MARGIN: (f64, f64) = (1e-9, 1e-6);

/// The equal-mass quantiles of `g`, one table for every layer.
static PERSISTENCE: LazyLock<[f64; QUANTILES]> =
    LazyLock::new(|| std::array::from_fn(|index| normal_quantile((index as f64 + 0.5) / QUANTILES as f64)));
/// Each `(E, k)`'s model.
static MODELS: LazyLock<Mutex<BTreeMap<(u64, u64), Arc<MissModel>>>> = LazyLock::new(Mutex::default);

/// One `c`'s curve: `misses[i]` at `shares[i]`, the share ascending, and every
/// expert's `p`.
struct Node {
    weight: f64,
    shares: Vec<f64>,
    misses: Vec<f64>,
    chosen: [f64; QUANTILES],
}

/// A layer's generic model ([`MissModel::of`]).
pub struct MissModel {
    choices: f64,
    nodes: Vec<Node>,
}

impl MissModel {
    /// The model of a layer of `experts` experts routing `choices` a row.
    pub fn of(experts: u64, choices: u64) -> Arc<Self> {
        assert!(0 < choices && choices <= experts, "{choices} chosen of {experts} experts");
        let mut models = MODELS.lock().expect("no build of a miss model panics");
        models.entry((experts, choices)).or_insert_with(|| Arc::new(Self::build(experts, choices))).clone()
    }

    fn build(experts: u64, choices: u64) -> Self {
        let share = choices as f64 / experts as f64;
        let wide = 1. / (2f64.powf(1. / HALF_LIFE) - 1.);
        let narrow = 1. / (2f64.powf(2. / HALF_LIFE) - 1.);
        // Past every chosen expert's mean count, nothing is held.
        let widest = wide * 1.05;
        let ratio = (widest / SMALLEST_THRESHOLD).powf(1. / (THRESHOLDS - 1) as f64);
        let threshold = normal_quantile(1. - share);
        let nodes = NODES
            .iter()
            .map(|&(correlation, weight)| {
                let chosen: [f64; QUANTILES] = std::array::from_fn(|index| {
                    normal_cdf((correlation.sqrt() * PERSISTENCE[index] - threshold) / (1. - correlation).sqrt())
                        .clamp(CHOSEN_MARGIN.0, 1. - CHOSEN_MARGIN.1)
                });
                let total = chosen.iter().sum::<f64>();
                // Each count's Gamma: its shape, scale and `ln Γ(shape)`.
                let counts = chosen.map(|p| {
                    let shape = p * wide * wide / ((1. - p) * narrow);
                    (shape, (1. - p) * narrow / wide, log_gamma(shape))
                });
                let mut shares = Vec::with_capacity(THRESHOLDS + 2);
                let mut misses = Vec::with_capacity(THRESHOLDS + 2);
                // No bank misses everything, a whole one nothing.
                shares.push(0.);
                misses.push(1.);
                for step in (0..THRESHOLDS).rev() {
                    let theta = SMALLEST_THRESHOLD * ratio.powi(step as i32);
                    let (mut held, mut caught) = (0., 0.);
                    for (&p, &(shape, scale, log_shape)) in chosen.iter().zip(&counts) {
                        let passes = gamma_upper(shape, log_shape, theta / scale);
                        held += passes;
                        caught += p * passes;
                    }
                    shares.push(held / QUANTILES as f64);
                    misses.push(1. - caught / total);
                }
                shares.push(1.);
                misses.push(0.);
                Node { weight, shares, misses, chosen }
            })
            .collect();
        Self { choices: choices as f64, nodes }
    }

    /// `m̄(s)`: the share of what a row routes that a bank of a share `share`
    /// of the layer's experts lacks.
    pub fn misses(&self, share: f64) -> f64 {
        assert!((0. ..=1.).contains(&share), "a share of a layer's experts: {share}");
        self.nodes.iter().map(|node| node.weight * interpolate(&node.shares, &node.misses, share)).sum()
    }

    /// `q = 1 − (1 − m)^k`: the share of a single row's steps that miss at
    /// all, its `k` choices each missing `misses`.
    pub fn requests(&self, misses: f64) -> f64 {
        1. - (1. - misses).powf(self.choices)
    }

    /// `D(r) = E·(1 − E_g[(1 − p)^r])`, of the table's quantiles scaled to
    /// the `k` a row routes by definition: the distinct experts `rows` rows
    /// route. Persistent experts are chosen by rows together.
    pub fn routed(&self, rows: f64) -> f64 {
        let distinct = |rows: f64| {
            let none = self
                .nodes
                .iter()
                .map(|node| node.weight * node.chosen.iter().map(|p| (1. - p).powf(rows)).sum::<f64>() / QUANTILES as f64)
                .sum::<f64>();
            1. - none
        };
        self.choices * distinct(rows) / distinct(1.)
    }
}

/// `y` at `x` along ascending `xs`.
fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let after = xs.partition_point(|&at| at < x).clamp(1, xs.len() - 1);
    let (low, high) = (xs[after - 1], xs[after]);
    if high <= low {
        return ys[after];
    }
    ys[after - 1] + (ys[after] - ys[after - 1]) * (x - low) / (high - low)
}

/// `Φ(x)`.
fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// `Φ⁻¹(p)`, by bisection: a miss model's table asks it a few dozen times.
fn normal_quantile(p: f64) -> f64 {
    let (mut low, mut high) = (-40f64, 40f64);
    for _ in 0..100 {
        let middle = (low + high) / 2.;
        if normal_cdf(middle) < p {
            low = middle;
        } else {
            high = middle;
        }
    }
    (low + high) / 2.
}

/// The complementary error function, by the regularized incomplete gamma it is.
fn erfc(x: f64) -> f64 {
    let tail = gamma_upper(0.5, log_gamma(0.5), x * x);
    if x >= 0. { tail } else { 2. - tail }
}

/// `ln Γ(x)`, Lanczos.
fn log_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -0.000_005_395_239_384_953,
    ];
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let series = COEFFICIENTS.iter().enumerate().fold(1.000_000_000_190_015, |series, (index, coefficient)| {
        series + coefficient / (x + 1. + index as f64)
    });
    -tmp + (2.506_628_274_631_000_5 * series / x).ln()
}

/// `Q(a, x)`, the regularized upper incomplete gamma, `log_a` being `ln Γ(a)`:
/// its series below `a + 1`, its continued fraction above.
fn gamma_upper(a: f64, log_a: f64, x: f64) -> f64 {
    const CLOSE: f64 = 1e-10;
    const TINY: f64 = 1e-300;
    if x <= 0. {
        return 1.;
    }
    let front = (a * x.ln() - x - log_a).exp();
    if x < a + 1. {
        let (mut term, mut sum, mut next) = (1. / a, 1. / a, a);
        for _ in 0..500 {
            next += 1.;
            term *= x / next;
            sum += term;
            if term.abs() < sum.abs() * CLOSE {
                break;
            }
        }
        (1. - sum * front).clamp(0., 1.)
    } else {
        let mut b = x + 1. - a;
        let mut c = 1. / TINY;
        let mut d = 1. / b;
        let mut fraction = d;
        for index in 1..500 {
            let index = index as f64;
            let an = -index * (index - a);
            b += 2.;
            d = an * d + b;
            d = if d.abs() < TINY { TINY } else { d };
            c = b + an / c;
            c = if c.abs() < TINY { TINY } else { c };
            d = 1. / d;
            let delta = d * c;
            fraction *= delta;
            if (delta - 1.).abs() < CLOSE {
                break;
            }
        }
        (front * fraction).clamp(0., 1.)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model's reference integral (a fine grid in the study behind it, at
    /// no fitted constant), within what the coarser table costs.
    #[test]
    fn the_table_gives_the_models_shares() {
        let cases = [
            (256, 8, 0.2266, 0.210),
            (32, 4, 0.25, 0.285),
            (32, 4, 0.28, 0.254),
            (512, 10, 0.289, 0.155),
            (512, 10, 0.443, 0.092),
            (128, 8, 0.19, 0.288),
            (32, 4, 0.5, 0.112),
            (256, 10, 0.1, 0.387),
            (256, 10, 0.25, 0.202),
            (256, 10, 0.4, 0.119),
            (256, 6, 0.1, 0.344),
            (256, 6, 0.25, 0.183),
            (256, 6, 0.4, 0.109),
            (512, 22, 0.1, 0.397),
            (512, 22, 0.25, 0.206),
            (512, 22, 0.4, 0.121),
        ];
        for (experts, choices, share, reference) in cases {
            let misses = MissModel::of(experts, choices).misses(share);
            assert!((misses / reference - 1.).abs() < 0.03, "{experts} {choices} at {share}: {misses} against {reference}");
        }
    }

    /// A larger bank never misses more, none misses everything, a whole one nothing.
    #[test]
    fn the_curve_falls_from_all_to_none() {
        let model = MissModel::of(256, 8);
        assert_eq!(model.misses(0.), 1.);
        assert_eq!(model.misses(1.), 0.);
        let curve = (0..=100).map(|share| model.misses(share as f64 / 100.)).collect::<Vec<_>>();
        assert!(curve.windows(2).all(|pair| pair[1] <= pair[0]), "{curve:?}");
    }

    /// A single row routes `k`; rows overlap more than if each chose at random.
    #[test]
    fn rows_share_what_persists() {
        let model = MissModel::of(256, 8);
        assert!((model.routed(1.) - 8.).abs() < 1e-9, "{}", model.routed(1.));
        let uniform = 256. * (1. - (1. - 8. / 256f64).powi(8));
        assert!(model.routed(8.) < uniform && model.routed(8.) > 8., "{} of {uniform}", model.routed(8.));
    }

    #[test]
    fn the_functions_are_their_known_values() {
        assert!((normal_cdf(0.) - 0.5).abs() < 1e-9);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-4);
        assert!((normal_quantile(0.975) - 1.96).abs() < 1e-3);
        assert!((gamma_upper(1., log_gamma(1.), 2.) - (-2f64).exp()).abs() < 1e-9);
        assert!((gamma_upper(3., log_gamma(3.), 2.5) - 0.543_813_6).abs() < 1e-6);
    }
}
