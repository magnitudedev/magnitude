//! The host reference of Metal's token-packing arithmetic (`packing.h`), the
//! one the packed forms' tests judge a kernel against: the codes each token
//! takes in each run under its lanes' held scales (`packing_lane_scales`,
//! `packing_code`), multiplied exactly by the weights' values, with each
//! output's budget for what the reference leaves unmirrored. A defect is a
//! difference beyond the budget at any lane count; the form's approximation
//! (the codes' rounding) is the reference's own, which admission judges.
//!
//! The budget covers, per output:
//! - every code, scale and widening choice the host cannot decide exactly
//!   (the device's F32 norms, division and square root against an exact
//!   host, an input's rounding flip or wider uncertainty): a code at a tie
//!   takes either neighbour, a value under an undecided scale or input is
//!   within half a code step of it, and an undecided widening takes the
//!   hull of both;
//! - the run accumulator's F32 roundings, which land in the bottom lane:
//!   none where every partial sum is exact (all its lanes' terms multiples
//!   of 2^g below 2^(24 + g)), else half an ulp of the accumulator's bound
//!   for each product and each addition;
//! - with residue bits, the read of the top sum (floor, borrow and mean
//!   deficit, two of its units) and the low sum's truncation;
//! - the F16 fold (`folded`, in run order): each step's rounding against
//!   its running sum and its terms' magnitudes, F16's subnormal floor below.
//!
//! The entry's store (an F32 residual sum, an activation rounding, an
//! activation function) is its caller's to add (`assert_within`).

// Each test binary that judges a packed form takes the part it needs.
#![allow(dead_code)]

use seismic_lang::forms::MetalPacking;

/// The tokens an operand element of the packed GEMM carries
/// (`packing_gemm_lanes`).
pub const LANES: usize = 2;

/// A packed operand: runs a 32-column block (`folds`), the codes' centre
/// and residue bits (0 without).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Packed {
    pub folds: usize,
    pub centre: u32,
    pub residue: u32,
}

impl Packed {
    /// The operand a launch of residue width `width` (`RESIDUE`) codes for
    /// weights of `packing`.
    pub fn of(packing: MetalPacking, width: u64) -> Self {
        Self {
            folds: packing.folds as usize,
            centre: packing.centre,
            residue: if packing.residue { width as u32 } else { 0 },
        }
    }

    /// Columns of a run.
    pub fn run(self) -> usize {
        32 / self.folds
    }
}

/// The phase of block `g` of `groups` (`packing_phase`): 1 from the second
/// half of the blocks.
fn phase(g: usize, groups: usize) -> usize {
    (2 * g + 1) / groups
}

/// The lane token `position` of an element takes in `phase`
/// (`packing_role`): the first token is the top one in phase 0.
fn role(position: usize, phase: usize) -> usize {
    (1 - position + phase) % 2
}

/// The fields of an element and its run accumulator (`packing_lanes<N>`, or
/// `packing_fields<R>` with residue bits), bottom lane first: each lane's
/// largest code, the largest |S_l| of a run (`limit`), the exponent of its
/// unit in the accumulator, and the accumulator's bound 2^exponent.
pub struct Fields {
    pub codes: Vec<f64>,
    pub limits: Vec<f64>,
    shifts: Vec<i32>,
    pub exponent: i32,
}

impl Fields {
    pub fn new(lanes: usize, residue: u32, n: f64) -> Self {
        if residue != 0 {
            assert_eq!(lanes, 2, "residue bits pack two lanes");
            let top_bit = 16 - residue as i32;
            let top_shift = top_bit + 3;
            return Self {
                codes: vec![2f64.powi(top_shift) - 1.0, 2f64.powi(24 - top_shift) - 1.0],
                limits: vec![(32767.0 - n * 64.0 - 2.0) * 8.0, 2f64.powi(31 - top_bit) - 4.0],
                shifts: vec![-3, top_bit],
                exponent: 31,
            };
        }
        let width = if lanes == 2 { 16 } else { 10 };
        let top_shift = width * (lanes as i32 - 1);
        let exponent = (18 + width).min(31);
        let field = 2f64.powi(width - 1) - 1.0;
        Self {
            codes: (0..lanes)
                .map(|l| if l == lanes - 1 { 2f64.powi(24 - top_shift) - 1.0 } else { 2f64.powi(width) - 1.0 })
                .collect(),
            limits: (0..lanes)
                .map(|l| match l {
                    _ if l == lanes - 1 => 2f64.powi(exponent - top_shift) - 1.0,
                    0 => field - n * 2f64.powi(exponent - 25),
                    _ => field,
                })
                .collect(),
            shifts: (0..lanes as i32).map(|l| l * width).collect(),
            exponent,
        }
    }
}

/// `packing_lane_scale`: lane `l`'s certified scale for a token of a run's
/// norms |x|_1 and |x|_2^2, codes in multiples of `step`, against codes
/// minus `centre`, with the tensors' bound B^2 of the run where given.
pub fn certified_scale(
    fields: &Fields,
    l: usize,
    l1: f64,
    squares: f64,
    bound: Option<f64>,
    centre: f64,
    folds: usize,
    step: f64,
) -> f64 {
    let n = 32.0 / folds as f64;
    let limit = fields.limits[l];
    let unit = 64.0 / 0.9990234375;
    let certified = l1 * unit / (limit / centre - 0.5 * n * step);
    match bound {
        None => certified,
        Some(b2) => certified.min(unit * squares.sqrt() / (limit / b2.sqrt() - 0.5 * n.sqrt() * step)),
    }
}

/// An upper bound on a run's factor, what one unit of its integer codes is
/// worth: its values are factor x integer + one constant, so any two that
/// differ, differ by a multiple of it.
pub fn run_factor(values: &[f32]) -> f64 {
    let mut sorted = values.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    let step = sorted
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|&d| d > 1e-7 * sorted[sorted.len() - 1].abs().max(sorted[0].abs()))
        .fold(f64::INFINITY, f64::min);
    if step.is_finite() { step } else { sorted[0].abs().max(sorted[sorted.len() - 1].abs()) }
}

/// The silences that make a run accumulator exact: each token zero in the
/// blocks where it takes the bottom lane (`Bottom`), or in every other block
/// (`Others`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Silence {
    Bottom,
    Others,
}

/// `x` (`[tokens, k]`) with `silence` applied under the blocks' phases.
pub fn silenced(x: &[f32], k: usize, silence: Silence) -> Vec<f32> {
    let groups = k / 32;
    let mut x = x.to_vec();
    for (t, row) in x.chunks_mut(k).enumerate() {
        for g in 0..groups {
            let bottom = role(t % LANES, phase(g, groups)) == 0;
            if bottom == (silence == Silence::Bottom) {
                row[32 * g..32 * (g + 1)].fill(0.0);
            }
        }
    }
    x
}

/// The packed outputs of a reference (`[tokens, rows]`) and each one's
/// budget.
pub struct Reference {
    pub expected: Vec<f64>,
    pub budget: Vec<f64>,
}

/// The F32 relative margin of the device's scales against the host's: its
/// norms' sums in another order and its division and square root (2.5 and 2
/// ulps), with room.
const SCALE_MARGIN: f64 = 1.0 / 65536.0;
/// The relative margin of a code's gain (an F32 division and product).
const GAIN_MARGIN: f64 = 1.0 / 1048576.0;
/// F16's unit roundoff and its smallest subnormal.
const HALF_ROUNDOFF: f64 = 1.0 / 2048.0;
const HALF_TINY: f64 = 1.0 / 16777216.0;

/// `packing_held`: a scale's F32 fraction rounded up to 8 bits.
fn held(scale: f64) -> f64 {
    f64::from(f32::from_bits(((scale as f32).to_bits() + 0x7fff) & 0xffff_8000))
}

/// An interval `(lo, hi)`.
type Span = (f64, f64);

/// A token's norms of a run as intervals over its inputs' slack.
#[derive(Clone, Copy)]
struct Norms {
    l1: Span,
    peak: Span,
    squares: Span,
}

impl Norms {
    fn of(x: &[f32], slack: &[f64]) -> Self {
        let mut norms = Norms { l1: (0.0, 0.0), peak: (0.0, 0.0), squares: (0.0, 0.0) };
        for (c, &v) in x.iter().enumerate() {
            let s = slack.get(c).copied().unwrap_or(0.0);
            let (lo, hi) = ((f64::from(v).abs() - s).max(0.0), f64::from(v).abs() + s);
            norms.l1 = (norms.l1.0 + lo, norms.l1.1 + hi);
            norms.peak = (norms.peak.0.max(lo), norms.peak.1.max(hi));
            norms.squares = (norms.squares.0 + lo * lo, norms.squares.1 + hi * hi);
        }
        norms
    }
}

/// The held scales a lane may take, as an interval: its certified scale at
/// code step `step` over the norms' interval, capped by its codes
/// (`divisor`), held (monotone in each norm).
fn lane_held(packed: Packed, fields: &Fields, l: usize, norms: Norms, bound: Option<f64>, step: f64, divisor: f64) -> Span {
    let centre = f64::from(packed.centre);
    let certified = |l1, squares| certified_scale(fields, l, l1, squares, bound, centre, packed.folds, step);
    let lo = certified(norms.l1.0, norms.squares.0).max(norms.peak.0 * 64.0 / divisor);
    let hi = certified(norms.l1.1, norms.squares.1).max(norms.peak.1 * 64.0 / divisor);
    (held(lo * (1.0 - SCALE_MARGIN)), held(hi * (1.0 + SCALE_MARGIN)))
}

/// One choice of an element's run: each lane's held scales, the bottom
/// lane's code step (`packing_lane_scales`), and each lane's largest code.
struct Choice {
    scales: Vec<Span>,
    step: f64,
    codes: Vec<f64>,
}

/// An element's run choices (one, or both widenings where the host cannot
/// decide it).
fn choices(packed: Packed, fields: &Fields, norms: &[Norms], bound: Option<f64>) -> Vec<Choice> {
    let top = LANES - 1;
    let narrow = lane_held(packed, fields, top, norms[top], bound, 1.0, fields.codes[top]);
    let wide = lane_held(packed, fields, top, norms[top], bound, 1.0, 2.0 * fields.codes[top] + 1.0);
    let fine = lane_held(packed, fields, 0, norms[0], bound, 1.0, fields.codes[0]);
    let even = lane_held(packed, fields, 0, norms[0], bound, 2.0, fields.codes[0]);
    let (low, high) = (norms[0].l1, norms[top].l1);
    // The comparison's sides as intervals; the device compares in F32, so
    // sides within its rounding take either.
    let side = |a: Span, b: Span, weight: f64| {
        (a.0 * a.0 * low.0 * low.0 + weight * b.0 * b.0 * high.0 * high.0, a.1 * a.1 * low.1 * low.1 + weight * b.1 * b.1 * high.1 * high.1)
    };
    let (keep, wider) = (side(narrow, fine, 1.0), side(wide, even, 4.0));
    let margin = 1.0 + SCALE_MARGIN;
    let widen = [keep.0 <= wider.1 * margin, keep.1 * margin > wider.0];
    [false, true]
        .into_iter()
        .filter(|&wider| widen[usize::from(wider)])
        .map(|wider| {
            let scales = if wider { vec![even, wide] } else { vec![fine, narrow] };
            let mut codes = fields.codes.clone();
            if wider {
                codes[top] = 2.0 * fields.codes[top] + 1.0;
            }
            Choice { scales, step: if wider { 2.0 } else { 1.0 }, codes }
        })
        .collect()
}

/// A token's run under its choices: each column's value (held scale x code
/// / 64) as an interval's midpoint and radius, each column's largest code
/// magnitude, and the largest held scale.
struct Lane {
    mid: Vec<f64>,
    radius: Vec<f64>,
    codes: Vec<f64>,
    scale: f64,
}

/// A token's run `x` (`slack` its inputs' flips) coded under the held
/// scales `scales` at code step `step`, its codes at most `most`: where the
/// scale and inputs are exact, each code is rint's (both neighbours at a
/// tie within the gain's rounding); else each value is within half a code
/// step of its input.
fn lane(x: &[f32], slack: &[f64], scales: Span, step: f64, most: f64) -> Lane {
    let exact = scales.0 == scales.1 && slack.iter().all(|&s| s == 0.0);
    let n = x.len();
    let (mut mid, mut radius, mut codes) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for (c, &v) in x.iter().enumerate() {
        let (v, s) = (f64::from(v), slack.get(c).copied().unwrap_or(0.0));
        if scales.1 == 0.0 {
            mid.push(0.0);
            radius.push(0.0);
            codes.push(0.0);
        } else if exact {
            let t = v * 64.0 / scales.0 / step;
            let (a, b) = ((t * (1.0 - GAIN_MARGIN)).round_ties_even(), (t * (1.0 + GAIN_MARGIN)).round_ties_even());
            let (lo, hi) = (a.min(b) * step, a.max(b) * step);
            mid.push(scales.0 / 64.0 * 0.5 * (lo + hi));
            radius.push(scales.0 / 64.0 * 0.5 * (hi - lo));
            codes.push(lo.abs().max(hi.abs()));
        } else {
            let reach = v.abs() + s;
            mid.push(v);
            radius.push(s + scales.1 * step / 128.0 + reach * GAIN_MARGIN);
            let bound = if scales.0 > 0.0 { (reach * 64.0 / scales.0 * (1.0 + GAIN_MARGIN)).ceil() + step } else { most };
            codes.push(bound.min(most));
        }
    }
    Lane { mid, radius, codes, scale: scales.1 }
}

/// The hull of a token's runs under each choice.
fn hull(lanes: Vec<Lane>) -> Lane {
    let mut lanes = lanes.into_iter();
    let mut all = lanes.next().expect("an element's run has a choice");
    for other in lanes {
        for c in 0..all.mid.len() {
            let lo = (all.mid[c] - all.radius[c]).min(other.mid[c] - other.radius[c]);
            let hi = (all.mid[c] + all.radius[c]).max(other.mid[c] + other.radius[c]);
            all.mid[c] = 0.5 * (lo + hi);
            all.radius[c] = 0.5 * (hi - lo);
            all.codes[c] = all.codes[c].max(other.codes[c]);
        }
        all.scale = all.scale.max(other.scale);
    }
    all
}

/// The reference of `packed` over tokens `x` (`[tokens, k]`, as the operand
/// pass reads them; `slack` each one's rounding flip or wider uncertainty,
/// empty where exact) against `weights` (`[rows, k]` values), under the
/// bound `bound` (B^2 per run: the largest over the weights the operand
/// serves) where given.
pub fn reference(
    packed: Packed,
    k: usize,
    x: &[f32],
    slack: &[f64],
    weights: &[f32],
    bound: Option<&[f32]>,
) -> Reference {
    let n = packed.run();
    let (tokens, rows, runs, groups) = (x.len() / k, weights.len() / k, k / n, k / 32);
    let fields = Fields::new(LANES, packed.residue, n as f64);
    let centre = f64::from(packed.centre);
    // Each row's factor bound and largest magnitude a run, and the fold's
    // power of two over its factors (`weight_scales`), bounded above.
    let factor = (0..rows * runs).map(|i| run_factor(&weights[i * n..(i + 1) * n])).collect::<Vec<_>>();
    let largest = (0..rows * runs)
        .map(|i| weights[i * n..(i + 1) * n].iter().fold(0f64, |a, &v| a.max(f64::from(v).abs())))
        .collect::<Vec<_>>();
    let weight_scale = (0..rows)
        .map(|j| {
            (0..runs)
                .map(|r| factor[j * runs + r] * 512.0 + 128.0 * (largest[j * runs + r] + centre * factor[j * runs + r]))
                .fold(0.0, f64::max)
                * 2.0
        })
        .collect::<Vec<_>>();
    let mut expected = vec![0.0; tokens * rows];
    let mut budget = vec![0.0; tokens * rows];
    let lanes = LANES;
    let zero = vec![0f32; n];
    let none: &[f64] = &[];
    for element in 0..tokens.div_ceil(lanes) {
        // Per run: each position's lane and run, and the accumulator's
        // rounding in the bottom token's output units over its factor.
        let mut element_runs: Vec<Vec<(usize, Lane)>> = Vec::with_capacity(runs);
        let mut rounding = Vec::with_capacity(runs);
        let mut largest_scale = vec![0f64; lanes];
        for r in 0..runs {
            let phase_now = phase(r * n / 32, groups);
            // Position p's columns of the run and their slack (zero past the
            // tokens).
            let input = |p: usize| {
                let t = lanes * element + p;
                if t >= tokens {
                    return (&zero[..], none);
                }
                let at = t * k + r * n;
                (&x[at..at + n], if slack.is_empty() { none } else { &slack[at..at + n] })
            };
            // The position of each lane's token.
            let position = |l: usize| (0..lanes).find(|&p| role(p, phase_now) == l).unwrap();
            let norms = (0..lanes)
                .map(|l| {
                    let (v, s) = input(position(l));
                    Norms::of(v, s)
                })
                .collect::<Vec<_>>();
            let choices = choices(packed, &fields, &norms, bound.map(|b| f64::from(b[r])));
            let mut by_lane = Vec::with_capacity(lanes);
            let mut magnitude = vec![0f64; n];
            let mut nonzero = i32::MAX;
            for l in 0..lanes {
                let p = position(l);
                let (v, s) = input(p);
                let run = hull(
                    choices
                        .iter()
                        .map(|choice| lane(v, s, choice.scales[l], if l == 0 { choice.step } else { 1.0 }, choice.codes[l]))
                        .collect(),
                );
                for c in 0..n {
                    magnitude[c] += run.codes[c] * 2f64.powi(fields.shifts[l]);
                    if run.codes[c] > 0.0 {
                        nonzero = nonzero.min(fields.shifts[l]);
                    }
                }
                largest_scale[p] = largest_scale[p].max(run.scale);
                by_lane.push((p, run));
            }
            // The accumulator's bound (|code - centre| <= centre); its
            // products and additions round only where a partial sum can
            // leave F32's exact integers of its granularity.
            let reach = magnitude.iter().sum::<f64>() * centre;
            let mut error = 0.0;
            if nonzero != i32::MAX && reach >= 2f64.powi(24 + nonzero) {
                error = 2.0 * n as f64 * 2f64.powi(reach.log2().floor() as i32 - 24);
            }
            // In bottom-lane units, with the residue's low truncation (one
            // accumulator unit).
            let mut bottom = error * 2f64.powi(-fields.shifts[0]);
            if packed.residue != 0 {
                bottom += 8.0;
            }
            rounding.push(bottom * by_lane[0].1.scale / 64.0);
            element_runs.push(by_lane);
        }
        // The fold's power of two over each token's scales (with residue
        // bits, its factors carry 2^R and 8), bounded above.
        let token_scale = largest_scale.iter().map(|s| 2.0 * 8.0 * s).collect::<Vec<_>>();
        for j in 0..rows {
            let row = &weights[j * k..(j + 1) * k];
            for p in 0..lanes {
                let t = lanes * element + p;
                if t >= tokens {
                    continue;
                }
                let (mut sum, mut error) = (0f64, 0f64);
                let floor = 4.0 * HALF_TINY * token_scale[p] * weight_scale[j];
                for r in 0..runs {
                    let w = &row[r * n..(r + 1) * n];
                    let f = factor[j * runs + r];
                    let (l, (_, run)) = element_runs[r].iter().enumerate().find(|(_, (owner, _))| *owner == p).unwrap();
                    let (mut centre_sum, mut radius, mut codes) = (0f64, 0f64, 0f64);
                    for c in 0..n {
                        centre_sum += f64::from(w[c]) * run.mid[c];
                        radius += f64::from(w[c]).abs() * run.radius[c];
                        codes += run.codes[c];
                    }
                    sum += centre_sum;
                    error += radius;
                    if l == 0 {
                        error += rounding[r] * f;
                    }
                    if l == lanes - 1 && packed.residue != 0 {
                        error += 2f64.powi(packed.residue as i32 + 1) * run.scale / 64.0 * f;
                    }
                    // The run's terms: its scale's factor of the sum, and
                    // its min term, each at most this.
                    let most = run.scale / 64.0 * codes * (2.0 * centre * f + largest[j * runs + r]);
                    error += HALF_ROUNDOFF * (sum.abs() + error) + 4.0 * HALF_ROUNDOFF * most + floor;
                }
                expected[t * rows + j] = sum;
                budget[t * rows + j] = error;
            }
        }
    }
    Reference { expected, budget }
}

/// The elementwise largest of packing bounds (`packing_bounds::at`: an
/// operand's bound of a run is the largest of the weights' it serves).
pub fn served_bound(bounds: &[&[f32]]) -> Vec<f32> {
    let runs = bounds.iter().map(|b| b.len()).max().unwrap_or(0);
    (0..runs).map(|r| bounds.iter().filter_map(|b| b.get(r)).copied().fold(0.0, f32::max)).collect()
}

/// Asserts each of `actual` is within what `within` makes of its reference
/// value and budget (output `i`, the expected value, its budget: the
/// entry's store on top), and returns the largest ratio of a difference to
/// that.
pub fn assert_within(
    label: &str,
    reference: &Reference,
    actual: &[f64],
    within: impl Fn(usize, f64, f64) -> f64,
) -> f64 {
    assert_eq!(reference.expected.len(), actual.len(), "{label}: outputs");
    let mut worst = 0f64;
    for (i, (&value, (&expected, &budget))) in
        actual.iter().zip(reference.expected.iter().zip(&reference.budget)).enumerate()
    {
        let within = within(i, expected, budget);
        let difference = (value - expected).abs();
        assert!(
            difference <= within,
            "{label}: output {i} is {value:.6e}, the packed reference {expected:.6e} within {within:.3e} (difference {difference:.3e})"
        );
        if within > 0.0 {
            worst = worst.max(difference / within);
        }
    }
    worst
}
