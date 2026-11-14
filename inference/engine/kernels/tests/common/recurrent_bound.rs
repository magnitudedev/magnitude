//! How far a recurrent entry (the gated delta rule and Kimi's per-channel
//! form, step or chunk) may stray from the f64 host model of its contract:
//! a rounding analysis of the arithmetic the entry performs, per output,
//! never a tolerance. It builds on [`super::bound`] (`Float`, `gamma`,
//! `Library`) and answers, for each backend's form, how far every gated
//! output, published state element and tape element may be from the host's.
//!
//! # The delta rule's error is carried by norm
//!
//! A state row `s` (W columns of one value channel) advances by
//!
//! ```text
//! s' = d ∘ s         (decay, one factor per key channel, 0 < d <= 1)
//! m  = k · s'
//! r  = (v - m) β     (the residual, β in (0, 1])
//! s'' = s' + r k     (= s' (I - β k kᵀ) + β v k)
//! o  = q · s''
//! ```
//!
//! `‖k‖² = Σy² / (Σy² + ε) < 1` (the L2 norm of the convolved key channels
//! `y`, with ε > 0), so `I - β k kᵀ` has eigenvalues in `(0, 1]` and the map
//! `s ↦ s'' ` is non-expansive in the 2-norm up to the decay `λ = max d`.
//! The error `e = ‖ŝ - s‖₂` of a state row therefore obeys
//! `e ← λ e + L`, where `L` collects what a row adds locally: the decay's and
//! the inputs' errors acting on this row's exact trajectory, and the rounding
//! of the row's own arithmetic. A componentwise propagation does not: the
//! map's absolute row sums are about 2 at W = 128, and a per-element
//! recurrence grows as 2^rows. The per-element bound of a state element is
//! `e`; only the local terms are componentwise.
//!
//! The local terms are first order in the unit roundoff `u = 2^-24` except
//! where a second-order product is carried explicitly (the hatted norms).
//!
//! # The forms
//!
//! - Step (every backend's `*_step`; the chunk entries' short slots, rows
//!   after the stop row and, on Vulkan, every row: its `chunk_scan` is the
//!   step's row-sequential scan with the state on lanes, a different order
//!   of the same F32 sums): per row, `m` and `o` are F32 sums of W products
//!   in some order, `γ_W` of their magnitude; the rest are single roundings.
//! - Metal's chunk: the WY form over pieces of `METAL_CHUNK_PIECE` rows in F32 on 8x8 matrix
//!   units, `T = (I + A)^-1` by `(I - A)(I + A²)(I + A⁴)`.
//! - CUDA's chunk: the same form over pieces of `CUDA_CHUNK_PIECE` rows with `T` by forward
//!   substitution; operands on f16 tensor cores with F32 accumulation: `V`,
//!   `Tb`, `D`, `Q` and the output's `U` as one f16 (2^-11 relative), `S`,
//!   `K`, `Z` and the state update's `U` and `K'` as a high and a low f16
//!   (2^-22 relative), and every f16 element in the subnormal range held to
//!   2^-25 absolutely.
//!
//! The chunk forms compute the same real map as the step, so the incoming
//! error and the inputs' errors propagate through the step's recursion; only
//! the rounding terms are the form's, computed once per piece from the piece's
//! magnitudes (`|Tb|`, `|Z|` amplify products of operand roundings).
//!
//! # CUDA's chunk state error
//!
//! The state leaves a piece as `S' = S diag(f) + Uᵀ K'` with `U = Tb Z`. The
//! split operands (`S`, `K`, `Z`, the update's `U` and `K'`) carry 2^-22, as
//! an F32 form does; the single f16 `Y`, `D`, `Q` and output `U` reach only
//! the outputs. What reaches the state at 2^-11 is `Tb` and `V` (one f16
//! each, plus the 2^-25 floor): `ΔU = Σ_s (2^-11 |Tb_ts| |Z_s| + ΔV_s |Tb_ts|)`,
//! which `S'` adds at the size of its own increments. A CPU model of the form
//! with each class of operand rounded alone puts 2.5e-3 of the state's RMS on
//! `V`, 2.1e-3 on `Tb`, and about 1e-6 on everything else: the margin the
//! hand bound (1.5e-2) left is these two roundings, three to four orders
//! above an F32 form's.

#![allow(dead_code)]

use super::bound::{gamma, Float, Library};
use magnitude_kernels::constants;
use seismic::BackendName;

/// Rows at or below which a slot advances row-sequentially with the step's
/// bits on every backend, and the WY form's piece of Metal's and CUDA's
/// chunks: the module constants of `recurrent.seismic`, which the kernels
/// read as `SEISMIC_CONST_*`.
pub const SEQUENTIAL_ROWS: usize = constants::SEQUENTIAL_ROWS as usize;
pub const METAL_CHUNK_PIECE: usize = constants::METAL_CHUNK_PIECE as usize;
pub const CUDA_CHUNK_PIECE: usize = constants::CUDA_CHUNK_PIECE as usize;

const U: f64 = 1.0 / 16_777_216.0;
/// An f16's unit roundoff and half its subnormal step.
const U16: f64 = 1.0 / 2048.0;
const ETA16: f64 = 1.0 / 33_554_432.0;
/// An F32 value held as a high and a low f16 (`low = f16(x - high)`) is
/// within `2^-22 |x| + η` of `x`.
const SPLIT: f64 = 1.0 / 4_194_304.0;

/// The form a backend's entry computes its rows in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Native {
    /// A `*_step` entry.
    Step(BackendName),
    /// A `*_chunk` entry.
    Chunk(BackendName),
    /// A `*_chunk` entry over the first rows of a slot, the rest row by row
    /// from the state it published (the rows of its decode calls).
    Prefill(BackendName, usize),
}

impl Native {
    fn backend(self) -> BackendName {
        match self {
            Native::Step(b) | Native::Chunk(b) | Native::Prefill(b, _) => b,
        }
    }

    /// The rows of `slot` the entry advances in pieces: those before its stop
    /// row, in a slot of more than [`SEQUENTIAL_ROWS`] rows, where the
    /// backend's chunk is a WY form; the others advance row by row.
    fn chunked(self, slot: &SlotTrace) -> usize {
        let rows = match self {
            Native::Step(_) => 0,
            Native::Chunk(_) => slot.rows,
            Native::Prefill(_, rows) => rows,
        };
        if rows > SEQUENTIAL_ROWS && slot.stop > 0 && self.pieces().is_some() {
            rows.min(slot.stop)
        } else {
            0
        }
    }

    /// The WY form's pieces, `None` where the entry advances row by row.
    fn pieces(self) -> Option<Pieces> {
        match self {
            Native::Chunk(BackendName::Metal) | Native::Prefill(BackendName::Metal, _) => {
                Some(Pieces {
                    size: METAL_CHUNK_PIECE,
                    operands: Operands::F32,
                    inverse: Inverse::Polynomial,
                })
            }
            Native::Chunk(BackendName::Cuda) | Native::Prefill(BackendName::Cuda, _) => {
                Some(Pieces {
                    size: CUDA_CHUNK_PIECE,
                    operands: Operands::F16,
                    inverse: Inverse::Substitution,
                })
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Pieces {
    size: usize,
    operands: Operands,
    inverse: Inverse,
}

#[derive(Clone, Copy)]
enum Inverse {
    Polynomial,
    Substitution,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Operands {
    F32,
    F16,
}

/// How an operand reaches a matrix unit.
#[derive(Clone, Copy)]
enum Staged {
    /// As F32.
    Exact,
    /// As one f16.
    Single,
    /// As a high and a low f16.
    Split,
}

impl Operands {
    /// The most `staged` moves one element `x`.
    fn element(self, staged: Staged, x: f64) -> f64 {
        match (self, staged) {
            (Operands::F32, _) | (_, Staged::Exact) => 0.0,
            (_, Staged::Single) => U16 * x.abs() + ETA16,
            (_, Staged::Split) => SPLIT * x.abs() + ETA16,
        }
    }

    /// The 2-norm of the error of a W-vector of 2-norm `norm` staged.
    fn vector(self, staged: Staged, norm: f64, width: usize) -> f64 {
        match (self, staged) {
            (Operands::F32, _) | (_, Staged::Exact) => 0.0,
            (_, Staged::Single) => U16 * norm + ETA16 * (width as f64).sqrt(),
            (_, Staged::Split) => SPLIT * norm + ETA16 * (width as f64).sqrt(),
        }
    }

    /// How far a dot product of W-vectors of 2-norms `a` and `b` (the
    /// operand staged as `sa`, `sb`) is from the exact one, before the F32
    /// accumulation; a split pair's product omits the low × low part.
    fn dot(self, sa: Staged, a: f64, sb: Staged, b: f64, width: usize) -> f64 {
        let (ea, eb) = (self.vector(sa, a, width), self.vector(sb, b, width));
        let mut error = ea * b + a * eb + ea * eb;
        if matches!((sa, sb), (Staged::Split, Staged::Split)) && self == Operands::F16 {
            let low = |norm: f64| U16 * norm + ETA16 * (width as f64).sqrt();
            error += low(a) * low(b);
        }
        error
    }
}

/// The documented accuracy of a backend's F32 library as the recurrent
/// entries call it, on top of [`Library`] (`divide`, `rsqrt`): the plain
/// `exp` and `log` where Vulkan's recurrent kernels do not use its
/// `precise_exp`.
/// - CPU: the platform `expf`, `logf` within 1 ulp.
/// - Metal (precise): `exp` and `log` within 4 ulp (`Library`'s `exp`).
/// - CUDA: `expf` within 2 ulp, `logf` within 1 ulp.
/// - Vulkan: `exp(x)` within `3 + 2|x|` ulp, `log(x)` within 3 ulp outside
///   `[0.5, 2]` and absolutely 2^-21 inside (the Vulkan specification's
///   precision of SPIR-V instructions).
#[derive(Clone, Copy)]
pub struct Functions {
    backend: BackendName,
    pub divide: f64,
    pub rsqrt: f64,
    exp: f64,
}

impl Functions {
    pub fn of(backend: BackendName) -> Functions {
        let library = Library::of(backend);
        let exp = match backend {
            // `Library`'s Vulkan `exp` is the GELU's `precise_exp`.
            BackendName::Vulkan => 0.0,
            _ => library.exp,
        };
        Functions {
            backend,
            divide: library.divide,
            rsqrt: library.rsqrt,
            exp,
        }
    }

    /// The relative error of `exp(x)`.
    pub fn exp(&self, x: f64) -> f64 {
        match self.backend {
            BackendName::Vulkan => (3.0 + 2.0 * x.abs()) * 2.0 * U,
            _ => self.exp,
        }
    }

    /// The absolute error of `log(v)`, `v` in `[1, 2]`.
    fn log(&self, v: f64) -> f64 {
        let ulp = 2.0 * U;
        match self.backend {
            BackendName::Cpu | BackendName::Cuda => ulp * v.ln(),
            BackendName::Metal => 4.0 * ulp * v.ln(),
            BackendName::Vulkan => 1.0 / 2_097_152.0,
        }
    }

    /// The relative error of `x / (1 + exp(-x))` (SiLU) and of
    /// `1 / (1 + exp(-x))` (sigmoid) at `x`: `exp`, the addition, the
    /// division.
    fn logistic(&self, x: f64) -> f64 {
        (1.0 + self.exp(x)) * (1.0 + U) * (1.0 + self.divide) - 1.0
    }
}

/// One state row's quantities at a row of the trajectory, from the f64 host.
#[derive(Clone, Copy, Default)]
pub struct StateRow {
    /// `‖s‖` before the row's decay.
    pub before: f32,
    /// `‖s'‖`.
    pub decayed: f32,
    /// `Σ|s'_c k_c|`.
    pub abs_k: f32,
    /// `|v - m|`.
    pub residual: f32,
    /// `‖s''‖`.
    pub after: f32,
    /// `Σ|q_c s''_c|`.
    pub abs_q: f32,
    /// `|o|`.
    pub output: f32,
}

/// One row's inputs and trajectory, from the f64 host.
pub struct RowTrace {
    /// The causal convolution's `Σ|w x|` per channel.
    pub abs_convolution: Vec<f32>,
    /// The convolution's sum per channel, SiLU's argument.
    pub convolved: Vec<f32>,
    /// SiLU of the convolution per channel.
    pub activated: Vec<f32>,
    /// The prepared channels: q and k L2-normalized (q scaled by W^-1/2),
    /// v as activated.
    pub prepared: Vec<f32>,
    /// β's argument and β, per value head.
    pub beta_input: Vec<f64>,
    pub beta: Vec<f64>,
    /// The decay inputs' `a + time_bias` and log decays, per value head and
    /// decay input.
    pub shifted: Vec<f64>,
    pub log_decay: Vec<f64>,
    /// The output gate's input z, per value head and channel.
    pub gate: Vec<f32>,
    /// Per value head and state row.
    pub state: Vec<StateRow>,
}

pub struct SlotTrace {
    pub first: usize,
    pub rows: usize,
    pub stop: usize,
    pub previous: usize,
    pub following: usize,
    /// The version's replayed tape rows.
    pub taped: usize,
    /// Per replayed entry, value head and state row: the entry's largest
    /// decay factor, `‖s'‖` and `‖s''‖` of the replay's update.
    pub replay: Vec<[f32; 3]>,
}

pub struct Shape {
    pub key_heads: usize,
    pub value_heads: usize,
    pub width: usize,
    pub decay_width: usize,
    pub convolution: usize,
    pub tape: usize,
    pub banks: usize,
    pub grouped: bool,
    pub activation: Float,
    pub epsilon: f64,
    pub rms_epsilon: f64,
    pub sigmoid_gate: bool,
}

impl Shape {
    fn channels(&self) -> usize {
        (2 * self.key_heads + self.value_heads) * self.width
    }
    fn key_head(&self, head: usize) -> usize {
        if self.grouped {
            head * self.key_heads / self.value_heads
        } else {
            head % self.key_heads
        }
    }
    fn tape_row(&self) -> usize {
        (self.value_heads + self.key_heads) * self.width + self.value_heads * self.decay_width
    }
}

/// The f64 host's trajectory of one case, from which any native form's
/// bounds follow.
pub struct Trace {
    pub shape: Shape,
    /// The bounded decay's lower bound (Kimi's form); the softplus decay
    /// without.
    pub decay_lower: Option<f64>,
    pub rate: Vec<f64>,
    /// The recurrent norm's weights.
    pub norm: Vec<f64>,
    pub slots: Vec<SlotTrace>,
    pub rows: Vec<RowTrace>,
    /// The raw outputs `o` as the host stores them, [M, NV, W].
    pub mixed: Vec<f32>,
}

/// The bounds of every output a case compares, absolute per element.
#[derive(Clone)]
pub struct Bounds {
    /// The raw outputs `o` of the recurrence, before their rounding to the
    /// activation, [M, NV, W].
    pub raw: Vec<f64>,
    pub gated: Vec<f64>,
    /// Per bank, per value head and state row (a state element is within
    /// the state row's 2-norm error); `None` where no slot's successor is
    /// the bank.
    pub delta: Vec<Option<Vec<f64>>>,
    /// Per bank, the recorded tape rows in the tape row's layout.
    pub tape: Vec<Option<Vec<f64>>>,
}

impl Bounds {
    /// The bounds of two outputs compared with each other.
    pub fn plus(&self, other: &Bounds) -> Bounds {
        let mut sum = self.clone();
        let add = |a: &mut [f64], b: &[f64]| {
            assert_eq!(a.len(), b.len());
            a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
        };
        add(&mut sum.raw, &other.raw);
        add(&mut sum.gated, &other.gated);
        for (mine, theirs) in sum.delta.iter_mut().zip(&other.delta) {
            if let (Some(mine), Some(theirs)) = (mine, theirs) {
                add(mine, theirs);
            }
        }
        for (mine, theirs) in sum.tape.iter_mut().zip(&other.tape) {
            if let (Some(mine), Some(theirs)) = (mine, theirs) {
                add(mine, theirs);
            }
        }
        sum
    }
}

/// One row's inputs' errors, absolute per element, for a library.
struct InputErrors {
    /// q and k per element (key head, column) and their 2-norms and the
    /// vectors' 2-norms.
    dq: Vec<f64>,
    dk: Vec<f64>,
    q_norm: Vec<f64>,
    k_norm: Vec<f64>,
    dq_norm: Vec<f64>,
    dk_norm: Vec<f64>,
    /// v per element (value head, column).
    dv: Vec<f64>,
    /// β and its error per value head.
    dbeta: Vec<f64>,
    /// The decay factors' relative error per value head and decay input.
    rho: Vec<f64>,
}

impl Trace {
    fn inputs(&self, row: &RowTrace, f: &Functions) -> InputErrors {
        let s = &self.shape;
        let (nk, nv, w) = (s.key_heads, s.value_heads, s.width);
        let taps = gamma(s.convolution);
        let channels = s.channels();
        // The activated channels' errors: the convolution's F32 sum (γ_C),
        // carried by SiLU's slope (at most 1.1), and SiLU's own evaluation.
        let delta_y: Vec<f64> = (0..channels)
            .map(|c| {
                let x = f64::from(row.convolved[c]);
                1.1 * taps * f64::from(row.abs_convolution[c])
                    + f.logistic(x) * f64::from(row.activated[c]).abs()
            })
            .collect();
        let mut dq = vec![0.0; nk * w];
        let mut dk = vec![0.0; nk * w];
        let (mut q_norm, mut k_norm) = (vec![0.0; nk], vec![0.0; nk]);
        let (mut dq_norm, mut dk_norm) = (vec![0.0; nk], vec![0.0; nk]);
        for head in 0..nk {
            for (part, scaled) in [(0, true), (nk, false)] {
                let first = (part + head) * w;
                let y = &row.activated[first..][..w];
                let dy = &delta_y[first..][..w];
                let squares: f64 = y.iter().map(|v| f64::from(*v).powi(2)).sum();
                let y_norm = squares.sqrt();
                let dy_norm = dy.iter().map(|v| v * v).sum::<f64>().sqrt();
                // The F32 sum of squares (γ_W of its terms), the ε, the
                // reciprocal square root and the products.
                let sum = squares + s.epsilon;
                let drift = 2.0 * y_norm * dy_norm + dy_norm * dy_norm;
                let argument = (drift + gamma(w) * (squares + drift)) / sum + U;
                assert!(argument < 0.5, "the L2 norm's argument is unbounded");
                let mut factor = (1.0 - argument).powf(-0.5) * (1.0 + f.rsqrt) * (1.0 + U) - 1.0;
                let mut inverse = 1.0 / sum.sqrt();
                if scaled {
                    // q's W^-1/2: its own rsqrt and a product.
                    factor = (1.0 + factor) * (1.0 + f.rsqrt) * (1.0 + U) - 1.0;
                    inverse /= (w as f64).sqrt();
                }
                let (errors, norms, error_norms) = if scaled {
                    (&mut dq, &mut q_norm, &mut dq_norm)
                } else {
                    (&mut dk, &mut k_norm, &mut dk_norm)
                };
                for column in 0..w {
                    errors[head * w + column] = inverse * (1.0 + factor) * dy[column]
                        + factor * f64::from(row.prepared[first + column]).abs();
                }
                norms[head] = row.prepared[first..][..w]
                    .iter()
                    .map(|v| f64::from(*v).powi(2))
                    .sum::<f64>()
                    .sqrt();
                error_norms[head] = errors[head * w..][..w]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt();
            }
        }
        let dv = (0..nv * w).map(|i| delta_y[2 * nk * w + i]).collect();
        let dbeta = (0..nv)
            .map(|h| row.beta[h] * f.logistic(-row.beta_input[h]))
            .collect();
        let rho = (0..nv * s.decay_width)
            .map(|index| self.decay_error(row, f, index / s.decay_width, index))
            .collect();
        InputErrors {
            dq,
            dk,
            q_norm,
            k_norm,
            dq_norm,
            dk_norm,
            dv,
            dbeta,
            rho,
        }
    }

    /// The relative error of decay input `index` of `head`'s factor
    /// `exp(L)`. `L` comes from the rule's `x = a + time_bias` (one
    /// rounding), then for the softplus decay from
    /// `rate * (max(x, 0) + log(1 + exp(-|x|)))`: the softplus's slope in
    /// `x` is at most 1, then the `exp`, the sum with 1, the `log`, the sum
    /// and the product each round; for Kimi's decay from
    /// `lower / (1 + exp(rate x))`: the two products, the `exp`, the sum with
    /// 1 and the division.
    fn decay_error(&self, row: &RowTrace, f: &Functions, head: usize, index: usize) -> f64 {
        let x = row.shifted[index];
        let rate = self.rate[head];
        let log_decay = row.log_decay[index].abs();
        let log_error = match self.decay_lower {
            None => {
                let t = (-x.abs()).exp();
                let inner = t * f.exp(-x.abs()) + U * (1.0 + t);
                let softplus = log_decay / rate.abs();
                rate.abs() * (U * x.abs() + inner / (1.0 + t) + f.log(1.0 + t) + U * softplus)
                    + U * log_decay
            }
            Some(_) => {
                let z = rate * x;
                let exp = z.exp();
                let argument = 2.0 * U * z.abs();
                let relative =
                    (exp * (f.exp(z) + argument) + U * (1.0 + exp)) / (1.0 + exp) + f.divide;
                relative * log_decay
            }
        };
        (1.0 + f.exp(row.log_decay[index])) * log_error.exp() - 1.0
    }

    /// Every output's bound for the entry's form.
    pub fn bounds(&self, native: Native) -> Bounds {
        let f = Functions::of(native.backend());
        let s = &self.shape;
        let (nv, w, dw) = (s.value_heads, s.width, s.decay_width);
        let inputs: Vec<InputErrors> = self.rows.iter().map(|row| self.inputs(row, &f)).collect();
        let mut raw = vec![0.0f64; self.mixed.len()];
        let mut delta = vec![None; s.banks];
        let mut tape = vec![None; s.banks];
        for slot in &self.slots {
            let recorded = s.tape.min(slot.rows - slot.stop);
            let mut state = vec![0.0f64; nv * w];
            let mut recorded_tape = vec![0.0f64; recorded * s.tape_row()];
            for entry in 0..recorded {
                let at = slot.first + slot.stop + entry;
                let base = entry * s.tape_row();
                let row = &self.rows[at];
                for key_head in 0..s.key_heads {
                    for column in 0..w {
                        // The tape holds the key as F32.
                        recorded_tape[base + nv * w + key_head * w + column] = inputs[at].dk
                            [key_head * w + column]
                            + U * f64::from(row.prepared[(s.key_heads + key_head) * w + column])
                                .abs();
                    }
                }
                for head in 0..nv {
                    for input in 0..dw {
                        let decay = row.log_decay[head * dw + input].exp();
                        recorded_tape[base + (nv + s.key_heads) * w + head * dw + input] =
                            decay * (inputs[at].rho[head * dw + input] + U);
                    }
                }
            }
            for head in 0..nv {
                let chunked = native.chunked(slot);
                let pieces = native
                    .pieces()
                    .filter(|_| chunked > 0)
                    .map(|form| self.pieces(&f, form, slot, chunked, head));
                for i in 0..w {
                    let (published, residuals) = self.advance(
                        slot,
                        &inputs,
                        pieces.as_deref(),
                        head,
                        i,
                        recorded,
                        &mut raw,
                    );
                    state[head * w + i] = published;
                    for (entry, residual) in residuals.into_iter().enumerate() {
                        recorded_tape[entry * s.tape_row() + head * w + i] = residual;
                    }
                }
            }
            delta[slot.following] = Some(state);
            tape[slot.following] = Some(recorded_tape);
        }
        Bounds {
            gated: self.gate_bounds(native.backend(), &raw),
            raw,
            delta,
            tape,
        }
    }

    /// The step's error of one row, from the error `e` of the state row
    /// before it: `local` includes the row's own rounding (the step form's)
    /// or only what the exact map carries (a chunk form's rows, whose
    /// rounding is the piece's). Returns the state row's error after the
    /// row, the raw output's, and the residual's.
    fn row(
        &self,
        at: usize,
        head: usize,
        i: usize,
        inputs: &InputErrors,
        e: f64,
        local: bool,
    ) -> (f64, f64, f64) {
        let s = &self.shape;
        let w = s.width;
        let kh = s.key_head(head);
        let st = &self.rows[at].state[head * w + i];
        let row = &self.rows[at];
        let dw = s.decay_width;
        let (decayed, after, a) = (
            f64::from(st.decayed),
            f64::from(st.after),
            f64::from(st.residual),
        );
        let beta = row.beta[head];
        let dbeta = inputs.dbeta[head];
        let lambda = (0..dw)
            .map(|j| row.log_decay[head * dw + j].exp())
            .fold(0.0, f64::max);
        let rho = (0..dw)
            .map(|j| inputs.rho[head * dw + j])
            .fold(0.0, f64::max);
        let (dk, dq) = (inputs.dk_norm[kh], inputs.dq_norm[kh]);
        let (kbar, qbar) = (inputs.k_norm[kh] + dk, inputs.q_norm[kh] + dq);
        let dv = inputs.dv[head * w + i];
        let gw = gamma(w);
        // The decay: its factors' error and, where the row rounds, the
        // product's.
        let mu = rho + if local { U * (1.0 + rho) } else { 0.0 };
        let e1 = lambda * e * (1.0 + mu) + mu * decayed;
        // The inputs: the map's change under k, v and β, each at the exact
        // trajectory.
        let m_input = kbar * e1 + decayed * dk;
        let sensitivity = dbeta * (a + dv + decayed * dk) * kbar
            + beta * (dv + decayed * dk) * kbar
            + beta * a * dk;
        let sum_k = f64::from(st.abs_k) + m_input;
        let (m_round, local_r, local_s) = if local {
            let m_round = gw * sum_k;
            let abar = a + dv + m_input + m_round;
            let local_r = (beta + dbeta) * (m_round + 2.0 * U * (1.0 + U) * abar);
            (m_round, local_r, U * after)
        } else {
            (0.0, 0.0, 0.0)
        };
        let e2 =
            (e1 + sensitivity + local_r * kbar + local_s) / (1.0 - if local { U } else { 0.0 });
        let sum_q = f64::from(st.abs_q) + dq * after + qbar * e2;
        let output = qbar * e2 + dq * after + if local { gw * sum_q } else { 0.0 };
        let dm = m_input + m_round;
        let residual = dbeta * (a + dv + dm)
            + beta * (dv + dm)
            + if local {
                2.0 * U * (1.0 + U) * (beta + dbeta) * (a + dv + dm)
            } else {
                0.0
            };
        (e2, output, residual + U * beta * a)
    }

    /// One state row's trajectory through a slot: the error of its published
    /// state and of each recorded tape row's residual; the raw outputs'
    /// errors into `raw`.
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &self,
        slot: &SlotTrace,
        inputs: &[InputErrors],
        pieces: Option<&[PieceContext]>,
        head: usize,
        i: usize,
        recorded: usize,
        raw: &mut [f64],
    ) -> (f64, Vec<f64>) {
        let s = &self.shape;
        let (nv, w) = (s.value_heads, s.width);
        // The source version: the bank's state advanced by the tape's first
        // rows, with the step's update on recorded inputs.
        let mut e = 0.0;
        let mut norm = 0.0;
        for entry in 0..slot.taped {
            let [lambda, decayed, after] = slot.replay[(entry * nv + head) * w + i].map(f64::from);
            e = lambda * e + U * decayed;
            e += U * after;
            norm = after;
        }
        let mut residuals = Vec::with_capacity(recorded);
        let mut published = (slot.stop == 0).then_some((e, norm));
        let mut chunked = 0;
        for piece in pieces.unwrap_or(&[]) {
            let first = slot.first + chunked;
            let sigma = f64::from(self.rows[first].state[head * w + i].before);
            let (state_local, output_local) = piece.rounding(self, first, head, i, sigma);
            let mut propagated = e;
            for (t, local) in output_local.iter().enumerate() {
                let at = first + t;
                let (next, output, _) = self.row(at, head, i, &inputs[at], propagated, false);
                propagated = next;
                raw[(at * nv + head) * w + i] = output + local;
            }
            e = propagated + state_local;
            chunked += piece.rows;
        }
        if chunked > 0 {
            norm = f64::from(self.rows[slot.first + chunked - 1].state[head * w + i].after);
            if chunked == slot.stop {
                published = Some((e, norm));
            }
        }
        for t in chunked..slot.rows {
            let at = slot.first + t;
            let (next, output, residual) = self.row(at, head, i, &inputs[at], e, true);
            e = next;
            norm = f64::from(self.rows[at].state[head * w + i].after);
            raw[(at * nv + head) * w + i] = output;
            if t >= slot.stop && t - slot.stop < recorded {
                residuals.push(residual);
            }
            if t + 1 == slot.stop {
                published = Some((e, norm));
            }
        }
        // The published state is the F32 value of the exact one: the host's
        // `as f32` of it.
        let (published, norm) = published.expect("a slot's stop row lies within it");
        (published + U * norm, residuals)
    }

    /// The pieces of a slot's chunked rows for `head`.
    fn pieces(
        &self,
        f: &Functions,
        form: Pieces,
        slot: &SlotTrace,
        chunked: usize,
        head: usize,
    ) -> Vec<PieceContext> {
        let mut out = Vec::new();
        let mut first = 0;
        while first < chunked {
            let rows = form.size.min(chunked - first);
            out.push(PieceContext::new(
                self,
                f,
                form,
                slot.first + first,
                rows,
                head,
            ));
            first += rows;
        }
        out
    }
}

/// A chunked piece's matrices for one value head: what its rounding
/// depends on beyond the state row.
struct PieceContext {
    form: Pieces,
    rows: usize,
    width: usize,
    beta: Vec<f64>,
    /// `‖k_t ∘ e^{G_t}‖`, `‖q_t ∘ e^{G_t}‖` and the relative error of
    /// `e^{G_t}`.
    kappa: Vec<f64>,
    kappa_q: Vec<f64>,
    theta: Vec<f64>,
    /// The state update's `tail_s` scalar (the decay's factor outside the
    /// products), `‖K'_s‖` and the relative error of `e^{G_last - G_s}`.
    tail: Vec<f64>,
    update_norm: Vec<f64>,
    tail_theta: Vec<f64>,
    /// The state's decay `max e^{G_last}` and its relative error.
    last: f64,
    last_theta: f64,
    /// `|Tb|` and the error of `Tb`'s F32 value.
    tb: Vec<Vec<f64>>,
    d_tb: Vec<Vec<f64>>,
    /// `Σ_c |q k| e^{ΔG}` (|D|) and its F32 value's error.
    d: Vec<Vec<f64>>,
    d_error: Vec<Vec<f64>>,
}

impl PieceContext {
    fn new(
        trace: &Trace,
        f: &Functions,
        form: Pieces,
        first: usize,
        rows: usize,
        head: usize,
    ) -> Self {
        let s = &trace.shape;
        let (nk, w, dw) = (s.key_heads, s.width, s.decay_width);
        let kh = s.key_head(head);
        let n = rows;
        let p = form.size;
        let accumulate = gamma(w + 2);
        let sum_error = gamma(p);
        // Cumulative log decays per row and decay input (each term <= 0, so
        // the F32 running sum is within γ_n of its magnitude).
        let mut g = vec![vec![0.0f64; dw]; n];
        for t in 0..n {
            for j in 0..dw {
                let step = trace.rows[first + t].log_decay[head * dw + j];
                g[t][j] = if t == 0 { step } else { g[t - 1][j] + step };
            }
        }
        // The relative error of `exp(G_t - G_s)`: the exponent's sums and
        // difference, then the library.
        let theta_pair = |t: usize, s: usize, j: usize| {
            let (a, b) = (g[t][j], g[s][j]);
            let exponent = sum_error * (a.abs() + b.abs()) + U * (a - b).abs();
            (1.0 + f.exp(a - b)) * exponent.exp() - 1.0
        };
        let q = |t: usize, c: usize| f64::from(trace.rows[first + t].prepared[kh * w + c]);
        let k = |t: usize, c: usize| f64::from(trace.rows[first + t].prepared[(nk + kh) * w + c]);
        let beta: Vec<f64> = (0..n).map(|t| trace.rows[first + t].beta[head]).collect();
        let (mut a_exact, mut a_abs, mut a_error) = (
            vec![vec![0.0; n]; n],
            vec![vec![0.0; n]; n],
            vec![vec![0.0; n]; n],
        );
        let (mut d, mut d_error) = (vec![vec![0.0; n]; n], vec![vec![0.0; n]; n]);
        for t in 0..n {
            for s_row in 0..=t {
                let (mut kk, mut kk_abs, mut kk_error) = (0.0, 0.0, 0.0);
                let (mut qk_abs, mut qk_error) = (0.0, 0.0);
                for c in 0..w {
                    let j = c % dw;
                    let decay = (g[t][j] - g[s_row][j]).exp();
                    let theta = theta_pair(t, s_row, j);
                    let product = k(t, c) * k(s_row, c) * decay;
                    kk += product;
                    kk_abs += product.abs();
                    kk_error += product.abs() * (accumulate + theta);
                    let query = q(t, c) * k(s_row, c) * decay;
                    qk_abs += query.abs();
                    qk_error += query.abs() * (accumulate + theta);
                }
                d[t][s_row] = qk_abs;
                d_error[t][s_row] = qk_error;
                if s_row < t {
                    a_exact[t][s_row] = beta[t] * kk;
                    a_abs[t][s_row] = beta[t] * kk_abs;
                    a_error[t][s_row] = beta[t] * kk_error + U * a_abs[t][s_row];
                }
            }
        }
        // T = (I + A)^-1 (lower unit triangular), |T| and its F32 value's
        // error: A's own error carried through the inverse (the Neumann
        // series of |T| |ΔA| terminates, both strictly lower) and the
        // form's evaluation of the inverse.
        let mut inverse = vec![vec![0.0f64; n]; n];
        for column in 0..n {
            for row in column..n {
                let mut value = if row == column { 1.0 } else { 0.0 };
                for k_index in column..row {
                    value -= a_exact[row][k_index] * inverse[k_index][column];
                }
                inverse[row][column] = value;
            }
        }
        let magnitude: Vec<Vec<f64>> = inverse
            .iter()
            .map(|r| r.iter().map(|v| v.abs()).collect())
            .collect();
        let multiply = |x: &Vec<Vec<f64>>, y: &Vec<Vec<f64>>| {
            let mut out = vec![vec![0.0; n]; n];
            for i in 0..n {
                for k_index in 0..n {
                    if x[i][k_index] != 0.0 {
                        for j in 0..n {
                            out[i][j] += x[i][k_index] * y[k_index][j];
                        }
                    }
                }
            }
            out
        };
        let reach = multiply(&magnitude, &a_error);
        let mut power = reach.clone();
        let mut series = reach.clone();
        for _ in 2..n {
            power = multiply(&power, &reach);
            for i in 0..n {
                for j in 0..n {
                    series[i][j] += power[i][j];
                }
            }
        }
        let carried = multiply(&series, &magnitude);
        let identity_plus = |m: &Vec<Vec<f64>>| {
            let mut out = m.clone();
            for i in 0..n {
                out[i][i] += 1.0;
            }
            out
        };
        let evaluation = match form.inverse {
            Inverse::Polynomial => {
                // (I - A)(I + A²)(I + A⁴): three products of 8-term sums
                // and the additions.
                let a2 = multiply(&a_abs, &a_abs);
                let a4 = multiply(&a2, &a2);
                let chain = multiply(
                    &multiply(&identity_plus(&a_abs), &identity_plus(&a2)),
                    &identity_plus(&a4),
                );
                let scale = gamma(form.size + 2) / (1.0 - gamma(form.size + 2));
                chain
                    .iter()
                    .map(|r| r.iter().map(|v| v * scale).collect())
                    .collect::<Vec<Vec<f64>>>()
            }
            Inverse::Substitution => {
                // Forward substitution: (I + A + Δ) x = e with
                // |Δ| <= γ_n |I + A| (Higham, Theorem 8.5).
                let wide = multiply(&multiply(&magnitude, &identity_plus(&a_abs)), &{
                    let mut m = magnitude.clone();
                    for i in 0..n {
                        for j in 0..n {
                            m[i][j] += carried[i][j];
                        }
                    }
                    m
                });
                let scale = gamma(n) / (1.0 - gamma(n));
                wide.iter()
                    .map(|r| r.iter().map(|v| v * scale).collect())
                    .collect::<Vec<Vec<f64>>>()
            }
        };
        let mut tb = vec![vec![0.0; n]; n];
        let mut d_tb = vec![vec![0.0; n]; n];
        for t in 0..n {
            for s_row in 0..=t {
                tb[t][s_row] = magnitude[t][s_row] * beta[s_row];
                d_tb[t][s_row] =
                    (carried[t][s_row] + evaluation[t][s_row]) * beta[s_row] + U * tb[t][s_row];
            }
        }
        let theta_at = |t: usize, j: usize| {
            let exponent = sum_error * g[t][j].abs();
            (1.0 + f.exp(g[t][j])) * exponent.exp() - 1.0
        };
        let last_row = n - 1;
        let tail_theta_at = |t: usize, j: usize| {
            let (a, b) = (g[last_row][j], g[t][j]);
            let exponent = sum_error * (a.abs() + b.abs()) + U * (a - b).abs();
            (1.0 + f.exp(a - b)) * exponent.exp() - 1.0
        };
        let (mut kappa, mut kappa_q, mut theta) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let (mut tail, mut update_norm, mut tail_theta) =
            (vec![1.0; n], vec![0.0; n], vec![0.0; n]);
        for t in 0..n {
            let (mut kk, mut qq, mut uu) = (0.0, 0.0, 0.0);
            for c in 0..w {
                let j = c % dw;
                let gamma_t = g[t][j].exp();
                let tail_t = (g[last_row][j] - g[t][j]).exp();
                kk += (k(t, c) * gamma_t).powi(2);
                qq += (q(t, c) * gamma_t).powi(2);
                // One decay per head factors out of the update's keys.
                uu += (k(t, c) * if dw == 1 { 1.0 } else { tail_t }).powi(2);
                theta[t] = f64::max(theta[t], theta_at(t, j));
                tail_theta[t] = f64::max(tail_theta[t], tail_theta_at(t, j));
            }
            kappa[t] = kk.sqrt();
            kappa_q[t] = qq.sqrt();
            update_norm[t] = uu.sqrt();
            if dw == 1 {
                tail[t] = (g[last_row][0] - g[t][0]).exp();
            }
        }
        let last = (0..dw).map(|j| g[last_row][j].exp()).fold(0.0, f64::max);
        let last_theta = (0..dw).map(|j| theta_at(last_row, j)).fold(0.0, f64::max);
        PieceContext {
            form,
            rows,
            width: w,
            beta,
            kappa,
            kappa_q,
            theta,
            tail,
            update_norm,
            tail_theta,
            last,
            last_theta,
            tb,
            d_tb,
            d,
            d_error,
        }
    }

    /// The rounding of the piece's computation for state row `i` of norm
    /// `sigma` at the piece's start: the error it adds to the piece's final
    /// state row, and to each row's raw output.
    fn rounding(
        &self,
        trace: &Trace,
        first: usize,
        head: usize,
        i: usize,
        sigma: f64,
    ) -> (f64, Vec<f64>) {
        let s = &trace.shape;
        let (nk, w) = (s.key_heads, s.width);
        let n = self.rows;
        let ops = self.form.operands;
        let accumulate = gamma(w);
        let multiples = if ops == Operands::F16 { 2 } else { 1 };
        let u_abs: Vec<f64> = (0..n)
            .map(|t| {
                let st = &trace.rows[first + t];
                self.beta[t] * f64::from(st.state[head * w + i].residual)
            })
            .collect();
        let z_abs: Vec<f64> = (0..n)
            .map(|t| {
                let v = f64::from(trace.rows[first + t].prepared[2 * nk * w + head * w + i]);
                v.abs() + self.kappa[t] * sigma
            })
            .collect();
        // Z = V - γ X: X's operands (S and the decayed K, both high + low)
        // and F32 accumulation, V as one f16.
        let d_z: Vec<f64> = (0..n)
            .map(|t| {
                let v = f64::from(trace.rows[first + t].prepared[2 * nk * w + head * w + i]);
                let x = (accumulate + self.theta[t] + U) * self.kappa[t] * sigma
                    + ops.dot(Staged::Split, self.kappa[t], Staged::Split, sigma, w);
                x + ops.element(Staged::Single, v) + U * z_abs[t]
            })
            .collect();
        // U = Tb Z: Z as a high and a low f16, Tb as one, accumulated in F32.
        let d_u: Vec<f64> = (0..n)
            .map(|t| {
                let (mut error, mut magnitude) = (0.0, 0.0);
                for s_row in 0..=t {
                    let zz = z_abs[s_row] + d_z[s_row];
                    let (tb, dtb) = (self.tb[t][s_row], self.d_tb[t][s_row]);
                    let (ez, et) = (
                        ops.element(Staged::Split, zz),
                        ops.element(Staged::Single, tb + dtb),
                    );
                    error +=
                        dtb * zz + (tb + dtb) * d_z[s_row] + ez * (tb + dtb) + zz * et + ez * et;
                    magnitude += (tb + dtb) * zz;
                }
                error + gamma(multiples * n) * magnitude
            })
            .collect();
        // O = Y γ + D U: Y from one f16 pair, D and U each one f16.
        let out: Vec<f64> = (0..n)
            .map(|t| {
                let y = (accumulate + self.theta[t] + U) * self.kappa_q[t] * sigma
                    + ops.dot(Staged::Single, self.kappa_q[t], Staged::Single, sigma, w);
                let (mut error, mut magnitude) = (0.0, self.kappa_q[t] * sigma);
                for s_row in 0..=t {
                    let du = d_u[s_row] + ops.element(Staged::Single, u_abs[s_row] + d_u[s_row]);
                    let dd = self.d_error[t][s_row]
                        + ops.element(Staged::Single, self.d[t][s_row] + self.d_error[t][s_row]);
                    error += dd * (u_abs[s_row] + du) + self.d[t][s_row] * du;
                    magnitude += (self.d[t][s_row] + dd) * (u_abs[s_row] + du);
                }
                y + error
                    + gamma(n + 2) * magnitude
                    + U * f64::from(trace.rows[first + t].state[head * w + i].output)
            })
            .collect();
        // S' = S diag(f) + Σ_s U_s K'_s: U_s tail_s and K'_s each a high and
        // a low f16.
        let mut state = (self.last_theta + U) * self.last * sigma;
        let mut magnitude = self.last * sigma;
        for s_row in 0..n {
            let weight = u_abs[s_row] * self.tail[s_row];
            let d_weight = d_u[s_row] * self.tail[s_row] + (self.tail_theta[s_row] + U) * weight;
            let kp = self.update_norm[s_row];
            let ew = ops.element(Staged::Split, weight + d_weight);
            let ek = ops.vector(Staged::Split, kp, w);
            let low = if ops == Operands::F16 {
                (U16 * (weight + d_weight) + ETA16) * (U16 * kp + ETA16 * (w as f64).sqrt())
            } else {
                0.0
            };
            state += d_weight * kp + ew * kp + (weight + d_weight) * ek + ew * ek + low;
            magnitude += weight * kp;
        }
        let after = f64::from(trace.rows[first + n - 1].state[head * w + i].after);
        state += gamma(3 * n + 1) * magnitude + U * after;
        (state, out)
    }
}

impl Trace {
    /// The gated outputs' bounds from the raw outputs' (`raw`, before their
    /// rounding to the activation): the stored raw rounded to the
    /// activation, the RMS normalization, the output gate and the product,
    /// each rounded to it again, as the contract's gate and the host's
    /// `Case::gate` do.
    fn gate_bounds(&self, backend: BackendName, raw: &[f64]) -> Vec<f64> {
        let s = &self.shape;
        let (nv, w) = (s.value_heads, s.width);
        let a = s.activation;
        let f = Functions::of(backend);
        let rms = Library::of(backend).rms(w);
        let mut bounds = vec![0.0; self.mixed.len()];
        for (at, row) in self.rows.iter().enumerate() {
            for head in 0..nv {
                let first = (at * nv + head) * w;
                let y: Vec<f64> = self.mixed[first..][..w]
                    .iter()
                    .map(|o| a.round(f64::from(*o)))
                    .collect();
                let dy: Vec<f64> = (0..w)
                    .map(|j| a.rounded(raw[first + j], y[j] as f32))
                    .collect();
                let squares: f64 = y.iter().map(|v| v * v).sum();
                let drift: f64 = (0..w)
                    .map(|j| 2.0 * y[j].abs() * dy[j] + dy[j] * dy[j])
                    .sum();
                let argument = drift / (squares + w as f64 * s.rms_epsilon);
                assert!(argument < 0.5, "the RMS argument is unbounded");
                let inverse = 1.0 / (squares / w as f64 + s.rms_epsilon).sqrt();
                let spread = (1.0 - argument).powf(-0.5) - 1.0;
                for j in 0..w {
                    let weight = self.norm[j].abs();
                    let normalized = a.round(y[j] * inverse * self.norm[j]);
                    let before = inverse * (1.0 + spread) * weight * dy[j]
                        + y[j].abs() * inverse * weight * spread
                        + rms * (y[j].abs() + dy[j]) * inverse * (1.0 + spread) * weight;
                    let d_normalized = a.rounded(before, normalized as f32);
                    let z = f64::from(row.gate[head * w + j]);
                    let numerator = if s.sigmoid_gate { 1.0 } else { z };
                    let activated_exact = numerator / (1.0 + (-z).exp());
                    let activated = a.round(activated_exact);
                    let d_activated =
                        a.rounded(f.logistic(z) * activated_exact.abs(), activated as f32);
                    let product = d_normalized * activated.abs()
                        + normalized.abs() * d_activated
                        + d_normalized * d_activated
                        + U * (normalized * activated).abs();
                    bounds[first + j] = a.rounded(product, a.round(normalized * activated) as f32);
                }
            }
        }
        bounds
    }
}
