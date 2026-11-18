use super::super::core::{activation, reduce};
use seismic::cpu::quant;
use seismic::cpu::slab::Records;
use seismic::cpu::{Dense, Weights};
use std::ops::Range;

/// An operand a staged row's products read, by its NVFP4 operand recipe and
/// tiled int8 kernel: weights, or a record port's records of them, which
/// share one.
pub trait Recipe {
    fn nvfp4_operands(&self) -> bool;
    fn tiled(&self) -> bool;
}

impl Recipe for Weights<'_> {
    #[inline(always)]
    fn nvfp4_operands(&self) -> bool {
        Weights::nvfp4_operands(self)
    }
    #[inline(always)]
    fn tiled(&self) -> bool {
        Weights::tiled(self)
    }
}

impl Recipe for Records<Weights<'_>> {
    #[inline(always)]
    fn nvfp4_operands(&self) -> bool {
        Records::nvfp4_operands(self)
    }
    #[inline(always)]
    fn tiled(&self) -> bool {
        false
    }
}

/// Columns `columns` of the RMS normalization of residual row `x` with the
/// norm operand `norm`, decoded into the work item's private bytes `shared`
/// (only those columns: [`norm_columns`]), each value published to `A` as
/// `rms_row` publishes it, into `out`.
#[inline(always)]
pub fn normalize_columns<A: Dense>(
    x: &[f32],
    norm: &Weights<'_>,
    epsilon: f32,
    shared: &mut [u8],
    columns: Range<usize>,
    out: &mut [f32],
) {
    rms_columns::<A>(x, norm_columns(norm, columns.clone(), shared), epsilon, columns, out)
}

/// Columns `columns` of the one-row norm operand `norm`, decoded into
/// `shared`: a part of a row decodes only its own columns' storage groups.
#[inline(always)]
pub fn norm_columns<'s>(norm: &Weights<'_>, columns: Range<usize>, shared: &'s mut [u8]) -> &'s [f32] {
    let (first, view) = norm.columns(columns.clone());
    let weight = seismic::cpu::tensor::floats(shared, view.k());
    view.decode_row(0, weight);
    &weight[columns.start - first..columns.end - first]
}

/// A projection entry's input row (`form_input` in functions.seismic): the
/// RMS normalization of residual row `x` with `norm` when the entry has one,
/// else the sum in part order of the parts of `x`, rows of `out.len()`
/// values a hyper-connection mix site publishes; each value published to `A`.
#[inline(always)]
pub fn form_input<A: Dense>(
    x: &[f32],
    norm: Option<&Weights<'_>>,
    epsilon: f32,
    shared: &mut [u8],
    out: &mut [f32],
) {
    let width = out.len();
    form_input_columns::<A>(x, norm, epsilon, shared, width, 0..width, out)
}

/// Columns `columns` of [`form_input`] of a `width`-value input row, into
/// `out`. Every part of a row reduces its RMS over the whole row in one
/// order, so the parts of a row share its factor exactly, and each value
/// depends on its own column alone.
#[inline(always)]
pub fn form_input_columns<A: Dense>(
    x: &[f32],
    norm: Option<&Weights<'_>>,
    epsilon: f32,
    shared: &mut [u8],
    width: usize,
    columns: Range<usize>,
    out: &mut [f32],
) {
    match norm {
        Some(norm) => normalize_columns::<A>(x, norm, epsilon, shared, columns, out),
        None => sum_columns::<A>(x, width, columns, out),
    }
}

/// Columns `columns` of the sum in part order of the `width`-wide parts of
/// `x`, into `out`, each value published to `A`.
#[inline(always)]
pub fn sum_columns<A: Dense>(x: &[f32], width: usize, columns: Range<usize>, out: &mut [f32]) {
    for (column, target) in columns.zip(out) {
        let mut sum = 0.0f32;
        for part in x.chunks_exact(width) {
            sum += part[column];
        }
        *target = activation::publish::<A>(sum);
    }
}

/// The RMS normalization of residual row `x` with the decoded norm `weight`,
/// each value published to `A`: `A(x * rsqrt(sum(x^2) / len + epsilon) * w)`.
#[inline(always)]
pub fn rms_row<A: Dense>(x: &[f32], weight: &[f32], epsilon: f32, out: &mut [f32]) {
    rms_columns::<A>(x, &weight[..x.len()], epsilon, 0..x.len(), out)
}

/// Columns `columns` of [`rms_row`], `weight` those columns' decoded norm,
/// into `out`: the factor is the whole row's.
#[inline(always)]
pub fn rms_columns<A: Dense>(x: &[f32], weight: &[f32], epsilon: f32, columns: Range<usize>, out: &mut [f32]) {
    let inverse = reduce::rms_inverse(x, epsilon);
    assert_eq!(weight.len(), columns.len(), "the norm of columns {columns:?}");
    for ((value, w), target) in x[columns].iter().zip(weight).zip(out) {
        *target = activation::publish::<A>(value * inverse * w);
    }
}

/// An activation row of `A` elements staged as F32.
#[inline(always)]
pub fn stage<A: Dense>(row: &[A::Storage], out: &mut [f32]) {
    activation::widen::<A>(row, out)
}

/// A staged row as the products over `weights`, which share it, take it: in
/// place, as NVFP4 operands where the weights take them
/// (`quant::nvfp4::row`), else as it is.
#[inline(always)]
pub fn operand_row<W: Recipe + ?Sized>(weights: &[&W], row: &mut [f32]) {
    let nvfp4 = weights[0].nvfp4_operands();
    assert!(
        weights.iter().all(|weights| weights.nvfp4_operands() == nvfp4),
        "weights that share rows share their operand recipe"
    );
    if nvfp4 {
        quant::nvfp4::row(row)
    }
}
