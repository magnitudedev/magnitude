// What the general indexer's tests (`indexer_keys.rs`, `indexer_score.rs`,
// `indexer_topk.rs`) share: DeepSeek V4.1's indexer geometry, an F32 host
// model of a prepared head row (RMS norm, rotary, trained rounding), the
// portable bodies' interpreter and key planes in slabs.

#![allow(dead_code)]

use crate::rounding_model::{self, Rounding};
use crate::versions_common::*;
use seismic::{Device, Element, Tensor};
use seismic_lang::{
    checked::SourceFile,
    entry::ElementBindings,
    failure::SourceTermination,
    interp::{Arg, Interpreter, OutcomeValue, TensorData},
    reference_math::ReferenceScalar,
    registry,
    types::DType,
};

/// V4.1's indexer: 32 heads of 128 columns, the trailing 64 rotated in
/// adjacent pairs (RA = RT = 1).
pub const H: usize = 32;
pub const P: usize = 32;
pub const S: usize = 64;
pub const W: usize = 2 * P + S;
pub const EPSILON: f32 = 1e-6;
/// Key-plane rows a slab holds: lists break at its edges.
pub const SLAB_ROWS: usize = 4096;

/// A rotary layout: adjacent pairs (`RA`) over the trailing 2P columns
/// (`RT`), or half-split pairs over the leading ones.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub ra: usize,
    pub rt: usize,
}

pub const DEEPSEEK: Layout = Layout { ra: 1, rt: 1 };
pub const HALF_SPLIT: Layout = Layout { ra: 0, rt: 0 };

/// The compressed rotary's frequencies (theta 160000 over 64 dims).
pub fn frequencies() -> Vec<f32> {
    (0..P).map(|pair| 160_000f32.powf(-(2.0 * pair as f32) / (2.0 * P as f32))).collect()
}

/// A text row's coordinates at `position`.
pub fn coordinates(position: usize) -> [i32; 4] {
    [position as i32; 4]
}

/// One head row in F32: RMS-normed with `norm` (when given), its pairs
/// rotated at `coordinates` (axis 0), then rounded; a rounded row from its
/// norm and rotation each held in the BF16 activation element (DeepSeek
/// V4.1's `k_norm` output and in-place rotations).
pub fn prepare(x: &[f32], norm: Option<&[f32]>, position: usize, layout: Layout, rounding: Rounding) -> Vec<f32> {
    let hold = |y: &mut [f32]| {
        if rounding.block != 0 {
            y.iter_mut().for_each(|v| *v = bf16_round(*v));
        }
    };
    let mut y = x.to_vec();
    if let Some(norm) = norm {
        let squares = y.iter().map(|v| v * v).sum::<f32>();
        let inverse = 1.0 / (squares / W as f32 + EPSILON).sqrt();
        for (v, n) in y.iter_mut().zip(norm) {
            *v = *v * inverse * n;
        }
        hold(&mut y);
    }
    let first = if layout.rt == 1 { S } else { 0 };
    for (pair, frequency) in frequencies().iter().enumerate() {
        let (lo, hi) = if layout.ra == 1 { (first + 2 * pair, first + 2 * pair + 1) } else { (first + pair, first + pair + P) };
        let (sin, cos) = (position as f32 * frequency).sin_cos();
        let (low, high) = (y[lo], y[hi]);
        y[lo] = low * cos - high * sin;
        y[hi] = high * cos + low * sin;
    }
    hold(&mut y);
    rounding_model::round(rounding, &y)
}

/// The largest scale of `rounding`'s runs in `row` times its element's
/// coarsest step over a run (2 for E2M1 at 4..6, 32 for E4M3 at 256..448):
/// the most one value's code can move when its unrounded value moves across
/// a decision boundary.
pub fn code_step(rounding: Rounding, row: &[f32]) -> f32 {
    if rounding.block == 0 {
        return 0.0;
    }
    let (max, step) = if rounding.element == 0 { (448.0, 32.0) } else { (6.0, 2.0) };
    row.chunks(rounding.block as usize)
        .map(|run| run.iter().fold(rounding.amax_floor, |a, v| a.max(v.abs())) / max * 2.0 * step)
        .fold(0.0, f32::max)
}

/// Whether `actual` is a rounding of a row near `expected`'s unrounded
/// values: every value equal, or within one code step of its run, with at
/// most `flips` values differing.
/// Unrounded rows agree within F32 arithmetic order: 1e-5 of the row's
/// largest magnitude.
pub fn near_codes(actual: &[f32], expected: &[f32], rounding: Rounding, flips: usize) -> bool {
    if rounding.block == 0 {
        let top = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        return actual.iter().zip(expected).all(|(x, y)| (x - y).abs() <= 1e-5 * top);
    }
    let block = rounding.block as usize;
    let mut differing = 0;
    for (a, e) in actual.chunks(block).zip(expected.chunks(block)) {
        let step = code_step(rounding, e);
        for (x, y) in a.iter().zip(e) {
            if x.to_bits() != y.to_bits() {
                differing += 1;
                if (x - y).abs() > step {
                    return false;
                }
            }
        }
    }
    differing <= flips
}

/// The checked kernel module with the indexer's sources.
fn module() -> seismic_lang::checked::CheckedModule {
    let mut sources = seismic_std::sources();
    for (path, text) in [
        ("indexer.seismic", include_str!("../../kernels/indexer.seismic")),
        ("pooled.seismic", include_str!("../../kernels/pooled.seismic")),
        ("attention.seismic", include_str!("../../kernels/attention.seismic")),
        ("functions.seismic", include_str!("../../kernels/functions.seismic")),
        ("rounding.seismic", include_str!("../../kernels/rounding.seismic")),
    ] {
        sources.push(SourceFile { path: path.into(), text: text.into() });
    }
    crate::common::check(sources)
}

/// An interpreter argument.
pub enum Input {
    F32(Vec<usize>, Vec<f32>),
    Bf16(Vec<usize>, Vec<f32>),
    I32(Vec<usize>, Vec<i32>),
    ScalarF32(f32),
    ScalarI32(i32),
    ScalarU32(u32),
    U32(Vec<usize>, Vec<u32>),
}

/// Entry `entry`'s portable body over `inputs` with element `A` BF16 when
/// `activation`: its results, and its inputs after the call.
pub fn interpret(entry: &str, activation: bool, inputs: Vec<Input>) -> (Vec<Vec<f64>>, Vec<(usize, Vec<f64>)>) {
    let bindings = match activation {
        true => ElementBindings::new().bind("A", registry::dense(DType::BF16)),
        false => ElementBindings::new(),
    };
    interpret_with(entry, &bindings, inputs)
}

/// [`interpret`] with the element bindings given.
pub fn interpret_with(
    entry: &str,
    bindings: &ElementBindings,
    inputs: Vec<Input>,
) -> (Vec<Vec<f64>>, Vec<(usize, Vec<f64>)>) {
    let module = module();
    let logical = module.entry(module.entry_named(entry).unwrap(), bindings).unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let args = inputs
        .into_iter()
        .map(|input| match input {
            Input::F32(shape, v) => Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::F32, shape, v.iter().map(|x| f64::from(*x)).collect()))),
            Input::Bf16(shape, v) => Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::BF16, shape, v.iter().map(|x| f64::from(*x)).collect()))),
            Input::I32(shape, v) => Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::I32, shape, v.iter().map(|x| f64::from(*x)).collect()))),
            Input::U32(shape, v) => Arg::Tensor(interpreter.add_tensor(TensorData::dense(DType::U32, shape, v.iter().map(|x| f64::from(*x)).collect()))),
            Input::ScalarF32(x) => Arg::Scalar(ReferenceScalar::F32(x.to_bits())),
            Input::ScalarI32(x) => Arg::Scalar(ReferenceScalar::I32(x)),
            Input::ScalarU32(x) => Arg::Scalar(ReferenceScalar::U32(x)),
        })
        .collect::<Vec<_>>();
    let outcome = interpreter.run_bounded(&args, u64::MAX).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("portable {entry} failed: {failure}");
    }
    macro_rules! values {
        ($reader:expr) => {{
            let reader = $reader;
            (0..reader.element_count()).map(|i| reader.read(i).unwrap()).collect::<Vec<f64>>()
        }};
    }
    let results = outcome
        .results()
        .map(|result| match result.value() {
            OutcomeValue::Tensor(reader) => values!(reader),
            _ => panic!("a tensor result"),
        })
        .collect();
    let inputs = outcome.inputs().map(|input| (input.ordinal(), values!(input.tensor()))).collect();
    (results, inputs)
}

/// Parameter `ordinal`'s values after a call (`interpret`'s second result).
pub fn after(inputs: &[(usize, Vec<f64>)], ordinal: usize) -> &[f64] {
    &inputs.iter().find(|(at, _)| *at == ordinal).expect("the parameter's values after the call").1
}

/// A key plane of `rows` rows of W columns in `element`, in slabs of
/// SLAB_ROWS rows: the slabs (kept alive by the caller) and the logical
/// plane.
pub fn plane(device: &Device, element: Element, rows: usize, values: &[f32]) -> (seismic::SlabTensor, Tensor) {
    let (slabs, logical) =
        slab_arenas(device, rows, SLAB_ROWS as u64, &[Arena { element, bank_shape: vec![W as u64], values }], false);
    (slabs, logical.into_iter().next().unwrap())
}

/// Spans `[rows, R, 2]` of each row's list, padded with empty spans.
pub fn span_table(lists: &[Vec<[i32; 2]>]) -> (Vec<i32>, usize) {
    let r = lists.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut table = Vec::new();
    for list in lists {
        for span in 0..r {
            table.extend(list.get(span).copied().unwrap_or([0, 0]));
        }
    }
    (table, r)
}

/// The rows of a list of spans, in order.
pub fn list_rows(spans: &[[i32; 2]]) -> Vec<i32> {
    spans.iter().flat_map(|[lo, hi]| *lo..*hi).collect()
}

/// `count` key-plane rows as spans, in runs that cross slab edges and skip
/// rows (S4's narrowed lists, a pooled source's reused rows).
pub fn scattered_spans(count: usize, seed: u64) -> Vec<[i32; 2]> {
    let mut random = Random(seed);
    let mut spans = Vec::new();
    let mut at = (SLAB_ROWS - 37) as i32;
    let mut left = count as i32;
    while left > 0 {
        let length = ((random.next().abs() * 3000.0) as i32 + 1).min(left);
        // A run never crosses a slab edge.
        let length = length.min(SLAB_ROWS as i32 - at % SLAB_ROWS as i32);
        spans.push([at, at + length]);
        left -= length;
        at += length + (random.next().abs() * 5.0) as i32;
    }
    spans
}

/// A rounding as an entry's four scalar inputs.
pub fn rounding_inputs(rounding: Rounding) -> [Input; 4] {
    [
        Input::ScalarI32(rounding.block),
        Input::ScalarI32(rounding.element),
        Input::ScalarI32(rounding.scale),
        Input::ScalarF32(rounding.amax_floor),
    ]
}
