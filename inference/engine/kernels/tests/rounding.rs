// `block_round` (rounding.seismic) bit for bit against an independent host
// model of DeepSeek's three trained roundings (the window's E4M3/UE8M0, the
// indexer's E2M1/UE8M0, the compressed rows' E2M1/E4M3), over runs that
// reach every edge: all zeros (the floor's scale), signed zeros, exact ties
// on both element grids, saturation (an E4M3 scale below a / 6), subnormal
// element codes and subnormal E4M3 scales, and random rows.

mod common;
mod rounding_model;
mod versions_common;

use rounding_model::{COMPRESSED, INDEX, NONE, Rounding, WINDOW};
use versions_common::*;

fn portable(rounding: Rounding, row: &[f32]) -> Vec<f32> {
    use seismic_lang::{
        checked::SourceFile,
        entry::ElementBindings,
        failure::SourceTermination,
        interp::{Arg, Interpreter, OutcomeValue, TensorData},
        reference_math::ReferenceScalar,
        types::DType,
    };
    let mut sources = seismic_std::sources();
    sources.push(SourceFile { path: "rounding.seismic".into(), text: include_str!("../kernels/rounding.seismic").into() });
    let module = crate::common::check(sources);
    let logical = module.entry(module.entry_named("block_round").unwrap(), &ElementBindings::new()).unwrap();
    let mut interpreter = Interpreter::new(&logical);
    let x = interpreter.add_tensor(TensorData::dense(
        DType::F32,
        vec![row.len()],
        row.iter().map(|x| f64::from(*x)).collect(),
    ));
    let args = vec![
        Arg::Tensor(x),
        Arg::Scalar(ReferenceScalar::I32(rounding.block)),
        Arg::Scalar(ReferenceScalar::I32(rounding.element)),
        Arg::Scalar(ReferenceScalar::I32(rounding.scale)),
        Arg::Scalar(ReferenceScalar::F32(rounding.amax_floor.to_bits())),
    ];
    let outcome = interpreter.run_bounded(&args, u64::MAX).unwrap();
    if let SourceTermination::Failed(failure) = outcome.termination() {
        panic!("portable block_round failed: {failure}");
    }
    let result = outcome.results().next().unwrap();
    let OutcomeValue::Tensor(reader) = result.value() else { panic!("a tensor result") };
    (0..reader.element_count()).map(|i| reader.read(i).unwrap() as f32).collect()
}

/// Runs of `block` values reaching the edges of `rounding`, then random
/// rows at several magnitudes.
fn rows(rounding: Rounding) -> Vec<f32> {
    let block = rounding.block.max(16) as usize;
    let mut runs: Vec<Vec<f32>> = Vec::new();
    let run = |values: &[f32]| (0..block).map(|i| values[i % values.len()]).collect::<Vec<_>>();
    // All zeros, and signed zeros beside a value.
    runs.push(vec![0.0; block]);
    runs.push(run(&[-0.0, 0.0, 1.0, -0.0]));
    // Ties: with amax 6 (E2M1) the UE8M0 scale is 1, with amax 448 (E4M3)
    // as well, so the quotients are the values themselves.
    runs.push(run(&[6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -2.5, -5.0, 0.125]));
    runs.push(run(&[448.0, 17.0, 19.0, 21.0, 0.0029296875, -0.0048828125, 2.0f32.powi(-10), 3.0 * 2.0f32.powi(-10)]));
    // An E4M3 scale below a / 6 (a / 6 = 0.9375 + 2^-6 rounds down to
    // 0.9375), so the largest quotient saturates.
    runs.push(run(&[6.0 * (0.9375 + 2.0f32.powi(-6)), -6.0 * (0.9375 + 2.0f32.powi(-6)), 1.0, 0.1]));
    // Tiny runs: subnormal E4M3 scales (a / 6 below 2^-6) and the floors.
    runs.push(run(&[6.0 * 2.0f32.powi(-8), 1e-5, -3e-6]));
    runs.push(run(&[5e-5, -2e-5, 1e-6]));
    runs.push(run(&[1e-30, -2e-31]));
    // A wide magnitude spread: subnormal element codes beside the largest.
    runs.push(run(&[300.0, 0.01, -0.002, 1e-4, 7.0]));
    let mut random = Random(0x5eed);
    for magnitude in [1e-3f32, 1.0, 37.0, 3e3] {
        for _ in 0..4 {
            runs.push((0..block).map(|_| random.next() * magnitude).collect());
        }
    }
    runs.concat()
}

fn check(rounding: Rounding, label: &str) {
    let row = rows(rounding);
    let expected = rounding_model::round(rounding, &row);
    let actual = portable(rounding, &row);
    assert!(
        same_bits(&actual, &expected),
        "{label}: portable block_round differs from the host model at {:?}",
        first_difference(&actual, &expected)
    );
    if rounding.block > 0 {
        assert!(expected.iter().all(|x| bf16_round(*x).to_bits() == x.to_bits()), "{label}: rounded values are exact in BF16");
    }
}

#[test]
fn block_round_equals_the_reference_roundings() {
    check(WINDOW, "window");
    check(INDEX, "index");
    check(COMPRESSED, "compressed");
    check(NONE, "none");
}
