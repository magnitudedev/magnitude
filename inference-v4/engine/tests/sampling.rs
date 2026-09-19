//! Semantic selection qualification; native automatic-selection qualification is separate.
#[path = "support/reference.rs"]
mod reference;
use reference::{Arg, TensorData, WIDTHS};
use seismic_lang::types::DType;
use std::collections::HashMap;

/// Selection is integral, so both partitionings must agree exactly.
fn sample(rows: &[Vec<f64>], masks: &[u32], draws: &[[u32; 6]]) -> Vec<[i32; 2]> {
    let [narrow, wide] = WIDTHS.map(|width| sample_at(width, rows, masks, draws));
    assert_eq!(narrow, wide);
    wide
}
fn sample_at(width: i64, rows: &[Vec<f64>], masks: &[u32], draws: &[[u32; 6]]) -> Vec<[i32; 2]> {
    let program = seismic_std::program().unwrap();
    let mut vm = reference::interpreter(&program, width);
    let m = rows.len();
    let v = rows[0].len();
    let logits = vm.add_tensor(TensorData::dense(DType::F32, vec![m, v], rows.concat()));
    let mask = vm.add_tensor(TensorData::dense(
        DType::U32,
        vec![m, v.div_ceil(32)],
        masks.iter().map(|&x| x as f64).collect(),
    ));
    let draws = vm.add_tensor(TensorData::dense(
        DType::U32,
        vec![m, 6],
        draws.iter().flatten().map(|&x| x as f64).collect(),
    ));
    let out = vm.add_tensor(TensorData::dense(DType::I32, vec![m, 2], vec![0.; m * 2]));
    reference::run(
        &mut vm,
        "sample_rows",
        &[
            Arg::Tensor(logits),
            Arg::Tensor(mask),
            Arg::Tensor(draws),
            Arg::Tensor(out),
        ],
        &HashMap::from([("M".into(), m as i64), ("V".into(), v as i64)]),
    );
    (0..m)
        .map(|i| {
            [
                vm.tensors[out].get(i * 2) as i32,
                vm.tensors[out].get(i * 2 + 1) as i32,
            ]
        })
        .collect()
}
#[test]
fn masks_ties_and_invalid_distributions_are_row_local() {
    let rows = vec![
        vec![3., 7., 7.],
        vec![f64::NAN, 1., 2.],
        vec![0., f64::INFINITY, 1.],
        vec![f64::NEG_INFINITY; 3],
        vec![1., 2., 3.],
        vec![f64::NAN, 1., 2.],
    ];
    assert_eq!(
        sample(&rows, &[7, 6, 7, 7, 0, 7], &[[0; 6]; 6]),
        vec![[1, 0], [-1, 2], [-1, 2], [-1, 1], [-1, 1], [-1, 2]]
    );
    let rows = vec![vec![f64::NEG_INFINITY, 0., f64::NEG_INFINITY]; 2];
    assert_eq!(
        sample(&rows, &[7; 2], &[[1, 42, 0, 0, 0, 0], [1, 99, 0, 1, 0, 0]]),
        vec![[1, 0]; 2]
    );
}
#[test]
fn masks_cross_word_boundaries_without_padding_candidates() {
    let mut row = vec![0.; 35];
    row[31] = 10.;
    row[32] = 20.;
    row[34] = 30.;
    assert_eq!(
        sample(&[row], &[1 << 31, u32::MAX ^ 4], &[[0; 6]]),
        vec![[32, 0]]
    );
}
#[test]
fn categorical_matches_v3_counters_and_survives_regrouping() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../validation/results/fixtures/sampling-v3-reference.json"
    ))
    .unwrap();
    let rows: Vec<Vec<f64>> = serde_json::from_value(fixture["logits"].clone()).unwrap();
    let draws: Vec<[u32; 6]> = serde_json::from_value(fixture["draws"].clone()).unwrap();
    let expected: Vec<i32> = serde_json::from_value(fixture["expected"].clone()).unwrap();
    let expected: Vec<_> = expected.into_iter().map(|token| [token, 0]).collect();
    assert_eq!(
        sample(&rows, &vec![u32::MAX; rows.len() * 2], &draws),
        expected
    );
    let order = [5, 2, 4, 0, 3, 1];
    let rows: Vec<_> = order.iter().map(|&i| rows[i].clone()).collect();
    let draws: Vec<_> = order.iter().map(|&i| draws[i]).collect();
    let expected: Vec<_> = order.iter().map(|&i| expected[i]).collect();
    assert_eq!(
        sample(&rows, &vec![u32::MAX; rows.len() * 2], &draws),
        expected
    );
}
