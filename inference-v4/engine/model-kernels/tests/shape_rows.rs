use magnitude_model_kernels::{sample_rows, shape_rows};

const ROWS: usize = 3;
const VOCABULARY: usize = 6;
const HISTORY: usize = 64;

fn fixture() -> (Vec<f32>, Vec<f32>, Vec<i32>) {
    let logits = vec![
        3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -2.0, 4.0, 3.0, 2.0, 1.0, 0.0, 4.0, 3.0, 2.0, 1.0, 0.0,
        -1.0,
    ];
    let params = vec![
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.2, 0.9, 2.0, 0.5, 0.25, 0.0, 2.0, 3.0,
        0.7, 0.2, 1.0, 0.0, 0.0, 0.0,
    ];
    let mut history = vec![-1; ROWS * HISTORY];
    history[HISTORY] = 0;
    history[HISTORY + 1] = 0;
    history[HISTORY + 2] = 1;
    (logits, params, history)
}

fn host_reference(logits: &[f32], params: &[f32], history: &[i32]) -> Vec<f32> {
    let mut result = vec![0.0; logits.len()];
    for row in 0..ROWS {
        let p = &params[row * 8..row * 8 + 8];
        let mut values = logits[row * VOCABULARY..row * VOCABULARY + VOCABULARY].to_vec();
        for (token, value) in values.iter_mut().enumerate() {
            let count = history[row * HISTORY..row * HISTORY + HISTORY]
                .iter()
                .filter(|entry| **entry == token as i32)
                .count() as f32;
            if count > 0.0 {
                *value = if *value < 0.0 {
                    *value * p[4]
                } else {
                    *value / p[4]
                };
                *value -= p[5] + p[6] * count;
            }
        }
        if p[0] != 0.0 {
            for value in &mut values {
                *value /= p[0];
            }
            let top_k = p[1] as usize;
            if top_k > 0 {
                let original = values.clone();
                for (token, value) in values.iter_mut().enumerate() {
                    let greater = original
                        .iter()
                        .filter(|other| **other > original[token])
                        .count();
                    if greater >= top_k {
                        *value = f32::NEG_INFINITY;
                    }
                }
            }
            if p[3] > 0.0 {
                let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for value in &mut values {
                    if (*value - maximum).exp() < p[3] {
                        *value = f32::NEG_INFINITY;
                    }
                }
            }
            if p[2] < 1.0 {
                let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = values
                    .iter()
                    .map(|value| (*value - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                let original = values.clone();
                for token in 0..VOCABULARY {
                    let preceding = (0..VOCABULARY)
                        .filter(|other| {
                            original[*other] > original[token]
                                || (original[*other] == original[token] && *other < token)
                        })
                        .map(|other| weights[other] / denominator)
                        .sum::<f32>();
                    if preceding >= p[2] {
                        values[token] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        result[row * VOCABULARY..row * VOCABULARY + VOCABULARY].copy_from_slice(&values);
    }
    result
}

fn assert_values(actual: &[f32], expected: &[f32]) {
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_infinite() {
            assert_eq!(actual, expected, "value {index}");
        } else {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "value {index}: {actual} != {expected}"
            );
        }
    }
}

fn philox_score(value: f32, token: u32, draw: [u32; 6]) -> f32 {
    let mut counter = [token, draw[3], draw[4], draw[5]];
    let mut key = [draw[1], draw[2]];
    for _ in 0..10 {
        let p0 = 3_528_531_795_u64 * u64::from(counter[0]);
        let p1 = 3_449_720_151_u64 * u64::from(counter[2]);
        counter = [
            (p1 >> 32) as u32 ^ counter[1] ^ key[0],
            p1 as u32,
            (p0 >> 32) as u32 ^ counter[3] ^ key[1],
            p0 as u32,
        ];
        key[0] = key[0].wrapping_add(2_654_435_769);
        key[1] = key[1].wrapping_add(3_144_134_277);
    }
    let uniform = ((counter[0] >> 9) as f32 + 0.5) / 8_388_608.0;
    value - (-uniform.ln()).ln()
}

#[test]
// Remove this ignore as part of compiler-convergence coverage once the
// reference evaluator supports the index-to-i32 cast used by token history.
#[ignore = "Seismic reference math lacks the IDX-to-I32 cast required by Hn=64 history matching"]
fn portable_shaping_matches_the_ordered_host_reference() {
    use seismic_lang::{
        checked::{check_source, SourceFile, SourceSet},
        interp::{Arg, Interpreter, TensorData},
        types::DType,
    };
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "sampling.seismic".into(),
        text: include_str!("../kernels/sampling.seismic").into(),
    }]))
    .expect("shape_rows source must check independently");
    let logical = module
        .entry(
            module.entry_named("shape_rows").unwrap(),
            &Default::default(),
        )
        .unwrap();
    let (logits, params, history) = fixture();
    let expected = host_reference(&logits, &params, &history);
    let mut interpreter = Interpreter::new(&logical);
    let logits = interpreter.add_tensor(TensorData::dense(
        DType::F32,
        vec![ROWS, VOCABULARY],
        logits.iter().map(|value| f64::from(*value)).collect(),
    ));
    let params = interpreter.add_tensor(TensorData::dense(
        DType::F32,
        vec![ROWS, 8],
        params.iter().map(|value| f64::from(*value)).collect(),
    ));
    let history = interpreter.add_tensor(TensorData::dense(
        DType::I32,
        vec![ROWS, HISTORY],
        history.iter().map(|value| f64::from(*value)).collect(),
    ));
    let out = interpreter.add_tensor(TensorData::dense(
        DType::F32,
        vec![ROWS, VOCABULARY],
        vec![0.0; ROWS * VOCABULARY],
    ));
    let outcome = interpreter
        .run(&[
            Arg::Tensor(logits),
            Arg::Tensor(params),
            Arg::Tensor(history),
            Arg::Tensor(out),
        ])
        .unwrap();
    let out_input = outcome.inputs().nth(3).unwrap();
    let out = out_input.tensor();
    let actual = (0..out.element_count())
        .map(|index| out.read(index).unwrap() as f32)
        .collect::<Vec<_>>();
    assert_values(&actual, &expected);
    assert_eq!(&actual[..VOCABULARY], &[3.0, 2.0, 1.0, 0.0, -1.0, -2.0]);
}

#[test]
fn generated_surface_exposes_planned_and_native_preparation() {
    let planned: fn(&seismic::Device, seismic::PreparationOptions) -> _ = shape_rows::for_device;
    let native: fn(&seismic::Device, &seismic::NativeSpecialization) -> _ =
        shape_rows::native_for_device;
    let _ = (planned, native);
    let planned_sample: fn(&seismic::Device, seismic::PreparationOptions) -> _ =
        sample_rows::for_device;
    let native_sample: fn(&seismic::Device, &seismic::NativeSpecialization) -> _ =
        sample_rows::native_for_device;
    let _ = (planned_sample, native_sample);
}

#[cfg(target_os = "macos")]
#[test]
fn native_sampling_reports_greedy_empty_and_nonfinite_rows() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let values = [1.0f32, 3.0, 2.0, 4.0, 5.0, 6.0, f32::NAN, 0.0, 1.0];
    let logits = seismic::Tensor::from_host(
        &device,
        seismic::Element::f32(),
        &[3, 3],
        &values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mask = seismic::Tensor::from_host(
        &device,
        seismic::Element::u32(),
        &[3, 1],
        &[0b111u32, 0, 0b111]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let draws =
        seismic::Tensor::from_host(&device, seismic::Element::u32(), &[3, 6], &[0; 72]).unwrap();
    let mut result = seismic::Tensor::zeros(&device, seismic::Element::i32(), &[3, 2]).unwrap();
    sample_rows::native_for_device(&device, &seismic::NativeSpecialization::new())
        .unwrap()
        .call(sample_rows::Args {
            logits: &logits,
            mask: &mask,
            draws: &draws,
            result: &mut result,
        })
        .unwrap();
    let actual = result
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual, vec![1, 0, -1, 1, -1, 2]);
}

#[cfg(target_os = "macos")]
#[test]
fn native_sampling_reduces_the_vocabulary_cooperatively_with_stable_ties_and_rng() {
    const VOCAB: usize = 513;
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let mut values = vec![-4.0f32; 2 * VOCAB];
    values[5] = 7.0;
    values[300] = 7.0;
    for token in 0..VOCAB {
        values[VOCAB + token] = (token % 17) as f32 * 0.125 - 1.0;
    }
    let draw = [1, 0x1234_5678, 0x9abc_def0, 7, 11, 13];
    let stochastic = (0..VOCAB)
        .map(|token| philox_score(values[VOCAB + token], token as u32, draw))
        .enumerate()
        .max_by(|(left_token, left), (right_token, right)| {
            left.total_cmp(right)
                .then_with(|| right_token.cmp(left_token))
        })
        .unwrap()
        .0 as i32;
    let logits = seismic::Tensor::from_host(
        &device,
        seismic::Element::f32(),
        &[2, VOCAB as u64],
        &values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mask_words = VOCAB.div_ceil(32);
    let mask_values = vec![u32::MAX; 2 * mask_words];
    let mask = seismic::Tensor::from_host(
        &device,
        seismic::Element::u32(),
        &[2, mask_words as u64],
        &mask_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut draw_values = vec![0u32; 12];
    draw_values[6..].copy_from_slice(&draw);
    let draws = seismic::Tensor::from_host(
        &device,
        seismic::Element::u32(),
        &[2, 6],
        &draw_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut result = seismic::Tensor::zeros(&device, seismic::Element::i32(), &[2, 2]).unwrap();
    sample_rows::native_for_device(&device, &seismic::NativeSpecialization::new())
        .unwrap()
        .call(sample_rows::Args {
            logits: &logits,
            mask: &mask,
            draws: &draws,
            result: &mut result,
        })
        .unwrap();
    let actual = result
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual, vec![5, 0, stochastic, 0]);
}

#[cfg(target_os = "macos")]
#[test]
fn native_metal_shaping_matches_the_host_reference() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let (logits, params, history) = fixture();
    let expected = host_reference(&logits, &params, &history);
    let f32_bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let logits = seismic::Tensor::from_host(
        &device,
        seismic::Element::f32(),
        &[ROWS as u64, VOCABULARY as u64],
        &f32_bytes(&logits),
    )
    .unwrap();
    let params = seismic::Tensor::from_host(
        &device,
        seismic::Element::f32(),
        &[ROWS as u64, 8],
        &f32_bytes(&params),
    )
    .unwrap();
    let history_bytes = history
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let history = seismic::Tensor::from_host(
        &device,
        seismic::Element::i32(),
        &[ROWS as u64, HISTORY as u64],
        &history_bytes,
    )
    .unwrap();
    let mut out = seismic::Tensor::zeros(
        &device,
        seismic::Element::f32(),
        &[ROWS as u64, VOCABULARY as u64],
    )
    .unwrap();
    shape_rows::native_for_device(&device, &seismic::NativeSpecialization::new())
        .unwrap()
        .call(shape_rows::Args {
            logits: &logits,
            params: &params,
            history: &history,
            out: &mut out,
        })
        .unwrap();
    let actual = out
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_values(&actual, &expected);
}

#[cfg(target_os = "macos")]
#[test]
fn native_shaping_preserves_cross_simdgroup_cutoff_ties_and_history_counts() {
    const VOCAB: usize = 513;
    const HISTORY_WIDTH: usize = 513;
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let mut logits = vec![-10.0f32; 2 * VOCAB];
    logits[5] = 4.0;
    logits[300] = 4.0;
    logits[400] = 3.0;
    logits[VOCAB + 5] = 4.0;
    logits[VOCAB + 300] = 4.0;
    let params = [
        1.0, 2.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, // top-p cuts the later score tie
        1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, // frequency uses exact counts
    ];
    let mut history = vec![-1i32; 2 * HISTORY_WIDTH];
    history[HISTORY_WIDTH] = 5;
    history[HISTORY_WIDTH + 1] = 300;
    history[HISTORY_WIDTH + 2] = 300;
    let f32_bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let logits = seismic::Tensor::from_host(
        &device,
        seismic::Element::f32(),
        &[2, VOCAB as u64],
        &f32_bytes(&logits),
    )
    .unwrap();
    let params = seismic::Tensor::from_host(
        &device,
        seismic::Element::f32(),
        &[2, 8],
        &f32_bytes(&params),
    )
    .unwrap();
    let history = seismic::Tensor::from_host(
        &device,
        seismic::Element::i32(),
        &[2, HISTORY_WIDTH as u64],
        &history
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut out =
        seismic::Tensor::zeros(&device, seismic::Element::f32(), &[2, VOCAB as u64]).unwrap();
    shape_rows::native_for_device(&device, &seismic::NativeSpecialization::new())
        .unwrap()
        .call(shape_rows::Args {
            logits: &logits,
            params: &params,
            history: &history,
            out: &mut out,
        })
        .unwrap();
    let actual = out
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual[5], 4.0);
    assert_eq!(actual[300], f32::NEG_INFINITY);
    assert_eq!(actual[400], f32::NEG_INFINITY);
    assert_eq!(actual[VOCAB + 5], 3.0);
    assert_eq!(actual[VOCAB + 300], f32::NEG_INFINITY);
    assert!(actual[..VOCAB]
        .iter()
        .enumerate()
        .all(|(token, value)| token == 5 || *value == f32::NEG_INFINITY));
    assert!(actual[VOCAB..]
        .iter()
        .enumerate()
        .all(|(token, value)| token == 5 || *value == f32::NEG_INFINITY));
}
