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

const PARTITIONS: [u64; 3] = [64, 32, 128];

#[cfg(target_os = "macos")]
fn metal() -> seismic::Device {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    catalog.open_backend(seismic::BackendName::Metal).unwrap()
}

#[cfg(target_os = "macos")]
fn native_shape(device: &seismic::Device, logits: &[f32], rows: usize, vocabulary: usize,
    params: &[f32], history: &[i32], parts: u64) -> Vec<f32> {
    let hn = history.len() / rows;
    let f32_bytes = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let logits = seismic::Tensor::from_host(device, seismic::Element::f32(), &[rows as u64, vocabulary as u64], &f32_bytes(logits)).unwrap();
    let params = seismic::Tensor::from_host(device, seismic::Element::f32(), &[rows as u64, 8], &f32_bytes(params)).unwrap();
    let history = seismic::Tensor::from_host(device, seismic::Element::i32(), &[rows as u64, hn as u64],
        &history.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    let mut out = seismic::Tensor::zeros(device, seismic::Element::f32(), &[rows as u64, vocabulary as u64]).unwrap();
    shape_rows::native_for_device(
        device,
        &seismic::NativeSpecialization::new().with_param("PARTS", parts),
    )
    .unwrap()
    .call(shape_rows::Args { logits: &logits, params: &params, history: &history, out: &mut out })
    .unwrap();
    out.read_to_host().unwrap().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
}

#[cfg(target_os = "macos")]
fn native_sample(device: &seismic::Device, logits: &[f32], rows: usize, vocabulary: usize, mask: &[u32],
    constrained: &[i32], draws: &[u32], parts: u64) -> Vec<i32> {
    let words = vocabulary.div_ceil(32);
    let u32_bytes = |values: &[u32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
    let logits = seismic::Tensor::from_host(device, seismic::Element::f32(), &[rows as u64, vocabulary as u64],
        &logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    let mask = seismic::Tensor::from_host(device, seismic::Element::u32(), &[rows as u64, words as u64], &u32_bytes(mask)).unwrap();
    let constrained = seismic::Tensor::from_host(device, seismic::Element::i32(), &[rows as u64],
        &constrained.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    let draws = seismic::Tensor::from_host(device, seismic::Element::u32(), &[rows as u64, 6], &u32_bytes(draws)).unwrap();
    let mut result = seismic::Tensor::zeros(device, seismic::Element::i32(), &[rows as u64, 2]).unwrap();
    sample_rows::native_for_device(
        device,
        &seismic::NativeSpecialization::new().with_param("PARTS", parts),
    )
    .unwrap()
    .call(sample_rows::Args { logits: &logits, mask: &mask, constrained: &constrained, draws: &draws, result: &mut result })
    .unwrap();
    result.read_to_host().unwrap().chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect()
}

/// Deterministic logits of an approximately normal shape; `quantized` rounds
/// them to quarters so many ties exist.
fn logits_pattern(count: usize, seed: u64, quantized: bool) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut uniform = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / (1u64 << 24) as f32
    };
    (0..count)
        .map(|_| {
            let value = (uniform() + uniform() + uniform() + uniform() - 2.0) * 4.0;
            if quantized { (value * 4.0).round() / 4.0 } else { value }
        })
        .collect()
}

fn sample_reference(logits: &[f32], mask: &[u32], constrained: bool, draw: [u32; 6]) -> (i32, i32) {
    let mut bad = false;
    let mut best: Option<(f32, usize)> = None;
    for (token, value) in logits.iter().enumerate() {
        bad |= value.is_nan() || *value == f32::INFINITY;
        if (constrained && (mask[token / 32] >> (token % 32)) & 1 == 0) || !value.is_finite() {
            continue;
        }
        let score = if draw[0] == 1 { philox_score(*value, token as u32, draw) } else { *value };
        if best.map_or(true, |(b, _)| score > b) {
            best = Some((score, token));
        }
    }
    match (bad, best) {
        (true, _) => (-1, 2),
        (false, Some((_, token))) => (token as i32, 0),
        (false, None) => (-1, 1),
    }
}

/// The portable shaping semantics in O(V log V); `edge` marks tokens whose
/// top-p decision lies within rounding of the threshold.
fn shaping_reference(logits: &[f32], p: &[f32], history: &[i32]) -> (Vec<f32>, Vec<bool>) {
    let vocabulary = logits.len();
    let mut values = logits.to_vec();
    for (token, value) in values.iter_mut().enumerate() {
        let count = history.iter().filter(|h| **h == token as i32).count();
        if count > 0 {
            *value = if *value < 0.0 { *value * p[4] } else { *value / p[4] };
            *value = *value - p[5] - p[6] * count as f32;
        }
    }
    let mut edge = vec![false; vocabulary];
    if p[0] == 0.0 {
        return (values, edge);
    }
    for value in values.iter_mut() {
        *value /= p[0];
    }
    if values.iter().any(|x| x.is_nan() || *x == f32::INFINITY) {
        return (values, edge);
    }
    let top_k = p[1] as i32;
    if top_k > 0 {
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let kth = sorted[(top_k as usize).min(vocabulary) - 1];
        for value in values.iter_mut() {
            if *value < kth {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if p[3] > 0.0 {
        for value in values.iter_mut() {
            if (*value - maximum).exp() < p[3] {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    if p[2] < 1.0 {
        let weights = values.iter().map(|x| f64::from((*x - maximum).exp())).collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f64>();
        let mut order = (0..vocabulary).collect::<Vec<_>>();
        // IEEE order: -0.0 and +0.0 are one tie group, as in the portable body.
        order.sort_by(|a, b| values[*b].partial_cmp(&values[*a]).unwrap().then(a.cmp(b)));
        let mut before = 0f64;
        for token in order {
            let preceding = before / denominator;
            edge[token] = (preceding - f64::from(p[2])).abs() < 1e-5;
            if preceding >= f64::from(p[2]) {
                values[token] = f32::NEG_INFINITY;
            }
            before += weights[token];
        }
    }
    (values, edge)
}

#[cfg(target_os = "macos")]
#[test]
fn native_metal_shaping_matches_the_host_reference() {
    let device = metal();
    let (logits, params, history) = fixture();
    let expected = host_reference(&logits, &params, &history);
    for parts in PARTITIONS {
        let actual = native_shape(&device, &logits, ROWS, VOCABULARY, &params, &history, parts);
        assert_values(&actual, &expected);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn native_shaping_preserves_cross_partition_cutoff_ties_and_history_counts() {
    const VOCAB: usize = 513;
    let device = metal();
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
    let mut history = vec![-1i32; 2 * VOCAB];
    history[VOCAB] = 5;
    history[VOCAB + 1] = 300;
    history[VOCAB + 2] = 300;
    for parts in PARTITIONS {
        let actual = native_shape(&device, &logits, 2, VOCAB, &params, &history, parts);
        assert_eq!(actual[5], 4.0);
        assert_eq!(actual[300], f32::NEG_INFINITY);
        assert_eq!(actual[400], f32::NEG_INFINITY);
        assert_eq!(actual[VOCAB + 5], 3.0);
        assert_eq!(actual[VOCAB + 300], f32::NEG_INFINITY);
        assert!(actual[..VOCAB].iter().enumerate().all(|(t, v)| t == 5 || *v == f32::NEG_INFINITY));
        assert!(actual[VOCAB..].iter().enumerate().all(|(t, v)| t == 5 || *v == f32::NEG_INFINITY));
    }
}

#[cfg(target_os = "macos")]
#[test]
fn native_shaping_matches_the_ordered_semantics_at_full_vocabulary_for_every_partition_count() {
    let device = metal();
    let param_rows: [[f32; 8]; 8] = [
        [0.7, 40.0, 0.9, 0.05, 1.1, 0.2, 0.1, 0.0],
        [1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.3, 1_000_000.0, 0.95, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 40.0, 0.9, 0.05, 1.3, 0.5, 0.25, 0.0],
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.8, 20.0, 1.0, 0.1, 1.0, 0.0, 0.0, 0.0],
        [1.0, 64.0, 0.99, 0.0, 1.0, 0.0, 0.0, 0.0],
    ];
    let rows = param_rows.len();
    let params = param_rows.iter().flatten().copied().collect::<Vec<_>>();
    for (vocabulary, quantized) in [(248_320usize, false), (248_320, true), (1000, true), (513, false)] {
        let logits = logits_pattern(rows * vocabulary, vocabulary as u64 + u64::from(quantized), quantized);
        let history = (0..rows * 64).map(|i| if i % 3 == 0 { -1 } else { ((i * 7919) % (vocabulary / 2)) as i32 }).collect::<Vec<_>>();
        let base = native_shape(&device, &logits, rows, vocabulary, &params, &history, PARTITIONS[0]);
        for parts in &PARTITIONS[1..] {
            let other = native_shape(&device, &logits, rows, vocabulary, &params, &history, *parts);
            assert!(other.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()),
                "V {vocabulary}: PARTS {parts} changes the result");
        }
        for row in 0..rows {
            let (expected, edge) = shaping_reference(&logits[row * vocabulary..(row + 1) * vocabulary],
                &param_rows[row], &history[row * 64..(row + 1) * 64]);
            let actual = &base[row * vocabulary..(row + 1) * vocabulary];
            let wrong = (0..vocabulary)
                .filter(|t| !edge[*t] && actual[*t].to_bits() != expected[*t].to_bits()
                    && (actual[*t].is_infinite() || expected[*t].is_infinite()
                        || (actual[*t] - expected[*t]).abs() > 1e-6 * expected[*t].abs()))
                .collect::<Vec<_>>();
            assert!(wrong.is_empty(), "V {vocabulary} row {row}: {} tokens differ, first {:?}",
                wrong.len(), wrong.iter().take(4).map(|t| (*t, actual[*t], expected[*t])).collect::<Vec<_>>());
        }
    }
    // Equal logits: the top-p tie rule keeps the lowest indices.
    for (vocabulary, top_p, kept) in [(1000usize, 0.5f32, 500usize), (248_320, 0.3, 74_496), (1000, 0.0015, 2)] {
        let logits = vec![1.5f32; vocabulary];
        for parts in PARTITIONS {
            let out = native_shape(&device, &logits, 1, vocabulary, &[1.0, 0.0, top_p, 0.0, 1.0, 0.0, 0.0, 0.0], &[-1; 64], parts);
            let survivors = (0..vocabulary).filter(|t| out[*t] > f32::NEG_INFINITY).collect::<Vec<_>>();
            assert_eq!(survivors, (0..kept).collect::<Vec<_>>(), "V {vocabulary} top-p {top_p} PARTS {parts}");
        }
    }
    // A NaN row is left unfiltered (sampling reports it).
    let mut logits = logits_pattern(1000, 3, false);
    logits[17] = f32::NAN;
    let out = native_shape(&device, &logits, 1, 1000, &[1.0, 5.0, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0], &[-1; 64], 64);
    assert!(out.iter().all(|v| *v != f32::NEG_INFINITY));
}

#[cfg(target_os = "macos")]
#[test]
fn native_sampling_statuses_ties_masks_and_winners_are_partition_invariant() {
    let device = metal();
    for vocabulary in [248_320usize, 1000, 513, 33] {
        let rows = 6;
        let words = vocabulary.div_ceil(32);
        let mut logits = logits_pattern(rows * vocabulary, vocabulary as u64, vocabulary < 100);
        let mut mask = vec![u32::MAX; rows * words];
        if vocabulary % 32 != 0 {
            for row in 0..rows {
                mask[row * words + words - 1] = (1u32 << (vocabulary % 32)) - 1;
            }
        }
        // Row 1: a greedy tie between an early and the last token.
        logits[vocabulary..2 * vocabulary].fill(-1.0);
        logits[vocabulary + 3] = 5.0;
        logits[2 * vocabulary - 1] = 5.0;
        // Row 2: only the last token (the mask boundary) competes.
        mask[2 * words..3 * words].fill(0);
        mask[3 * words - 1] = 1u32 << ((vocabulary - 1) % 32);
        // Row 3: nothing competes: status 1.
        mask[3 * words..4 * words].fill(0);
        // Row 4: a masked-out +inf still makes the row invalid: status 2.
        logits[4 * vocabulary + vocabulary / 2] = f32::INFINITY;
        mask[4 * words + (vocabulary / 2) / 32] &= !(1u32 << ((vocabulary / 2) % 32));
        // Rows 2..=4 are constrained; the others admit the vocabulary, so
        // their all-zero mask rows are ignored.
        let constrained = [0, 0, 1, 1, 1, 0];
        for row in [0usize, 1, 5] {
            mask[row * words..(row + 1) * words].fill(0);
        }
        let mut draws = vec![0u32; rows * 6];
        for row in [0usize, 2, 5] {
            draws[row * 6..row * 6 + 6].copy_from_slice(&[1, 0x1234_5678 + row as u32, 0x9abc_def0, 7 + row as u32, 11, 13]);
        }
        let base = native_sample(&device, &logits, rows, vocabulary, &mask, &constrained, &draws, PARTITIONS[0]);
        for parts in &PARTITIONS[1..] {
            assert_eq!(native_sample(&device, &logits, rows, vocabulary, &mask, &constrained, &draws, *parts), base,
                "V {vocabulary}: PARTS {parts} changes the winners");
        }
        for row in 0..rows {
            let draw: [u32; 6] = draws[row * 6..row * 6 + 6].try_into().unwrap();
            let expected = sample_reference(&logits[row * vocabulary..(row + 1) * vocabulary],
                &mask[row * words..(row + 1) * words], constrained[row] != 0, draw);
            assert_eq!((base[2 * row], base[2 * row + 1]), expected, "V {vocabulary} row {row}");
        }
        assert_eq!((base[2], base[3]), (3, 0), "V {vocabulary}: ties go to the lowest token");
        assert_eq!((base[4], base[5]), ((vocabulary - 1) as i32, 0), "V {vocabulary}: mask boundary");
        assert_eq!((base[6], base[7]), (-1, 1), "V {vocabulary}: empty row");
        assert_eq!((base[8], base[9]), (-1, 2), "V {vocabulary}: non-finite row");
    }
}
