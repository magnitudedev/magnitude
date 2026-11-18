mod common;
mod decision;
mod declared;

use decision::{Cut, Quantity};
use magnitude_kernels::{sample_rows, shape_rows};

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

/// The Gumbel score of `value` at `token` under `draw`, as the host
/// computes it, with its rounding scale: the uniform is exact (a 23-bit
/// integer and a half over 2^23), each logarithm rounds once and the outer
/// one carries the inner's scale through its slope `1 / |ln u|`, and the
/// difference with the logit rounds once.
fn philox_score(value: f32, token: u32, draw: [u32; 6]) -> Quantity {
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
    let inner = Quantity::rounded(f64::from(uniform.ln()), 1);
    let outer = (-uniform.ln()).ln();
    let carried = Quantity::carried(f64::from(outer), inner.scale / inner.value.abs());
    let score = value - outer;
    Quantity::carried(f64::from(score), carried.scale)
}

#[test]
// Remove this ignore as part of compiler-convergence coverage once the
// reference evaluator supports the index-to-i32 cast used by token history.
#[ignore = "Seismic reference math lacks the IDX-to-I32 cast required by Hn=64 history matching"]
fn portable_shaping_matches_the_ordered_host_reference() {
    use seismic_lang::{
        interp::{Arg, Interpreter, TensorData},
        types::DType,
    };
    let module = crate::common::kernels();
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
    let mask = interpreter.add_tensor(TensorData::dense(
        DType::U32,
        vec![ROWS, VOCABULARY.div_ceil(32)],
        vec![0.0; ROWS * VOCABULARY.div_ceil(32)],
    ));
    let constrained =
        interpreter.add_tensor(TensorData::dense(DType::I32, vec![ROWS], vec![0.0; ROWS]));
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
    let outcome = interpreter
        .run(&[
            Arg::Tensor(logits),
            Arg::Tensor(mask),
            Arg::Tensor(constrained),
            Arg::Tensor(params),
            Arg::Tensor(history),
        ])
        .unwrap();
    // The rows are shaped in place.
    let logits_input = outcome.inputs().next().unwrap();
    let shaped = logits_input.tensor();
    let actual = (0..shaped.element_count())
        .map(|index| shaped.read(index).unwrap() as f32)
        .collect::<Vec<_>>();
    assert_values(&actual, &expected);
    assert_eq!(&actual[..VOCABULARY], &[3.0, 2.0, 1.0, 0.0, -1.0, -2.0]);
}

#[test]
fn generated_surface_exposes_planned_and_native_preparation() {
    let planned: fn(&seismic::Charges, seismic::PreparationOptions) -> _ = shape_rows::for_device;
    let native: fn(&seismic::Device, &seismic::NativeSpecialization) -> _ =
        shape_rows::native_for_device;
    let _ = (planned, native);
    let planned_sample: fn(&seismic::Charges, seismic::PreparationOptions) -> _ =
        sample_rows::for_device;
    let native_sample: fn(&seismic::Device, &seismic::NativeSpecialization) -> _ =
        sample_rows::native_for_device;
    let _ = (planned_sample, native_sample);
}

const PARTITIONS: [u64; 3] = [64, 32, 128];

/// Every accelerator the host has, then the CPU; a requested backend alone
/// ([`seismic_native_tests::devices`]).
fn devices() -> Vec<seismic::Device> {
    use seismic::BackendName::{Cpu, Cuda, Metal, Vulkan};
    seismic_native_tests::devices(&[Metal, Cuda, Vulkan, Cpu])
}

fn is_cpu(device: &seismic::Device) -> bool {
    device.backend() == seismic::BackendName::Cpu
}

/// The partition counts `device`'s native forms declare: the CPU's one,
/// the pool's (`device.units`, `None`: no setting), else each PARTS.
fn partitions(device: &seismic::Device) -> Vec<Option<u64>> {
    if is_cpu(device) {
        vec![None]
    } else {
        PARTITIONS.into_iter().map(Some).collect()
    }
}

/// PARTS on `device` where it is a setting, plus the static dimensions and
/// WIDTH of the CUDA and Vulkan forms (Metal and the CPU have neither).
fn specialization_on(
    device: &seismic::Device,
    parts: Option<u64>,
    statics: &[(&str, u64)],
) -> seismic::NativeSpecialization {
    let specialization = parts
        .into_iter()
        .fold(seismic::NativeSpecialization::new(), |specialization, parts| specialization.with_param("PARTS", parts));
    if matches!(
        device.backend(),
        seismic::BackendName::Metal | seismic::BackendName::Cpu
    ) {
        return specialization;
    }
    statics
        .iter()
        .fold(specialization, |specialization, (name, value)| {
            specialization.with_static(*name, *value)
        })
        .with_param("WIDTH", 256)
}

/// Shaping of unconstrained rows (their all-zero masks are never read).
fn native_shape(
    device: &seismic::Device,
    logits: &[f32],
    rows: usize,
    vocabulary: usize,
    params: &[f32],
    history: &[i32],
    parts: Option<u64>,
) -> Vec<f32> {
    native_shape_constrained(
        device,
        logits,
        rows,
        vocabulary,
        &vec![0; rows * vocabulary.div_ceil(32)],
        &vec![0; rows],
        params,
        history,
        parts,
    )
}

#[allow(clippy::too_many_arguments)]
fn native_shape_constrained(
    device: &seismic::Device,
    logits: &[f32],
    rows: usize,
    vocabulary: usize,
    mask: &[u32],
    constrained: &[i32],
    params: &[f32],
    history: &[i32],
    parts: Option<u64>,
) -> Vec<f32> {
    let charges = seismic::Charges::tally(device);
    let hn = history.len() / rows;
    let f32_bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let mut logits = seismic::Tensor::from_host(
        &charges,
        seismic::Element::f32(),
        &[rows as u64, vocabulary as u64],
        &f32_bytes(logits),
    )
    .unwrap();
    let mask = seismic::Tensor::from_host(
        &charges,
        seismic::Element::u32(),
        &[rows as u64, vocabulary.div_ceil(32) as u64],
        &mask
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let constrained = seismic::Tensor::from_host(
        &charges,
        seismic::Element::i32(),
        &[rows as u64],
        &constrained
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let params = seismic::Tensor::from_host(
        &charges,
        seismic::Element::f32(),
        &[rows as u64, 8],
        &f32_bytes(params),
    )
    .unwrap();
    let history = seismic::Tensor::from_host(
        &charges,
        seismic::Element::i32(),
        &[rows as u64, hn as u64],
        &history
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let specialization = specialization_on(
        device,
        parts,
        &[("V", vocabulary as u64), ("Hn", hn as u64)],
    );
    declared::prepared(device, &shape_rows::native_entry(), &specialization)
        .unwrap()
        .call(&charges, shape_rows::Args {
            logits: &mut logits,
            mask: &mask,
            constrained: &constrained,
            params: &params,
            history: &history,
        })
        .unwrap();
    logits
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn native_sample(
    device: &seismic::Device,
    logits: &[f32],
    rows: usize,
    vocabulary: usize,
    mask: &[u32],
    constrained: &[i32],
    draws: &[u32],
    parts: Option<u64>,
) -> Vec<i32> {
    let words = vocabulary.div_ceil(32);
    let charges = seismic::Charges::tally(device);
    let u32_bytes = |values: &[u32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let logits = seismic::Tensor::from_host(
        &charges,
        seismic::Element::f32(),
        &[rows as u64, vocabulary as u64],
        &logits
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mask = seismic::Tensor::from_host(
        &charges,
        seismic::Element::u32(),
        &[rows as u64, words as u64],
        &u32_bytes(mask),
    )
    .unwrap();
    let constrained = seismic::Tensor::from_host(
        &charges,
        seismic::Element::i32(),
        &[rows as u64],
        &constrained
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let draws = seismic::Tensor::from_host(
        &charges,
        seismic::Element::u32(),
        &[rows as u64, 6],
        &u32_bytes(draws),
    )
    .unwrap();
    let mut result =
        seismic::Tensor::zeros(&charges, seismic::Element::i32(), &[rows as u64, 2]).unwrap();
    let specialization = specialization_on(device, parts, &[("V", vocabulary as u64)]);
    declared::prepared(device, &sample_rows::native_entry(), &specialization)
        .unwrap()
        .call(&charges, sample_rows::Args {
            logits: &logits,
            mask: &mask,
            constrained: &constrained,
            draws: &draws,
            result: &mut result,
        })
        .unwrap();
    result
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Deterministic logits of an approximately normal shape; `quantized` rounds
/// them to quarters so many ties exist.
fn logits_pattern(count: usize, seed: u64, quantized: bool) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / (1u64 << 24) as f32
    };
    (0..count)
        .map(|_| {
            let value = (uniform() + uniform() + uniform() + uniform() - 2.0) * 4.0;
            if quantized {
                (value * 4.0).round() / 4.0
            } else {
                value
            }
        })
        .collect()
}

/// The host's draw of one row: its status, its winner, and every token's
/// score (an ineligible token's is [`Quantity::INELIGIBLE`]): the logit
/// itself when greedy, its Gumbel score when sampling.
struct Sampled {
    token: i32,
    status: i32,
    scores: Vec<Quantity>,
}

fn sample_reference(logits: &[f32], mask: &[u32], constrained: bool, draw: [u32; 6]) -> Sampled {
    let mut bad = false;
    let mut best: Option<(f32, usize)> = None;
    let mut scores = vec![Quantity::INELIGIBLE; logits.len()];
    for (token, value) in logits.iter().enumerate() {
        bad |= value.is_nan() || *value == f32::INFINITY;
        if (constrained && (mask[token / 32] >> (token % 32)) & 1 == 0) || !value.is_finite() {
            continue;
        }
        let score = if draw[0] == 1 {
            philox_score(*value, token as u32, draw)
        } else {
            Quantity::exact(f64::from(*value))
        };
        scores[token] = score;
        let score = score.value as f32;
        if best.map_or(true, |(b, _)| score > b) {
            best = Some((score, token));
        }
    }
    let (token, status) = match (bad, best) {
        (true, _) => (-1, 2),
        (false, Some((_, token))) => (token as i32, 0),
        (false, None) => (-1, 1),
    };
    Sampled { token, status, scores }
}

/// A native's `(token, status)` against the host's draw: the status
/// exactly, and the winner the host's except where their scores are within
/// rounding of each other (`decision::chosen`).
fn assert_sampled(label: &str, (token, status): (i32, i32), expected: &Sampled) {
    assert_eq!(status, expected.status, "{label}: status");
    if status != 0 {
        assert_eq!(token, expected.token, "{label}: token");
        return;
    }
    let index = |token: i32| usize::try_from(token).unwrap_or(usize::MAX);
    assert!(
        decision::chosen(&expected.scores, &[index(expected.token)], &[index(token)]).is_empty(),
        "{label}: token {token}, reference {}, scores {:?} and {:?}",
        expected.token,
        expected.scores.get(index(token)),
        expected.scores[index(expected.token)]
    );
}

/// The portable shaping semantics in O(V log V), with the cuts each token
/// was decided by (`decision::selected`): its top-k value against the
/// (k + 1)-th where it ranks within k and against the k-th where it does not,
/// its min-p weight against min-p, its preceding top-p mass against top-p,
/// each stage's it reached. A row left unfiltered (greedy, or holding a NaN
/// or +inf) has none.
fn shaping_reference(logits: &[f32], p: &[f32], history: &[i32]) -> (Vec<f32>, Vec<Vec<Cut>>) {
    let vocabulary = logits.len();
    let mut values = logits.to_vec();
    // The F32 operations of each token's value before its cuts: a
    // penalized token's repetition, presence, frequency product and
    // difference, then every token's temperature.
    let mut operations = vec![0usize; vocabulary];
    for (token, value) in values.iter_mut().enumerate() {
        let count = history.iter().filter(|h| **h == token as i32).count();
        if count > 0 {
            *value = if *value < 0.0 {
                *value * p[4]
            } else {
                *value / p[4]
            };
            *value = *value - p[5] - p[6] * count as f32;
            operations[token] = 4;
        }
    }
    let mut cuts = vec![Vec::new(); vocabulary];
    if p[0] == 0.0 {
        return (values, cuts);
    }
    for value in values.iter_mut() {
        *value /= p[0];
    }
    if values.iter().any(|x| x.is_nan() || *x == f32::INFINITY) {
        return (values, cuts);
    }
    let tempered = values
        .iter()
        .zip(&operations)
        .map(|(value, operations)| Quantity::rounded(f64::from(*value), operations + 1))
        .collect::<Vec<_>>();
    let top_k = p[1] as i32;
    if top_k > 0 {
        let mut order = (0..vocabulary).collect::<Vec<_>>();
        order.sort_by(|a, b| values[*b].total_cmp(&values[*a]));
        let k = (top_k as usize).min(vocabulary);
        let kth = order[k - 1];
        // A token ranked within k leaves the top k only by falling below
        // the (k + 1)-th value, one ranked below by rising above the k-th:
        // each is decided against the other side's nearest. With k at the
        // vocabulary, top-k removes nothing and decides no token.
        for (rank, &token) in order.iter().enumerate() {
            let against = if rank < k { order.get(k).copied() } else { Some(kth) };
            if let Some(against) = against {
                cuts[token].push(Cut {
                    quantity: tempered[token],
                    boundary: tempered[against],
                });
            }
        }
        let boundary = values[kth];
        for value in values.iter_mut() {
            if *value < boundary {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let top = values
        .iter()
        .position(|value| *value == maximum)
        .map_or(Quantity::exact(f64::from(maximum)), |token| tempered[token]);
    // A surviving token's weight exp(v - max): the difference carries both
    // values' scales, the exponential carries the difference's through its
    // slope, the weight itself.
    let weight = |token: usize, value: f32| {
        let difference = value - maximum;
        let carried = Quantity::carried(f64::from(difference), tempered[token].scale + top.scale);
        let weight = difference.exp();
        Quantity::carried(f64::from(weight), f64::from(weight) * carried.scale)
    };
    if p[3] > 0.0 {
        for (token, value) in values.iter_mut().enumerate() {
            if *value != f32::NEG_INFINITY {
                cuts[token].push(Cut {
                    quantity: weight(token, *value),
                    boundary: Quantity::exact(f64::from(p[3])),
                });
            }
            if (*value - maximum).exp() < p[3] {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    if p[2] < 1.0 {
        let weights = values
            .iter()
            .enumerate()
            .map(|(token, value)| weight(token, *value))
            .collect::<Vec<_>>();
        let denominator = weights.iter().map(|w| w.value).sum::<f64>();
        let spread = weights.iter().map(|w| w.scale).sum::<f64>();
        let mut order = (0..vocabulary).collect::<Vec<_>>();
        // IEEE order: -0.0 and +0.0 are one tie group, as in the portable body.
        order.sort_by(|a, b| values[*b].partial_cmp(&values[*a]).unwrap().then(a.cmp(b)));
        let (mut before, mut before_spread) = (0f64, 0f64);
        for token in order {
            // The mass before the token: its sum's and the denominator's
            // scales, the sum itself in F64.
            let preceding = before / denominator;
            if values[token] != f32::NEG_INFINITY {
                cuts[token].push(Cut {
                    quantity: Quantity {
                        value: preceding,
                        scale: (before_spread + preceding * spread) / denominator,
                    },
                    boundary: Quantity::exact(f64::from(p[2])),
                });
            }
            if preceding >= f64::from(p[2]) {
                values[token] = f32::NEG_INFINITY;
            }
            before += weights[token].value;
            before_spread += weights[token].scale;
        }
    }
    (values, cuts)
}

/// A native's shaping of one row against the host's: each kept value the
/// host's (its bits, or within 1e-6 relative), and each token kept or
/// removed as the host decided it except within rounding of one of its cuts
/// (`decision::selected`).
fn assert_shaped(label: &str, actual: &[f32], expected: &[f32], cuts: &[Vec<Cut>]) {
    let kept = |row: &[f32]| row.iter().map(|value| *value != f32::NEG_INFINITY).collect::<Vec<_>>();
    let decided = decision::selected(&kept(expected), &kept(actual), |token| cuts[token].clone());
    assert!(
        decided.is_empty(),
        "{label}: {} tokens kept or removed beyond rounding of their cuts, first {:?}",
        decided.len(),
        decided
            .iter()
            .take(4)
            .map(|t| (*t, actual[*t], expected[*t], &cuts[*t]))
            .collect::<Vec<_>>()
    );
    let wrong = (0..expected.len())
        .filter(|t| {
            let (a, e) = (actual[*t], expected[*t]);
            a != f32::NEG_INFINITY
                && e != f32::NEG_INFINITY
                && a.to_bits() != e.to_bits()
                && (a.is_infinite() || e.is_infinite() || (a - e).abs() > 1e-6 * e.abs())
        })
        .collect::<Vec<_>>();
    assert!(
        wrong.is_empty(),
        "{label}: {} kept values differ, first {:?}",
        wrong.len(),
        wrong.iter().take(4).map(|t| (*t, actual[*t], expected[*t])).collect::<Vec<_>>()
    );
}

/// A top-k token is decided against the value across the cut from it, so a
/// native that keeps k - 1 tokens, removing the k-th whose value lies far
/// above the (k + 1)-th, is refused (`decision::selected`), as is one that
/// keeps k + 1.
#[test]
fn shaping_reference_refuses_a_top_k_off_by_one() {
    let logits = [4.0f32, 3.0, 2.0, 1.0, 0.0];
    // Temperature 1, top-k 3; top-p, min-p and penalties off.
    let params = [1.0f32, 3.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let (expected, cuts) = shaping_reference(&logits, &params, &[-1; 4]);
    let kept = |row: &[f32]| row.iter().map(|value| *value != f32::NEG_INFINITY).collect::<Vec<_>>();
    assert_eq!(kept(&expected), [true, true, true, false, false]);
    let decided = |actual: &[f32]| decision::selected(&kept(&expected), &kept(actual), |token| cuts[token].clone());
    assert!(decided(&expected).is_empty());
    let mut fewer = expected.clone();
    fewer[2] = f32::NEG_INFINITY;
    assert_eq!(decided(&fewer), [2], "keeping k - 1 drops the k-th beyond rounding");
    let mut more = expected.clone();
    more[3] = logits[3];
    assert_eq!(decided(&more), [3], "keeping k + 1 keeps the (k + 1)-th beyond rounding");
}

#[test]
fn native_shaping_preserves_unit_mass_when_top_k_precedes_top_p() {
    // The maximum has mass exactly 2^32. Its low word is zero, not u32::MAX;
    // the histogram and total must agree for both tied and unequal peaks.
    // Every case runs, and every mismatch is reported.
    let mut mismatches = Vec::new();
    for device in devices() {
        eprintln!("unit probability mass: {:?}", device.backend());
        for vocabulary in [6usize, 513, 248_320, 262_144] {
            for (second, top_p, keep_second) in [
                (4.0, 0.5, false),
                (4.0, 0.95, true),
                (3.0, 0.8, true),
                (3.0, 0.7, false),
            ] {
                let mut logits = vec![f32::NEG_INFINITY; vocabulary];
                logits[0] = 4.0;
                logits[vocabulary - 1] = second;
                let params = [1.0, 2.0, top_p, 0.0, 1.0, 0.0, 0.0, 0.0];
                for parts in partitions(&device) {
                    let actual = native_shape(&device, &logits, 1, vocabulary, &params, &[-1; 64], parts);
                    let mut expected = vec![f32::NEG_INFINITY; vocabulary];
                    expected[0] = 4.0;
                    if keep_second {
                        expected[vocabulary - 1] = second;
                    }
                    if actual != expected {
                        let kept = |row: &[f32]| row.iter().enumerate().filter(|(_, value)| value.is_finite()).map(|(token, _)| token).collect::<Vec<_>>();
                        mismatches.push(format!(
                            "{:?}, V={vocabulary}, second={second}, top_p={top_p}, PARTS={parts:?}: kept {:?}, expected {:?}",
                            device.backend(),
                            kept(&actual),
                            kept(&expected)
                        ));
                    }
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
}

#[test]
fn native_shaping_matches_the_host_reference() {
    for device in devices() {
        native_shaping_matches_the_host_reference_on(&device);
    }
}

fn native_shaping_matches_the_host_reference_on(device: &seismic::Device) {
    let (logits, params, history) = fixture();
    let expected = host_reference(&logits, &params, &history);
    for parts in partitions(device) {
        let actual = native_shape(device, &logits, ROWS, VOCABULARY, &params, &history, parts);
        assert_values(&actual, &expected);
    }
}

#[test]
fn native_shaping_preserves_cross_partition_cutoff_ties_and_history_counts() {
    for device in devices() {
        cutoff_ties_and_history_counts_on(&device);
    }
}

fn cutoff_ties_and_history_counts_on(device: &seismic::Device) {
    const VOCAB: usize = 513;
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
    for parts in partitions(device) {
        let actual = native_shape(device, &logits, 2, VOCAB, &params, &history, parts);
        assert_eq!(actual[5], 4.0);
        assert_eq!(actual[300], f32::NEG_INFINITY);
        assert_eq!(actual[400], f32::NEG_INFINITY);
        assert_eq!(actual[VOCAB + 5], 3.0);
        assert_eq!(actual[VOCAB + 300], f32::NEG_INFINITY);
        assert!(actual[..VOCAB]
            .iter()
            .enumerate()
            .all(|(t, v)| t == 5 || *v == f32::NEG_INFINITY));
        assert!(actual[VOCAB..]
            .iter()
            .enumerate()
            .all(|(t, v)| t == 5 || *v == f32::NEG_INFINITY));
    }
}

#[test]
fn native_shaping_matches_the_ordered_semantics_at_full_vocabulary_for_every_partition_count() {
    for device in devices() {
        ordered_semantics_at_full_vocabulary_on(&device);
    }
}

fn ordered_semantics_at_full_vocabulary_on(device: &seismic::Device) {
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
    for (vocabulary, quantized) in [
        (248_320usize, false),
        (248_320, true),
        (1000, true),
        (513, false),
    ] {
        let logits = logits_pattern(
            rows * vocabulary,
            vocabulary as u64 + u64::from(quantized),
            quantized,
        );
        let history = (0..rows * 64)
            .map(|i| {
                if i % 3 == 0 {
                    -1
                } else {
                    ((i * 7919) % (vocabulary / 2)) as i32
                }
            })
            .collect::<Vec<_>>();
        let base = native_shape(
            device,
            &logits,
            rows,
            vocabulary,
            &params,
            &history,
            partitions(device)[0],
        );
        for parts in &partitions(device)[1..] {
            let other = native_shape(device, &logits, rows, vocabulary, &params, &history, *parts);
            assert!(
                other
                    .iter()
                    .zip(&base)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "V {vocabulary}: PARTS {parts:?} changes the result"
            );
        }
        for row in 0..rows {
            let (expected, cuts) = shaping_reference(
                &logits[row * vocabulary..(row + 1) * vocabulary],
                &param_rows[row],
                &history[row * 64..(row + 1) * 64],
            );
            assert_shaped(
                &format!("V {vocabulary} row {row}"),
                &base[row * vocabulary..(row + 1) * vocabulary],
                &expected,
                &cuts,
            );
        }
    }
    // Equal logits: the top-p tie rule keeps the lowest indices, exactly as
    // many as the f64 reference keeps (token t survives while t / V < top_p:
    // 500, 74,497 and 2 here; 74,496 / 248,320 = 0.3 < 0.3f).
    for (vocabulary, top_p) in [(1000usize, 0.5f32), (248_320, 0.3), (1000, 0.0015)] {
        let logits = vec![1.5f32; vocabulary];
        let params = [1.0, 0.0, top_p, 0.0, 1.0, 0.0, 0.0, 0.0];
        let (reference, _) = shaping_reference(&logits, &params, &[-1; 64]);
        let kept = (0..vocabulary)
            .filter(|t| reference[*t] > f32::NEG_INFINITY)
            .collect::<Vec<_>>();
        assert_eq!(
            kept,
            (0..kept.len()).collect::<Vec<_>>(),
            "the reference keeps the lowest indices"
        );
        for parts in partitions(device) {
            let out = native_shape(device, &logits, 1, vocabulary, &params, &[-1; 64], parts);
            let survivors = (0..vocabulary)
                .filter(|t| out[*t] > f32::NEG_INFINITY)
                .collect::<Vec<_>>();
            assert_eq!(
                survivors, kept,
                "V {vocabulary} top-p {top_p} PARTS {parts:?}"
            );
        }
    }
    // A NaN row is left unfiltered (sampling reports it).
    let mut logits = logits_pattern(1000, 3, false);
    logits[17] = f32::NAN;
    let out = native_shape(
        device,
        &logits,
        1,
        1000,
        &[1.0, 5.0, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0],
        &[-1; 64],
        partitions(device)[0],
    );
    assert!(out.iter().all(|v| *v != f32::NEG_INFINITY));
}

/// A constrained row's mask applies before the cuts: top-k, min-p and top-p
/// rank and renormalize only the admitted tokens. Regression: masking after
/// top-p emptied every row whose admitted tokens all lay outside the
/// unconstrained nucleus (a grammar-constrained speculative verification row
/// failed with "sampling distribution is empty").
#[test]
fn constrained_shaping_ranks_only_admitted_tokens() {
    for device in devices() {
        constrained_shaping_ranks_only_admitted_tokens_on(&device);
    }
}

fn constrained_shaping_ranks_only_admitted_tokens_on(device: &seismic::Device) {
    let param_rows: [[f32; 8]; 4] = [
        [0.7, 0.0, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 5.0, 0.8, 0.05, 1.1, 0.2, 0.1, 0.0],
        // Unconstrained: its nucleus is the high tokens alone.
        [0.7, 0.0, 0.9, 0.0, 1.0, 0.0, 0.0, 0.0],
        // A NaN at a rejected token leaves the row unfiltered.
        [1.0, 5.0, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0],
    ];
    let constrained = [1, 1, 0, 1];
    let rows = param_rows.len();
    let params = param_rows.iter().flatten().copied().collect::<Vec<_>>();
    for vocabulary in [248_320usize, 513] {
        let words = vocabulary.div_ceil(32);
        // Tokens 0..16 dominate every row; the admitted tokens (every 97th
        // from 100) all lie far outside the unconstrained 0.9 nucleus.
        let mut logits = logits_pattern(rows * vocabulary, vocabulary as u64 + 7, false);
        for row in 0..rows {
            logits[row * vocabulary..row * vocabulary + 16].fill(20.0);
        }
        logits[3 * vocabulary + 1] = f32::NAN;
        let admitted = |token: usize| token >= 100 && (token - 100) % 97 == 0;
        let mut mask = vec![0u32; rows * words];
        for row in 0..rows {
            for token in (0..vocabulary).filter(|token| admitted(*token)) {
                mask[row * words + token / 32] |= 1 << (token % 32);
            }
        }
        let mut history = vec![-1i32; rows * 64];
        history[64] = 100;
        history[65] = 197;
        history[66] = 100;
        let base = native_shape_constrained(
            device,
            &logits,
            rows,
            vocabulary,
            &mask,
            &constrained,
            &params,
            &history,
            partitions(device)[0],
        );
        for parts in &partitions(device)[1..] {
            let other = native_shape_constrained(
                device,
                &logits,
                rows,
                vocabulary,
                &mask,
                &constrained,
                &params,
                &history,
                *parts,
            );
            assert!(
                other
                    .iter()
                    .zip(&base)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "V {vocabulary}: PARTS {parts:?} changes the result"
            );
        }
        for row in 0..rows {
            let source = &logits[row * vocabulary..(row + 1) * vocabulary];
            let masked = source
                .iter()
                .enumerate()
                .map(|(token, value)| {
                    if constrained[row] != 0 && !admitted(token) && *value < f32::INFINITY {
                        f32::NEG_INFINITY
                    } else {
                        *value
                    }
                })
                .collect::<Vec<_>>();
            let (expected, cuts) = shaping_reference(
                &masked,
                &param_rows[row],
                &history[row * 64..(row + 1) * 64],
            );
            assert_shaped(
                &format!("V {vocabulary} row {row}"),
                &base[row * vocabulary..(row + 1) * vocabulary],
                &expected,
                &cuts,
            );
        }
        // The unconstrained nucleus holds no admitted token, so a mask
        // applied after top-p would leave row 0 empty.
        let unconstrained = native_shape(
            device,
            &logits[..vocabulary],
            1,
            vocabulary,
            &params[..8],
            &history[..64],
            partitions(device)[0],
        );
        assert!((0..vocabulary)
            .filter(|token| admitted(*token))
            .all(|token| unconstrained[token] == f32::NEG_INFINITY));
        // Sampling the constrained rows draws an admitted token.
        let draws = (0..rows)
            .flat_map(|row| [1, 0x1234_5678 + row as u32, 0x9abc_def0, 7, 11, 13])
            .collect::<Vec<_>>();
        let sampled = native_sample(
            device,
            &base,
            rows,
            vocabulary,
            &mask,
            &constrained,
            &draws,
            partitions(device)[0],
        );
        for row in [0usize, 1] {
            assert_eq!(sampled[2 * row + 1], 0, "V {vocabulary} row {row} status");
            assert!(
                admitted(sampled[2 * row] as usize),
                "V {vocabulary} row {row} drew a rejected token"
            );
        }
        assert_eq!(sampled[5], 0, "V {vocabulary}: unconstrained row status");
        assert!(
            (sampled[4] as usize) < 16,
            "V {vocabulary}: unconstrained nucleus"
        );
        assert_eq!(sampled[7], 2, "V {vocabulary}: non-finite row status");
    }
}

#[test]
fn native_sampling_statuses_ties_masks_and_winners_are_partition_invariant() {
    for device in devices() {
        sampling_statuses_ties_masks_and_winners_on(&device);
    }
}

fn sampling_statuses_ties_masks_and_winners_on(device: &seismic::Device) {
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
            draws[row * 6..row * 6 + 6].copy_from_slice(&[
                1,
                0x1234_5678 + row as u32,
                0x9abc_def0,
                7 + row as u32,
                11,
                13,
            ]);
        }
        let base = native_sample(
            device,
            &logits,
            rows,
            vocabulary,
            &mask,
            &constrained,
            &draws,
            partitions(device)[0],
        );
        for parts in &partitions(device)[1..] {
            assert_eq!(
                native_sample(
                    device,
                    &logits,
                    rows,
                    vocabulary,
                    &mask,
                    &constrained,
                    &draws,
                    *parts
                ),
                base,
                "V {vocabulary}: PARTS {parts:?} changes the winners"
            );
        }
        for row in 0..rows {
            let draw: [u32; 6] = draws[row * 6..row * 6 + 6].try_into().unwrap();
            let expected = sample_reference(
                &logits[row * vocabulary..(row + 1) * vocabulary],
                &mask[row * words..(row + 1) * words],
                constrained[row] != 0,
                draw,
            );
            assert_sampled(
                &format!("V {vocabulary} row {row}"),
                (base[2 * row], base[2 * row + 1]),
                &expected,
            );
        }
        assert_eq!(
            (base[2], base[3]),
            (3, 0),
            "V {vocabulary}: ties go to the lowest token"
        );
        assert_eq!(
            (base[4], base[5]),
            ((vocabulary - 1) as i32, 0),
            "V {vocabulary}: mask boundary"
        );
        assert_eq!((base[6], base[7]), (-1, 1), "V {vocabulary}: empty row");
        assert_eq!(
            (base[8], base[9]),
            (-1, 2),
            "V {vocabulary}: non-finite row"
        );
    }
}

/// On the CPU, sampling and shaping split the rows' logits, flattened, into
/// one span per pool participant (`device.units`), whatever the row count.
/// At row counts and vocabularies set against the pool, spans that hold no
/// token (fewer tokens than participants), spans of one token, spans that
/// hold several whole rows and spans that cross row edges each give every
/// row the host references' result.
#[test]
fn cpu_flattened_spans_match_the_host_references_at_every_split() {
    use seismic::device::DeviceValues;
    let device = seismic_native_tests::cpu();
    let units = device.properties().value("units") as usize;
    let shaping = [0.7f32, 3.0, 0.9, 0.05, 1.1, 0.2, 0.1, 0.0];
    for (rows, vocabulary) in [
        (1, 1),
        (1, units.div_ceil(2)),
        (1, units),
        (3, units / 2 + 1),
        (4 * units, 3),
        (7, units + 3),
        (2, 33 * units + 5),
    ] {
        let label = format!("{rows} rows of {vocabulary} tokens on {units} participants");
        let logits = logits_pattern(rows * vocabulary, (rows * 31 + vocabulary) as u64, vocabulary < 64);
        let words = vocabulary.div_ceil(32);
        // Odd rows admit every other token; greedy and Gumbel draws alternate.
        let constrained = (0..rows).map(|row| (row % 2) as i32).collect::<Vec<_>>();
        let mask = vec![0x5555_5555u32; rows * words];
        let draws = (0..rows)
            .flat_map(|row| {
                if row % 3 == 0 {
                    [0; 6]
                } else {
                    [1, 0x1234_5678 + row as u32, 0x9abc_def0, 7, 11, 13]
                }
            })
            .collect::<Vec<u32>>();
        let sampled = native_sample(&device, &logits, rows, vocabulary, &mask, &constrained, &draws, None);
        for row in 0..rows {
            let draw: [u32; 6] = draws[row * 6..row * 6 + 6].try_into().unwrap();
            let expected = sample_reference(
                &logits[row * vocabulary..(row + 1) * vocabulary],
                &mask[row * words..(row + 1) * words],
                constrained[row] != 0,
                draw,
            );
            assert_sampled(&format!("{label}: sampled row {row}"), (sampled[2 * row], sampled[2 * row + 1]), &expected);
        }

        let params = shaping.repeat(rows);
        let history = (0..rows * 64)
            .map(|i| if i % 5 == 0 { ((i * 7) % vocabulary) as i32 } else { -1 })
            .collect::<Vec<_>>();
        let shaped = native_shape(&device, &logits, rows, vocabulary, &params, &history, None);
        for row in 0..rows {
            let (expected, cuts) = shaping_reference(
                &logits[row * vocabulary..(row + 1) * vocabulary],
                &shaping,
                &history[row * 64..(row + 1) * 64],
            );
            assert_shaped(
                &format!("{label}: shaped row {row}"),
                &shaped[row * vocabulary..(row + 1) * vocabulary],
                &expected,
                &cuts,
            );
        }
    }
}

/// The decision checks' own rules (`decision`), held here so they run in
/// one test binary of those that declare the module.
mod decision_rules {
    use super::decision::*;

    /// A min-p cut at 0.05, and token weights near it and far from it, each
    /// formed by the reference's `exp` of a difference of about 3.
    fn weight(value: f64) -> Quantity {
        Quantity::carried(value, value * operation(3.0))
    }

    const MIN_P: Quantity = Quantity {
        value: 0.05,
        scale: 0.0,
    };

    #[test]
    fn a_mask_flip_within_rounding_of_its_cut_is_accepted() {
        // The reference keeps a weight a few F32 steps above the cut; the
        // other removes it.
        let at = weight(0.05 + 3.0 * step(0.05));
        assert!(at.value >= MIN_P.value);
        assert!(selected(&[true], &[false], |_| vec![Cut { quantity: at, boundary: MIN_P }]).is_empty());
    }

    #[test]
    fn a_mask_flip_far_from_its_cut_is_rejected() {
        // Removed far above the cut, and kept far below it.
        let (above, below) = (weight(0.2), weight(0.001));
        let cut = |token: usize| vec![Cut { quantity: [above, below][token], boundary: MIN_P }];
        assert_eq!(selected(&[true, false], &[false, true], cut), [0, 1]);
        // A close cut excuses only its own token.
        let at = weight(0.05);
        let cuts = |token: usize| vec![Cut { quantity: [at, below][token], boundary: MIN_P }];
        assert_eq!(selected(&[true, false], &[false, true], cuts), [1]);
    }

    #[test]
    fn a_swapped_choice_between_near_equal_scores_is_accepted() {
        // Two Gumbel scores a rounding apart, then a far one.
        let scores = [
            Quantity::rounded(2.5, 4),
            Quantity::rounded(2.5 + step(2.5), 4),
            Quantity::rounded(-1.0, 4),
        ];
        assert!(chosen(&scores, &[1], &[0]).is_empty());
        // Ranked slots holding the near pair in either order.
        assert!(chosen(&scores, &[1, 0], &[0, 1]).is_empty());
    }

    #[test]
    fn a_swapped_choice_between_distinct_scores_is_rejected() {
        let scores = [Quantity::rounded(2.5, 4), Quantity::rounded(2.0, 4)];
        assert_eq!(chosen(&scores, &[0], &[1]), [0]);
        // Equal scores held exactly (greedy logits) follow the tie rule
        // exactly; an item outside the scores is never a choice.
        let exact = [Quantity::exact(1.0), Quantity::exact(1.0)];
        assert_eq!(chosen(&exact, &[0], &[1]), [0]);
        assert_eq!(chosen(&exact, &[0], &[usize::MAX]), [0]);
        assert_eq!(chosen(&[Quantity::INELIGIBLE, Quantity::INELIGIBLE], &[0], &[1]), [0]);
    }

    #[test]
    fn a_non_monotone_mask_is_rejected() {
        // Weights in descending order: the reference keeps the two above the
        // cut. The other keeps the lowest and removes the highest instead,
        // each beyond rounding of the cut, while the near-cut token agrees.
        let tokens = [weight(0.4), weight(0.05 + 3.0 * step(0.05)), weight(0.01)];
        let cut = |token: usize| vec![Cut { quantity: tokens[token], boundary: MIN_P }];
        assert_eq!(selected(&[true, true, false], &[false, true, true], cut), [0, 2]);
        // A top-k whose k-th and next are a rounding apart is unsettled.
        assert!(settled(&tokens, 1));
        assert!(!settled(&[weight(0.4), weight(0.4 + step(0.4))], 1));
    }
}
