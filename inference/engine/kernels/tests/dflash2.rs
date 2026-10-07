//! DFlash2's entries (`draft_convolve_input`, `draft_convolve_residual`,
//! `draft_gated_rows`, `draft_top_k`, `draft_path_step`) against host models
//! of their contracts: the upstream grouped dynamic causal convolution
//! restarted at every draft block, the gated product, the ordered top-k and
//! one step of the candidate path. The CPU test runs everywhere; the
//! accelerator test covers every other device the host has.

use magnitude_kernels::{
    draft_convolve_input, draft_convolve_residual, draft_gated_rows, draft_path_step, draft_top_k,
};
use seismic::{BackendName, Device, Element, NativeSpecialization, Tensor};

fn bf16_round(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((bf16_round(*value).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn i32_values(bytes: &[u8]) -> Vec<i32> {
    bytes
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn values(count: usize, seed: u32) -> Vec<f32> {
    (0..count)
        .map(|index| {
            ((index as u32).wrapping_mul(2654435761) ^ seed) % 2001
        } as f32
            / 500.0
            - 2.0)
        .collect()
}

fn specialization(
    statics: &[String],
    defaults: impl Fn(&NativeSpecialization) -> NativeSpecialization,
    extents: &[(&str, u64)],
) -> NativeSpecialization {
    defaults(
        &statics
            .iter()
            .fold(NativeSpecialization::new(), |fixed, name| {
                let (_, value) = extents
                    .iter()
                    .find(|(extent, _)| extent == name)
                    .unwrap_or_else(|| panic!("static {name} has no extent"));
                fixed.with_static(name.clone(), *value)
            }),
    )
}

fn tensor(device: &Device, element: Element, shape: &[usize], bytes: &[u8]) -> Tensor {
    let shape = shape
        .iter()
        .map(|extent| *extent as u64)
        .collect::<Vec<_>>();
    Tensor::from_host(device, element, &shape, bytes).unwrap()
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "{what}[{index}]: {a} vs {e}"
        );
    }
}

/// The convolution of one half over rows restarted every `block` rows.
#[allow(clippy::too_many_arguments)]
fn convolve(
    rows: usize,
    groups: usize,
    channels: usize,
    taps: usize,
    block: usize,
    half: usize,
    input: &[f32],
    dynamic: &[f32],
    base: &[f32],
) -> Vec<f32> {
    let width = groups * channels;
    let mut out = vec![0.0f32; rows * width];
    for row in 0..rows {
        for column in 0..width {
            let mut sum = 0.0f32;
            for offset in 0..taps.min(row % block + 1) {
                let coefficient = base[(half * taps + offset) * width + column]
                    + dynamic[row * 2 * taps * groups
                        + (half * taps + offset) * groups
                        + column / channels];
                sum = coefficient.mul_add(input[(row - offset) * width + column], sum);
            }
            out[row * width + column] = sum;
        }
    }
    out
}

fn convolutions_restart_at_every_block(device: &Device) {
    let (rows, groups, channels, taps, block) = (10usize, 4usize, 8usize, 3usize, 4u32);
    let width = groups * channels;
    let extents = [
        ("M", rows as u64),
        ("G", groups as u64),
        ("C", channels as u64),
        ("K", taps as u64),
    ];
    let input_values = values(rows * width, 1)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    let output_values = values(rows * width, 2);
    let residual_values = values(rows * width, 3);
    let dynamic_values = values(rows * 2 * taps * groups, 4)
        .into_iter()
        .map(|value| value / 4.0)
        .collect::<Vec<_>>();
    let base_values = values(2 * taps * width, 5);
    let dynamic = tensor(
        device,
        Element::f32(),
        &[rows, 2, taps, groups],
        &f32_bytes(&dynamic_values),
    );
    let base = tensor(
        device,
        Element::f32(),
        &[2, taps, width],
        &f32_bytes(&base_values),
    );

    let implementation = draft_convolve_input::native_implementation(device)
        .unwrap()
        .expect("draft_convolve_input has a native implementation on every device");
    let kernel = draft_convolve_input::native_for_device_with(
        device,
        draft_convolve_input::Elements { A: Element::bf16() },
        &specialization(
            &implementation.statics,
            |fixed| implementation.default_specialization(fixed).unwrap(),
            &extents,
        ),
    )
    .unwrap();
    let input = tensor(
        device,
        Element::bf16(),
        &[rows, width],
        &bf16_bytes(&input_values),
    );
    let convolved = kernel
        .call(draft_convolve_input::Args {
            input: &input,
            dynamic: &dynamic,
            base: &base,
            block,
        })
        .unwrap()
        .value;
    let expected = convolve(
        rows,
        groups,
        channels,
        taps,
        block as usize,
        0,
        &input_values,
        &dynamic_values,
        &base_values,
    )
    .into_iter()
    .map(bf16_round)
    .collect::<Vec<_>>();
    close(
        &bf16_values(&convolved.read_to_host().unwrap()),
        &expected,
        1.0e-2,
        &format!("{:?} draft_convolve_input", device.backend()),
    );

    let implementation = draft_convolve_residual::native_implementation(device)
        .unwrap()
        .expect("draft_convolve_residual has a native implementation on every device");
    let kernel = draft_convolve_residual::native_for_device(
        device,
        &specialization(
            &implementation.statics,
            |fixed| implementation.default_specialization(fixed).unwrap(),
            &extents,
        ),
    )
    .unwrap();
    let output = tensor(
        device,
        Element::f32(),
        &[rows, width],
        &f32_bytes(&output_values),
    );
    let residual = tensor(
        device,
        Element::f32(),
        &[rows, width],
        &f32_bytes(&residual_values),
    );
    let finished = kernel
        .call(draft_convolve_residual::Args {
            residual: &residual,
            output: &output,
            dynamic: &dynamic,
            base: &base,
            block,
        })
        .unwrap()
        .value;
    let expected = convolve(
        rows,
        groups,
        channels,
        taps,
        block as usize,
        1,
        &output_values,
        &dynamic_values,
        &base_values,
    )
    .iter()
    .zip(&residual_values)
    .map(|(sum, residual)| residual + sum)
    .collect::<Vec<_>>();
    close(
        &f32_values(&finished.read_to_host().unwrap()),
        &expected,
        1.0e-5,
        &format!("{:?} draft_convolve_residual", device.backend()),
    );
}

fn gated_rows_are_silu_times_up(device: &Device) {
    let (rows, width) = (3usize, 300usize);
    let implementation = draft_gated_rows::native_implementation(device)
        .unwrap()
        .expect("draft_gated_rows has a native implementation on every device");
    let kernel = draft_gated_rows::native_for_device_with(
        device,
        draft_gated_rows::Elements { A: Element::bf16() },
        &specialization(
            &implementation.statics,
            |fixed| implementation.default_specialization(fixed).unwrap(),
            &[("M", rows as u64), ("F", width as u64)],
        ),
    )
    .unwrap();
    let gate_values = values(rows * width, 6)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    let up_values = values(rows * width, 7)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    let gated = kernel
        .call(draft_gated_rows::Args {
            gate: &tensor(
                device,
                Element::bf16(),
                &[rows, width],
                &bf16_bytes(&gate_values),
            ),
            up: &tensor(
                device,
                Element::bf16(),
                &[rows, width],
                &bf16_bytes(&up_values),
            ),
        })
        .unwrap()
        .value;
    let expected = gate_values
        .iter()
        .zip(&up_values)
        .map(|(a, b)| bf16_round(a / (1.0 + (-a).exp()) * b))
        .collect::<Vec<_>>();
    close(
        &bf16_values(&gated.read_to_host().unwrap()),
        &expected,
        1.0e-2,
        &format!("{:?} draft_gated_rows", device.backend()),
    );
}

fn top_k_orders_by_value_then_token(device: &Device, vocabulary: usize, count: usize) {
    let rows = 3usize;
    let mut logits = values(rows * vocabulary, 8);
    // Ties: row 1 holds its maximum at tokens 900 and 7; row 2 at 3 tokens.
    logits[vocabulary + 900] = 9.0;
    logits[vocabulary + 7] = 9.0;
    for token in [500, 20, 999] {
        logits[2 * vocabulary + token] = 5.5;
    }
    // A tail shorter than K and ties across partition boundaries must not
    // manufacture candidates or prefer the partition's local token index.
    if vocabulary > 4096 {
        logits[vocabulary - 1] = 10.0;
        logits[vocabulary + 4096] = 9.0;
    }
    if vocabulary > 65_536 {
        logits[2 * vocabulary + 79_469] = 11.0;
    }
    let implementation = draft_top_k::native_implementation(device)
        .unwrap()
        .expect("draft_top_k has a native implementation on every device");
    let kernel = draft_top_k::native_for_device(
        device,
        &specialization(
            &implementation.statics,
            |fixed| implementation.default_specialization(fixed).unwrap(),
            &[
                ("M", rows as u64),
                ("V", vocabulary as u64),
                ("K", count as u64),
            ],
        ),
    )
    .unwrap();
    let mut candidates = tensor(
        device,
        Element::i32(),
        &[rows * count, 2],
        &i32_bytes(&vec![-7; rows * count * 2]),
    );
    let mut unary = tensor(
        device,
        Element::f32(),
        &[rows, count],
        &f32_bytes(&vec![0.0; rows * count]),
    );
    kernel
        .call(draft_top_k::Args {
            logits: &tensor(
                device,
                Element::f32(),
                &[rows, vocabulary],
                &f32_bytes(&logits),
            ),
            candidates: &mut candidates,
            unary: &mut unary,
        })
        .unwrap();
    let mut expected_tokens = Vec::new();
    let mut expected_values = Vec::new();
    for row in 0..rows {
        let mut order = (0..vocabulary).collect::<Vec<_>>();
        let line = &logits[row * vocabulary..][..vocabulary];
        order.sort_by(|a, b| line[*b].total_cmp(&line[*a]).then(a.cmp(b)));
        for &token in &order[..count] {
            expected_tokens.extend([token as i32, 0]);
            expected_values.push(line[token]);
        }
    }
    assert_eq!(
        i32_values(&candidates.read_to_host().unwrap()),
        expected_tokens,
        "{:?} draft_top_k candidates",
        device.backend()
    );
    assert_eq!(f32_values(&unary.read_to_host().unwrap()), expected_values);
}

fn path_step_selects_the_best_joint_score(device: &Device) {
    let (slots, count, rank) = (4usize, 6usize, 64usize);
    let hidden_values = values(slots * rank, 9)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    let predecessor_values = values(slots * rank, 10)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    let successor_values = values(slots * count * rank, 11)
        .into_iter()
        .map(bf16_round)
        .collect::<Vec<_>>();
    let unary_values = values(slots * count, 12);
    let tokens = (0..slots * count)
        .map(|index| 100 + index as i32)
        .collect::<Vec<_>>();
    let candidate_rows = tokens
        .iter()
        .flat_map(|token| [*token, 0])
        .collect::<Vec<_>>();
    let score = |slot: usize, candidate: usize| {
        (0..rank)
            .map(|r| {
                predecessor_values[slot * rank + r]
                    * hidden_values[slot * rank + r]
                    * successor_values[(slot * count + candidate) * rank + r]
            })
            .sum::<f32>()
            + unary_values[slot * count + candidate]
    };
    let mut expected = Vec::new();
    for slot in 0..slots {
        let scores = (0..count)
            .map(|candidate| score(slot, candidate))
            .collect::<Vec<_>>();
        let best = (0..count)
            .max_by(|a, b| scores[*a].total_cmp(&scores[*b]).then(b.cmp(a)))
            .unwrap();
        let mut sorted = scores.clone();
        sorted.sort_by(f32::total_cmp);
        assert!(
            sorted[count - 1] - sorted[count - 2] > 1.0e-3,
            "fixture has a clear winner"
        );
        expected.extend([tokens[slot * count + best], 0]);
    }
    let implementation = draft_path_step::native_implementation(device)
        .unwrap()
        .expect("draft_path_step has a native implementation on every device");
    let kernel = draft_path_step::native_for_device_with(
        device,
        draft_path_step::Elements { A: Element::bf16() },
        &specialization(
            &implementation.statics,
            |fixed| implementation.default_specialization(fixed).unwrap(),
            &[("S", slots as u64), ("K", count as u64), ("R", rank as u64)],
        ),
    )
    .unwrap();
    let mut selection = tensor(
        device,
        Element::i32(),
        &[slots, 2],
        &i32_bytes(&vec![-1; slots * 2]),
    );
    kernel
        .call(draft_path_step::Args {
            candidates: &tensor(
                device,
                Element::i32(),
                &[slots * count, 2],
                &i32_bytes(&candidate_rows),
            ),
            unary: &tensor(
                device,
                Element::f32(),
                &[slots, count],
                &f32_bytes(&unary_values),
            ),
            hidden: &tensor(
                device,
                Element::bf16(),
                &[slots, rank],
                &bf16_bytes(&hidden_values),
            ),
            predecessor: &tensor(
                device,
                Element::bf16(),
                &[slots, rank],
                &bf16_bytes(&predecessor_values),
            ),
            successor: &tensor(
                device,
                Element::bf16(),
                &[slots * count, rank],
                &bf16_bytes(&successor_values),
            ),
            selection: &mut selection,
        })
        .unwrap();
    assert_eq!(
        i32_values(&selection.read_to_host().unwrap()),
        expected,
        "{:?} draft_path_step",
        device.backend()
    );
}

fn entries_match_on(device: &Device) {
    convolutions_restart_at_every_block(device);
    gated_rows_are_silu_times_up(device);
    top_k_orders_by_value_then_token(device, 1000, 5);
    path_step_selects_the_best_joint_score(device);
}

#[test]
fn dflash2_entries_match_the_host_model_on_cpu() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    entries_match_on(&catalog.open_backend(BackendName::Cpu).unwrap());
}

#[test]
fn dflash2_entries_match_the_host_model_on_accelerators() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    for backend in [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        entries_match_on(&device);
    }
}

#[test]
#[ignore = "one-sample full-vocabulary readout timing on the measurement host"]
fn time_top_k_full_vocabulary() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    for backend in [BackendName::Metal, BackendName::Cuda] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        for vocabulary in [65_536usize, 248_320] {
            for rows in [3usize, 7] {
                let count = 16usize;
                let logits = tensor(
                    &device,
                    Element::f32(),
                    &[rows, vocabulary],
                    &f32_bytes(&values(rows * vocabulary, 8)),
                );
                let mut candidates = tensor(
                    &device,
                    Element::i32(),
                    &[rows * count, 2],
                    &i32_bytes(&vec![0; rows * count * 2]),
                );
                let mut unary = tensor(
                    &device,
                    Element::f32(),
                    &[rows, count],
                    &f32_bytes(&vec![0.0; rows * count]),
                );
                let implementation = draft_top_k::native_implementation(&device)
                    .unwrap()
                    .unwrap();
                let kernel = draft_top_k::native_for_device(
                    &device,
                    &implementation
                        .default_specialization(&NativeSpecialization::new())
                        .unwrap(),
                )
                .unwrap();
                let measured = kernel
                    .measure(
                        vec![draft_top_k::Args {
                            logits: &logits,
                            candidates: &mut candidates,
                            unary: &mut unary,
                        }],
                        &seismic::MeasureOptions {
                            samples: 1,
                            min_sample_seconds: 0.0,
                        },
                    )
                    .unwrap();
                eprintln!(
                    "{backend:?} top_k V={vocabulary} M={rows} K={count} seconds={:.9}",
                    measured.median
                );
            }
        }
    }
}

#[test]
fn partitioned_top_k_matches_full_vocabulary_on_accelerators() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    for backend in [BackendName::Metal, BackendName::Cuda] {
        let Ok(device) = catalog.open_backend(backend) else {
            continue;
        };
        for vocabulary in [8192usize, 8193, 8199, 65_536, 248_320] {
            top_k_orders_by_value_then_token(&device, vocabulary, 16);
            eprintln!("{backend:?} partitioned top-k V={vocabulary} passed exact host comparison");
        }
    }
}
