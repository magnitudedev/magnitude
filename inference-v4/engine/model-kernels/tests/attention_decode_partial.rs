use magnitude_model_kernels::qwen_attention_attend;
use seismic::{BackendName, Device, DeviceCatalog, Element, Tensor};

fn floats(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    Tensor::from_host(
        device,
        Element::f32(),
        shape,
        &values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn ints(device: &Device, shape: &[u64], values: &[i32]) -> Tensor {
    Tensor::from_host(
        device,
        Element::i32(),
        shape,
        &values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[test]
#[cfg(target_os = "macos")]
fn decode_attention_merges_gqa_masked_fresh_and_extreme_partial_states() {
    const KV: usize = 2;
    const G: usize = 2;
    const W: usize = 2;
    const T: usize = 11;
    let device = DeviceCatalog::discover()
        .unwrap()
        .open_backend(BackendName::Metal)
        .unwrap();
    let query_values = [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, -1.0];
    let query = floats(&device, &[1, (KV * G) as u64, W as u64], &query_values);
    let prepared_key_values = [800.0, 0.0, 0.0, -800.0];
    let prepared_key = floats(&device, &[1, KV as u64, W as u64], &prepared_key_values);
    let fresh_value_values = [3.0, -2.0, 2.0, 5.0];
    let fresh_value = floats(&device, &[1, (KV * W) as u64], &fresh_value_values);
    let gate_values = [0.0, 1.0, -1.0, 0.5, 0.25, -0.5, 1.0, -1.0];
    let gate = floats(&device, &[1, (KV * G) as u64, W as u64], &gate_values);
    let mut history_key_values = vec![0.0; T * KV * W];
    let mut history_value_values = vec![0.0; T * KV * W];
    for token in 0..T {
        for kv in 0..KV {
            for column in 0..W {
                let offset = (token * KV + kv) * W + column;
                history_key_values[offset] = if column == 0 {
                    (token as f32 - 5.0) * 170.0
                } else {
                    (kv as f32 * 2.0 - 1.0) * (token as f32 - 3.0) * 180.0
                };
                history_value_values[offset] =
                    (token as f32 * 0.13 + kv as f32 * 0.7 + column as f32 * 0.3) - 0.4;
            }
        }
    }
    let history_key = floats(
        &device,
        &[T as u64, KV as u64, W as u64],
        &history_key_values,
    );
    let history_value = floats(
        &device,
        &[T as u64, KV as u64, W as u64],
        &history_value_values,
    );
    let kernel = qwen_attention_attend::native_for_device_with(
        &device,
        qwen_attention_attend::Elements { A: Element::f32() },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap();

    for (case, spans, fresh_range) in [
        ("masked and fresh", [0, 9, 10, 11], [0, 1]),
        ("empty", [0, 0, 0, 0], [0, 0]),
    ] {
        let visible = ints(&device, &[1, 2, 2], &spans);
        let fresh = ints(&device, &[1, 2], &fresh_range);
        let mut accumulator = floats(&device, &[1, (KV * G) as u64, W as u64], &[0.0; KV * G * W]);
        let actual = kernel
            .call(qwen_attention_attend::Args {
                query: &query,
                prepared_key: &prepared_key,
                value: &fresh_value,
                gate: &gate,
                visible: &visible,
                fresh: &fresh,
                history_key: &history_key,
                history_value: &history_value,
                accumulator: &mut accumulator,
                scale: 1.0,
            })
            .unwrap()
            .value;
        let actual = values(&actual);
        for head in 0..KV * G {
            let kv = head / G;
            let mut entries = Vec::new();
            for pair in spans.chunks_exact(2) {
                for token in pair[0]..pair[1] {
                    entries.push((
                        &history_key_values[(token as usize * KV + kv) * W..][..W],
                        &history_value_values[(token as usize * KV + kv) * W..][..W],
                    ));
                }
            }
            if fresh_range[1] > fresh_range[0] {
                entries.push((
                    &prepared_key_values[kv * W..][..W],
                    &fresh_value_values[kv * W..][..W],
                ));
            }
            let scores = entries
                .iter()
                .map(|(key, _)| {
                    (0..W).fold(0.0f32, |sum, column| {
                        query_values[head * W + column].mul_add(key[column], sum)
                    })
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probabilities = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = probabilities.iter().sum::<f32>();
            for column in 0..W {
                let weighted = entries
                    .iter()
                    .zip(&probabilities)
                    .map(|((_, value), probability)| value[column] * probability)
                    .sum::<f32>();
                let attended = if denominator > 0.0 {
                    weighted / denominator
                } else {
                    0.0
                };
                let gate = gate_values[head * W + column];
                let expected = attended / (1.0 + (-gate).exp());
                let index = head * W + column;
                assert!(
                    (actual[index] - expected).abs() < 2.0e-4,
                    "{case} head={head} column={column}: Metal {} host {expected}",
                    actual[index]
                );
            }
        }
    }
}
