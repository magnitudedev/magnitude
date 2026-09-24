use magnitude_model_kernels::{
    qwen_attention_output, qwen_attention_project, qwen_recurrent_output, qwen_recurrent_project,
};
use seismic::{BackendName, Device, DeviceCatalog, Element, Tensor};

const ROWS: usize = 9; // A full eight-row tile and a one-row tail.
const HIDDEN: usize = 4;

fn tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
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

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn assert_close(actual: &Tensor, expected: &[f32], name: &str) {
    let actual = values(actual);
    assert_eq!(actual.len(), expected.len(), "{name} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2.0e-4,
            "{name}[{index}]: Metal {actual}, host {expected}"
        );
    }
}

fn matrix(rows: usize, columns: usize, seed: usize) -> Vec<f32> {
    (0..rows * columns)
        .map(|index| ((index * 11 + seed * 7) % 37) as f32 * 0.013 - 0.2)
        .collect()
}

fn project(input: &[f32], width: usize, weight: &[f32], outputs: usize) -> Vec<f32> {
    let mut result = vec![0.0; ROWS * outputs];
    for row in 0..ROWS {
        for output in 0..outputs {
            let mut sum = 0.0f32;
            for source in 0..width {
                sum = input[row * width + source].mul_add(weight[output * width + source], sum);
            }
            result[row * outputs + output] = sum;
        }
    }
    result
}

#[test]
#[cfg(target_os = "macos")]
fn metal_projection_tiles_match_independent_nine_row_host_matmuls() {
    let catalog = DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(BackendName::Metal).unwrap();
    let f32e = Element::f32();

    // Attention: KV=1, G=1, W=2. Q/gate has four outputs, K and V two each.
    let normalized_values = matrix(ROWS, HIDDEN, 1);
    let normalized = tensor(&device, &[ROWS as u64, HIDDEN as u64], &normalized_values);
    let head_norm = tensor(&device, &[2], &[1.0, 1.0]);
    let query_values = matrix(4, HIDDEN, 2);
    let key_values = matrix(2, HIDDEN, 3);
    let value_values = matrix(2, HIDDEN, 4);
    let query = tensor(&device, &[4, HIDDEN as u64], &query_values);
    let key = tensor(&device, &[2, HIDDEN as u64], &key_values);
    let value = tensor(&device, &[2, HIDDEN as u64], &value_values);
    let attention = qwen_attention_project::native_for_device_with(
        &device,
        qwen_attention_project::Elements {
            QW: f32e,
            KW: f32e,
            VW: f32e,
            A: f32e,
        },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_attention_project::Args {
        normalized: &normalized,
        query_norm: &head_norm,
        query_gate_weight: &query,
        key_weight: &key,
        value_weight: &value,
    })
    .unwrap();
    assert_close(
        &attention.r0,
        &project(&normalized_values, HIDDEN, &query_values, 4),
        "query",
    );
    assert_close(
        &attention.r1,
        &project(&normalized_values, HIDDEN, &key_values, 2),
        "key",
    );
    assert_close(
        &attention.r2,
        &project(&normalized_values, HIDDEN, &value_values, 2),
        "value",
    );

    let hidden_values = matrix(ROWS, HIDDEN, 5);
    let hidden = tensor(&device, &[ROWS as u64, HIDDEN as u64], &hidden_values);
    let gated_values = matrix(ROWS, 2, 6);
    let gated = tensor(&device, &[ROWS as u64, 1, 2], &gated_values);
    let prepared_key_values = matrix(ROWS, 2, 7);
    let prepared_key = tensor(&device, &[ROWS as u64, 1, 2], &prepared_key_values);
    let values_for_history = matrix(ROWS, 2, 8);
    let value_for_history = tensor(&device, &[ROWS as u64, 2], &values_for_history);
    let output_values = matrix(HIDDEN, 2, 9);
    let output_weight = tensor(&device, &[HIDDEN as u64, 2], &output_values);
    let destinations = Tensor::from_host(
        &device,
        Element::i32(),
        &[ROWS as u64],
        &(0..ROWS as i32)
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut history_key = tensor(&device, &[ROWS as u64, 1, 2], &vec![0.0; ROWS * 2]);
    let mut history_value = tensor(&device, &[ROWS as u64, 1, 2], &vec![0.0; ROWS * 2]);
    let attention_output = qwen_attention_output::native_for_device_with(
        &device,
        qwen_attention_output::Elements { OW: f32e, A: f32e },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_attention_output::Args {
        hidden: &hidden,
        gated: &gated,
        prepared_key: &prepared_key,
        value: &value_for_history,
        output_weight: &output_weight,
        destinations: &destinations,
        history_key: &mut history_key,
        history_value: &mut history_value,
    })
    .unwrap()
    .value;
    let expected_attention_output = project(&gated_values, 2, &output_values, HIDDEN)
        .iter()
        .zip(&hidden_values)
        .map(|(projection, hidden)| projection + hidden)
        .collect::<Vec<_>>();
    assert_close(
        &attention_output,
        &expected_attention_output,
        "attention output",
    );
    assert_close(&history_key, &prepared_key_values, "history key");
    assert_close(&history_value, &values_for_history, "history value");

    // Recurrent: NK=NV=1 and W=2. The output uses the same nine-row tail.
    let qkv_values = matrix(6, HIDDEN, 10);
    let gate_values = matrix(2, HIDDEN, 11);
    let alpha_values = matrix(1, HIDDEN, 12);
    let beta_values = matrix(1, HIDDEN, 13);
    let qkv = tensor(&device, &[6, HIDDEN as u64], &qkv_values);
    let gate = tensor(&device, &[2, HIDDEN as u64], &gate_values);
    let alpha = tensor(&device, &[1, HIDDEN as u64], &alpha_values);
    let beta = tensor(&device, &[1, HIDDEN as u64], &beta_values);
    let recurrent = qwen_recurrent_project::native_for_device_with(
        &device,
        qwen_recurrent_project::Elements {
            QW: f32e,
            GW: f32e,
            AW: f32e,
            BW: f32e,
            A: f32e,
        },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_project::Args {
        normalized: &normalized,
        qkv_weight: &qkv,
        gate_weight: &gate,
        alpha_weight: &alpha,
        beta_weight: &beta,
    })
    .unwrap()
    .value;
    let parts = [
        project(&normalized_values, HIDDEN, &qkv_values, 6),
        project(&normalized_values, HIDDEN, &gate_values, 2),
        project(&normalized_values, HIDDEN, &alpha_values, 1),
        project(&normalized_values, HIDDEN, &beta_values, 1),
    ];
    let mut expected_recurrent = Vec::with_capacity(ROWS * 10);
    for row in 0..ROWS {
        for part in &parts {
            let width = part.len() / ROWS;
            expected_recurrent.extend_from_slice(&part[row * width..(row + 1) * width]);
        }
    }
    assert_close(&recurrent, &expected_recurrent, "recurrent projection");
    let recurrent_output = qwen_recurrent_output::native_for_device_with(
        &device,
        qwen_recurrent_output::Elements { OW: f32e, A: f32e },
        &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_output::Args {
        hidden: &hidden,
        gated: &tensor(&device, &[ROWS as u64, 2], &gated_values),
        output_weight: &output_weight,
    })
    .unwrap()
    .value;
    assert_close(
        &recurrent_output,
        &expected_attention_output,
        "recurrent output",
    );
}
