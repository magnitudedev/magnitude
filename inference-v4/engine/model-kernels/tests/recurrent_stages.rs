use magnitude_model_kernels::{
    qwen_recurrent_mix, qwen_recurrent_normalize, qwen_recurrent_output, qwen_recurrent_prepare,
    qwen_recurrent_project, qwen_recurrent_scan,
};
use seismic::{BackendName, Device, DeviceCatalog, Element, Tensor};

fn f32_tensor(device: &Device, shape: &[u64], values: &[f32]) -> Tensor {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    Tensor::from_host(device, Element::f32(), shape, &bytes).unwrap()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn close(actual: &Tensor, expected: &[f32], name: &str) {
    assert_eq!(values(actual).len(), expected.len(), "{name} shape");
    for (index, (actual, expected)) in values(actual)
        .into_iter()
        .zip(expected.iter().copied())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 2.0e-3,
            "{name}[{index}]: staged {actual}, reference {expected}"
        );
    }
}

#[test]
fn staged_recurrent_matches_small_nondegenerate_reference() {
    let catalog = DeviceCatalog::discover().unwrap();
    let Ok(device) = catalog.open_backend(BackendName::Metal) else {
        return;
    };
    let f32 = Element::f32();
    let hidden = f32_tensor(
        &device,
        &[2, 4],
        &[1.0, -2.0, 3.0, 0.5, 0.2, 1.5, -1.0, 2.0],
    );
    let input_norm = f32_tensor(&device, &[4], &[1.0, 0.9, 1.1, 0.8]);
    let qkv_values = (0..24)
        .map(|index| (index as f32 % 7.0 - 3.0) * 0.04)
        .collect::<Vec<_>>();
    let qkv_weight = f32_tensor(&device, &[6, 4], &qkv_values);
    let gate_weight = f32_tensor(
        &device,
        &[2, 4],
        &[0.1, -0.2, 0.3, 0.1, -0.1, 0.2, 0.1, 0.3],
    );
    let alpha_weight = f32_tensor(&device, &[1, 4], &[0.1, 0.02, -0.04, 0.08]);
    let beta_weight = f32_tensor(&device, &[1, 4], &[-0.03, 0.07, 0.09, -0.02]);
    let convolution = f32_tensor(
        &device,
        &[6, 2],
        &[
            0.25, 0.75, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.35, 0.65, 0.15, 0.85,
        ],
    );
    let rate = f32_tensor(&device, &[1], &[-0.4]);
    let time_bias = f32_tensor(&device, &[1], &[0.1]);
    let recurrent_norm = f32_tensor(&device, &[2], &[0.9, 1.1]);
    let output_weight = f32_tensor(
        &device,
        &[4, 2],
        &[0.2, -0.1, -0.15, 0.25, 0.1, 0.3, -0.2, 0.1],
    );
    for (case, segments_values, window_values, delta_values) in [
        (
            "two_rows_one_slot",
            vec![0_i32, 2, 2, 2],
            vec![0.2, -0.1, 0.3, 0.1, -0.2, 0.4],
            vec![0.1, -0.2, 0.05, 0.15],
        ),
        (
            "one_row_per_slot",
            vec![0_i32, 1, 1, 2, 2, 2],
            vec![
                0.2, -0.1, 0.3, 0.1, -0.2, 0.4, -0.15, 0.25, 0.4, -0.3, 0.2, 0.05,
            ],
            vec![0.1, -0.2, 0.05, 0.15, -0.07, 0.11, -0.03, 0.06],
        ),
    ] {
        let batch = (segments_values.len() / 2 - 1) as u64;
        let segments = Tensor::from_host(
            &device,
            Element::i32(),
            &[batch + 1, 2],
            &segments_values
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let window = f32_tensor(&device, &[batch, 1, 6], &window_values);
        let delta = f32_tensor(&device, &[batch, 1, 2, 2], &delta_values);

        let normalized = qwen_recurrent_normalize::native_for_device_with(
            &device,
            qwen_recurrent_normalize::Elements { NW: f32, A: f32 }, &seismic::NativeSpecialization::new(),
        )
        .unwrap()
        .call(qwen_recurrent_normalize::Args {
            hidden: &hidden,
            input_norm: &input_norm,
            epsilon: 1.0e-5,
        })
        .unwrap()
        .value;
        let projection = qwen_recurrent_project::native_for_device_with(
            &device,
            qwen_recurrent_project::Elements {
                QW: f32,
                GW: f32,
                AW: f32,
                BW: f32,
                A: f32,
            }, &seismic::NativeSpecialization::new(),
        )
        .unwrap()
        .call(qwen_recurrent_project::Args {
            normalized: &normalized,
            qkv_weight: &qkv_weight,
            gate_weight: &gate_weight,
            alpha_weight: &alpha_weight,
            beta_weight: &beta_weight,
        })
        .unwrap()
        .value;
        let prepared = qwen_recurrent_prepare::native_for_device_with(
            &device,
            qwen_recurrent_prepare::Elements { RN: f32, A: f32 }, &seismic::NativeSpecialization::new(),
        )
        .unwrap()
        .call(qwen_recurrent_prepare::Args {
            projection: &projection,
            convolution: &convolution,
            rate: &rate,
            time_bias: &time_bias,
            recurrent_norm: &recurrent_norm,
            segments: &segments,
            window: &window,
            preparation_epsilon: 1.0e-5,
        })
        .unwrap();
        let scanned = qwen_recurrent_scan::native_for_device_with(
            &device,
            qwen_recurrent_scan::Elements { A: f32 }, &seismic::NativeSpecialization::new(),
        )
            .unwrap()
            .call(qwen_recurrent_scan::Args {
                prepared: &prepared.r1,
                decay: &prepared.r2,
                segments: &segments,
                delta: &delta,
                grouped: true,
            })
            .unwrap();
        let gated = qwen_recurrent_mix::native_for_device_with(
            &device,
            qwen_recurrent_mix::Elements { RN: f32, A: f32 }, &seismic::NativeSpecialization::new(),
        )
        .unwrap()
        .call(qwen_recurrent_mix::Args {
            projection: &projection,
            mixed: &scanned.r1,
            recurrent_norm: &recurrent_norm,
            epsilon: 1.0e-5,
        })
        .unwrap()
        .value;
        let result = qwen_recurrent_output::native_for_device_with(
            &device,
            qwen_recurrent_output::Elements { OW: f32, A: f32 }, &seismic::NativeSpecialization::new(),
        )
        .unwrap()
        .call(qwen_recurrent_output::Args {
            hidden: &hidden,
            gated: &gated,
            output_weight: &output_weight,
        })
        .unwrap()
        .value;
        // Captured from the original monolithic checked/native entry before its
        // removal. These fixtures exercise time ordering and independent slots.
        let (window_golden, delta_golden, result_golden): (&[f32], &[f32], &[f32]) = match case {
            "two_rows_one_slot" => (
                &[
                    -0.065185,
                    -0.15407366,
                    -0.0044444306,
                    0.13481446,
                    0.056296144,
                    -0.08444421,
                ],
                &[0.014485664, -0.043550566, 0.05112263, 0.039182536],
                &[
                    0.90106714, -1.9262991, 2.949536, 0.59893286, 0.22543305, 1.4117786,
                    -1.1255766, 1.9745669,
                ],
            ),
            "one_row_per_slot" => (
                &[
                    -0.05721972,
                    0.12927417,
                    0.0042384993,
                    -0.2988141,
                    0.06569672,
                    0.029669486,
                    -0.065185,
                    -0.15407366,
                    -0.0044444306,
                    0.13481446,
                    0.056296144,
                    -0.08444421,
                ],
                &[
                    0.019486044,
                    -0.075740226,
                    0.07186916,
                    0.06572713,
                    0.009604574,
                    0.06030717,
                    -0.020504165,
                    0.042691804,
                ],
                &[
                    0.90106714, -1.9262991, 2.949536, 0.59893286, 0.24308623, 1.4485005,
                    -1.0168264, 1.9569137,
                ],
            ),
            _ => unreachable!(),
        };
        close(&prepared.r0, window_golden, &format!("{case} next_window"));
        close(&scanned.r0, delta_golden, &format!("{case} next_delta"));
        close(&result, result_golden, &format!("{case} result"));
    }
}
