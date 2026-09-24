use magnitude_model_kernels::{
    qwen_recurrent_chunk, qwen_recurrent_output, qwen_recurrent_project,
    qwen_recurrent_step, repack_weight,
};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
#[cfg(target_os = "macos")]
fn packed_stages_accept_repacked_q8_weights_and_bf16_activations() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let q8_external = seismic::Element::named("gguf_q8_0").unwrap();
    let q8 = seismic::Element::stored("q8g32s", seismic::Layout::Rows16).unwrap();
    let bf16 = seismic::Element::bf16();
    // One key and one value head of width 32 over a 32-wide hidden row, at
    // each entry's first declared configuration.
    fn declared_defaults<E: seismic::Entry>(device: &seismic::Device) -> seismic::NativeSpecialization {
        let statics = seismic::NativeSpecialization::new()
            .with_static("H", 32)
            .with_static("NK", 1)
            .with_static("NV", 1)
            .with_static("W", 32);
        seismic::generated::native_implementation::<E>(device)
            .unwrap()
            .unwrap()
            .default_specialization(&statics)
            .unwrap()
    }
    let resident = |shape: &[u64]| {
        let values = shape.iter().product::<u64>() as usize;
        assert_eq!(values % 32, 0);
        let mut bytes = Vec::with_capacity(values / 32 * 34);
        for _ in 0..values / 32 {
            bytes.extend_from_slice(&seismic_lang::registry::f16_bits(1.0).to_le_bytes());
            bytes.extend_from_slice(&[0; 32]);
        }
        let external =
            seismic::Tensor::from_host(&device, q8_external, &[1, shape[0], shape[1]], &bytes).unwrap();
        repack_weight::native_for_device_with(
            &device,
            repack_weight::Elements {
                E: q8_external,
                U: q8,
            }, &seismic::NativeSpecialization::new(),
        )
        .unwrap()
        .call(repack_weight::Args { source: &external })
        .unwrap()
        .value
        .reshape(shape)
        .unwrap()
    };
    let bf16_tensor = |shape: &[u64], values: &[f32]| {
        let bytes = values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        seismic::Tensor::from_host(&device, bf16, shape, &bytes).unwrap()
    };
    let epsilon = 1e-5;
    let hidden_values = (0..32)
        .map(|index| index as f32 / 32.0 + 0.25)
        .collect::<Vec<_>>();
    let hidden = tensor(&device, &[1, 32], &hidden_values);
    let norm = bf16_tensor(&[32], &vec![1.0; 32]);
    let zero_weight = resident(&[32, 32]);


    // One key and one value head of width 32, a two-tap convolution, and a
    // two-bank state arena (bank 0 read, bank 1 published) with one tape row
    // (u [1, 32] | k [1, 32] | d [1]).
    let segments = indices(&device, &[2, 2], &[0, 1, 1, 1]);
    let stop = indices(&device, &[1], &[1]);
    let previous_bank = indices(&device, &[1], &[0]);
    let previous_tape = indices(&device, &[1], &[0]);
    let following_bank = indices(&device, &[1], &[1]);
    let mut window = bf16_tensor(&[2, 2, 96], &[0.0; 384]);
    let mut tape = tensor(&device, &[2, 1, 65], &[0.0; 130]);
    let qkv_weight = resident(&[96, 32]);
    let alpha_weight = resident(&[1, 32]);
    let beta_weight = resident(&[1, 32]);
    let convolution = tensor(&device, &[96, 2], &[0.0; 192]);
    let rate = tensor(&device, &[1], &[0.0]);
    let time_bias = tensor(&device, &[1], &[0.0]);
    let mut delta = tensor(&device, &[2, 1, 32, 32], &[0.0; 2048]);
    let projection = qwen_recurrent_project::native_for_device_with(
        &device,
        qwen_recurrent_project::Elements {
            NW: bf16,
            QW: q8,
            GW: q8,
            AW: q8,
            BW: q8,
            A: bf16,
        },
        &declared_defaults::<qwen_recurrent_project::Entry>(&device),
    )
    .unwrap()
    .call(qwen_recurrent_project::Args {
        hidden: &hidden,
        input_norm: &norm,
        qkv_weight: &qkv_weight,
        gate_weight: &zero_weight,
        alpha_weight: &alpha_weight,
        beta_weight: &beta_weight,
        epsilon,
    })
    .unwrap()
    .value;
    let mixed = qwen_recurrent_step::native_for_device_with(
        &device,
        qwen_recurrent_step::Elements { A: bf16 },
        &seismic::NativeSpecialization::new()
            .with_static("NK", 1)
            .with_static("NV", 1)
            .with_static("W", 32)
            .with_static("C", 2)
            .with_param("ROWS", 16),
    )
    .unwrap()
    .call(qwen_recurrent_step::Args {
        projection: &projection,
        convolution: &convolution,
        rate: &rate,
        time_bias: &time_bias,
        segments: &segments,
        stop: &stop,
        previous_bank: &previous_bank,
        previous_tape: &previous_tape,
        following_bank: &following_bank,
        window: &mut window,
        delta: &mut delta,
        tape: &mut tape,
        norm_epsilon: epsilon,
        grouped: true,
    })
    .unwrap()
    .value;
    let recurrent = qwen_recurrent_output::native_for_device_with(
        &device,
        qwen_recurrent_output::Elements {
            RN: bf16,
            OW: q8,
            A: bf16,
        },
        &declared_defaults::<qwen_recurrent_output::Entry>(&device),
    )
    .unwrap()
    .call(qwen_recurrent_output::Args {
        hidden: &hidden,
        mixed: &mixed,
        projection: &projection,
        recurrent_norm: &norm,
        output_weight: &zero_weight,
        epsilon,
    })
    .unwrap()
    .value;
    assert_close(&read_f32(&recurrent), &hidden_values);

}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn tensor(device: &seismic::Device, shape: &[u64], values: &[f32]) -> seismic::Tensor {
    seismic::Tensor::from_host(device, seismic::Element::f32(), shape, &f32_bytes(values)).unwrap()
}

fn indices(device: &seismic::Device, shape: &[u64], values: &[i32]) -> seismic::Tensor {
    seismic::Tensor::from_host(device, seismic::Element::i32(), shape, &i32_bytes(values)).unwrap()
}

fn read_f32(tensor: &seismic::Tensor) -> Vec<f32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 3e-5,
            "{index}: {actual} != {expected}"
        );
    }
}

#[test]
fn packed_entries_are_the_only_generated_target_surface() {
    let _ = qwen_recurrent_project::for_device_with;
    let _ = qwen_recurrent_project::native_for_device_with;
    let _ = qwen_recurrent_step::for_device_with;
    let _ = qwen_recurrent_step::native_for_device_with;
    let _ = qwen_recurrent_chunk::for_device_with;
    let _ = qwen_recurrent_chunk::native_for_device_with;
    let _ = qwen_recurrent_output::for_device_with;
    let _ = qwen_recurrent_output::native_for_device_with;
    let sources = [
        include_str!("../kernels/target.seismic"),
        include_str!("../kernels/dense_rows.seismic"),
        include_str!("../kernels/routed.seismic"),
    ]
    .concat();
    for obsolete in [
        "qwen_append_rows",
        "qwen_attention_sequence",
        "qwen_recurrent_sequence",
        "qwen_dense_suffix",
        "qwen_routed_suffix",
        "fn route_topk",
        "fn routed_input",
        "fn routed_output",
    ] {
        assert!(
            !sources.contains(obsolete),
            "obsolete generated entry {obsolete}"
        );
    }
}
