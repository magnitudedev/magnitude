use magnitude_model_kernels::{
    qwen_attention_attend, qwen_attention_normalize, qwen_attention_output, qwen_attention_prepare,
    qwen_attention_project, qwen_attention_rows, qwen_recurrent_mix, qwen_recurrent_normalize,
    qwen_recurrent_output, qwen_recurrent_prepare, qwen_recurrent_project, qwen_recurrent_scan,
    qwen_routed_expand, qwen_routed_logits, qwen_routed_normalize, qwen_routed_output,
    qwen_routed_rows, qwen_routed_select, repack_weight,
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
    let q8 = seismic::Element::named("q8g32s").unwrap();
    let bf16 = seismic::Element::bf16();
    let resident = |shape: &[u64]| {
        let values = shape.iter().product::<u64>() as usize;
        assert_eq!(values % 32, 0);
        let mut bytes = Vec::with_capacity(values / 32 * 34);
        for _ in 0..values / 32 {
            bytes.extend_from_slice(&seismic_lang::registry::f16_bits(1.0).to_le_bytes());
            bytes.extend_from_slice(&[0; 32]);
        }
        let external =
            seismic::Tensor::from_host(&device, q8_external, &[values as u64], &bytes).unwrap();
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

    let query_weight = resident(&[64, 32]);
    let coords = indices(&device, &[1, 4], &[0; 4]);
    let components = indices(&device, &[1], &[0]);
    let visible = indices(&device, &[1, 1, 2], &[0, 0]);
    let fresh = indices(&device, &[1, 2], &[0, 1]);
    let destinations = indices(&device, &[1], &[0]);
    let mut history_key = bf16_tensor(&[1, 1, 32], &[0.0; 32]);
    let mut history_value = bf16_tensor(&[1, 1, 32], &[0.0; 32]);
    let attention = qwen_attention_rows::native_for_device_with(
        &device,
        qwen_attention_rows::Elements {
            NW: bf16,
            QW: q8,
            KW: q8,
            VW: q8,
            OW: q8,
            A: bf16,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_attention_rows::Args {
        hidden: &hidden,
        input_norm: &norm,
        query_gate_weight: &query_weight,
        key_weight: &zero_weight,
        value_weight: &zero_weight,
        query_norm: &tensor(&device, &[32], &[1.0; 32]),
        key_norm: &tensor(&device, &[32], &[1.0; 32]),
        output_weight: &zero_weight,
        coordinates: &coords,
        rotary_components: &components,
        visible: &visible,
        fresh: &fresh,
        destinations: &destinations,
        history_key: &mut history_key,
        history_value: &mut history_value,
        base: 10_000.0,
        epsilon,
        scale: 32.0f32.sqrt().recip(),
    })
    .unwrap()
    .value;
    assert_close(&read_f32(&attention), &hidden_values);

    let segments = indices(&device, &[2, 2], &[0, 1, 1, 1]);
    let window = bf16_tensor(&[1, 1, 96], &[0.0; 96]);
    let qkv_weight = resident(&[96, 32]);
    let alpha_weight = resident(&[1, 32]);
    let beta_weight = resident(&[1, 32]);
    let convolution = tensor(&device, &[96, 2], &[0.0; 192]);
    let rate = tensor(&device, &[1], &[0.0]);
    let time_bias = tensor(&device, &[1], &[0.0]);
    let delta = tensor(&device, &[1, 1, 32, 32], &[0.0; 1024]);
    let normalized = qwen_recurrent_normalize::native_for_device_with(
        &device,
        qwen_recurrent_normalize::Elements { NW: bf16, A: bf16 }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_normalize::Args {
        hidden: &hidden,
        input_norm: &norm,
        epsilon,
    })
    .unwrap()
    .value;
    let projection = qwen_recurrent_project::native_for_device_with(
        &device,
        qwen_recurrent_project::Elements {
            QW: q8,
            GW: q8,
            AW: q8,
            BW: q8,
            A: bf16,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_project::Args {
        normalized: &normalized,
        qkv_weight: &qkv_weight,
        gate_weight: &zero_weight,
        alpha_weight: &alpha_weight,
        beta_weight: &beta_weight,
    })
    .unwrap()
    .value;
    let prepared = qwen_recurrent_prepare::native_for_device_with(
        &device,
        qwen_recurrent_prepare::Elements { RN: bf16, A: bf16 }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_prepare::Args {
        projection: &projection,
        convolution: &convolution,
        rate: &rate,
        time_bias: &time_bias,
        recurrent_norm: &norm,
        segments: &segments,
        window: &window,
        preparation_epsilon: epsilon,
    })
    .unwrap();
    let scanned = qwen_recurrent_scan::native_for_device_with(
        &device,
        qwen_recurrent_scan::Elements { A: bf16 }, &seismic::NativeSpecialization::new(),
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
        qwen_recurrent_mix::Elements { RN: bf16, A: bf16 }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_mix::Args {
        projection: &projection,
        mixed: &scanned.r1,
        recurrent_norm: &norm,
        epsilon,
    })
    .unwrap()
    .value;
    let recurrent = qwen_recurrent_output::native_for_device_with(
        &device,
        qwen_recurrent_output::Elements { OW: q8, A: bf16 }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_output::Args {
        hidden: &hidden,
        gated: &gated,
        output_weight: &zero_weight,
    })
    .unwrap()
    .value;
    assert_close(&read_f32(&recurrent), &hidden_values);

    let identity_rows = indices(&device, &[1], &[0]);
    let normalized = qwen_routed_normalize::native_for_device_with(
        &device,
        qwen_routed_normalize::Elements { NW: bf16, A: bf16 }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_normalize::Args {
        residual: &hidden,
        norm: &norm,
        source_rows: &identity_rows,
        eps: epsilon,
    })
    .unwrap()
    .value;
    let logits = qwen_routed_logits::native_for_device_with(
        &device,
        qwen_routed_logits::Elements { A: bf16, RW: q8 }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_logits::Args {
        normalized: &normalized,
        router_weight: &resident(&[1, 32]),
    })
    .unwrap()
    .value;
    let mut selected_routes = indices(&device, &[1, 1], &[0]);
    let mut selected_scores = tensor(&device, &[1, 1], &[0.0]);
    qwen_routed_select::native_for_device(&device, &seismic::NativeSpecialization::new())
        .unwrap()
        .call(qwen_routed_select::Args {
            logits: &logits,
            selected: 1,
            routes: &mut selected_routes,
            scores: &mut selected_scores,
        })
        .unwrap();
    let expanded = qwen_routed_expand::native_for_device_with(
        &device,
        qwen_routed_expand::Elements {
            A: bf16,
            EGW: q8,
            EUW: q8,
            SGW: q8,
            SUW: q8,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_expand::Args {
        normalized: &normalized,
        expert_gate: &resident(&[1, 32, 32]),
        expert_up: &resident(&[1, 32, 32]),
        shared_gate: &resident(&[32, 32]),
        shared_up: &resident(&[32, 32]),
        shared_control: &tensor(&device, &[32], &[0.0; 32]),
        routes: &selected_routes,
    })
    .unwrap();
    let routed = qwen_routed_output::native_for_device_with(
        &device,
        qwen_routed_output::Elements {
            A: bf16,
            EDW: q8,
            SDW: q8,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_output::Args {
        residual: &hidden,
        source_rows: &identity_rows,
        expert_product: &expanded.r0,
        shared_product: &expanded.r1,
        shared_coefficient: &expanded.r2,
        expert_down: &resident(&[1, 32, 32]),
        shared_down: &resident(&[32, 32]),
        routes: &selected_routes,
        scores: &selected_scores,
    })
    .unwrap()
    .value;
    assert_close(&read_f32(&routed), &hidden_values);

    let mut demanded_values = vec![-3.0; 32];
    demanded_values.extend_from_slice(&hidden_values);
    let demanded_source = tensor(&device, &[2, 32], &demanded_values);
    let out_rows = indices(&device, &[1], &[1]);
    let demanded_normalized = qwen_routed_normalize::native_for_device_with(
        &device,
        qwen_routed_normalize::Elements { NW: bf16, A: bf16 }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_normalize::Args {
        residual: &demanded_source,
        norm: &norm,
        source_rows: &out_rows,
        eps: epsilon,
    })
    .unwrap()
    .value;
    let demanded_logits = qwen_routed_logits::native_for_device_with(
        &device,
        qwen_routed_logits::Elements { A: bf16, RW: q8 }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_logits::Args {
        normalized: &demanded_normalized,
        router_weight: &resident(&[1, 32]),
    })
    .unwrap()
    .value;
    let mut demanded_routes = indices(&device, &[1, 1], &[0]);
    let mut demanded_scores = tensor(&device, &[1, 1], &[0.0]);
    qwen_routed_select::native_for_device(&device, &seismic::NativeSpecialization::new())
        .unwrap()
        .call(qwen_routed_select::Args {
            logits: &demanded_logits,
            selected: 1,
            routes: &mut demanded_routes,
            scores: &mut demanded_scores,
        })
        .unwrap();
    let demanded_expanded = qwen_routed_expand::native_for_device_with(
        &device,
        qwen_routed_expand::Elements {
            A: bf16,
            EGW: q8,
            EUW: q8,
            SGW: q8,
            SUW: q8,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_expand::Args {
        normalized: &demanded_normalized,
        expert_gate: &resident(&[1, 32, 32]),
        expert_up: &resident(&[1, 32, 32]),
        shared_gate: &resident(&[32, 32]),
        shared_up: &resident(&[32, 32]),
        shared_control: &tensor(&device, &[32], &[0.0; 32]),
        routes: &demanded_routes,
    })
    .unwrap();
    let demanded = qwen_routed_output::native_for_device_with(
        &device,
        qwen_routed_output::Elements {
            A: bf16,
            EDW: q8,
            SDW: q8,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_output::Args {
        residual: &demanded_source,
        source_rows: &out_rows,
        expert_product: &demanded_expanded.r0,
        shared_product: &demanded_expanded.r1,
        shared_coefficient: &demanded_expanded.r2,
        expert_down: &resident(&[1, 32, 32]),
        shared_down: &resident(&[32, 32]),
        routes: &demanded_routes,
        scores: &demanded_scores,
    })
    .unwrap()
    .value;
    assert_close(&read_f32(&demanded), &hidden_values);
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

fn read_i32(tensor: &seismic::Tensor) -> Vec<i32> {
    tensor
        .read_to_host()
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| i32::from_le_bytes(bytes.try_into().unwrap()))
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

fn f32_elements() -> seismic::Element {
    seismic::Element::f32()
}

#[test]
fn packed_entries_are_the_only_generated_target_surface() {
    let _ = qwen_attention_rows::for_device_with;
    let _ = qwen_attention_rows::native_for_device_with;
    let _ = qwen_recurrent_project::for_device_with;
    let _ = qwen_recurrent_project::native_for_device_with;
    let _ = qwen_recurrent_prepare::for_device_with;
    let _ = qwen_recurrent_prepare::native_for_device_with;
    let _ = qwen_recurrent_scan::for_device_with;
    let _ = qwen_recurrent_scan::native_for_device_with;
    let _ = qwen_recurrent_mix::for_device_with;
    let _ = qwen_recurrent_mix::native_for_device_with;
    let _ = qwen_recurrent_output::for_device_with;
    let _ = qwen_recurrent_output::native_for_device_with;
    let _ = qwen_routed_rows::for_device_with;
    let _ = qwen_routed_normalize::native_for_device_with;
    let _ = qwen_routed_logits::native_for_device_with;
    let _ = qwen_routed_select::native_for_device;
    let _ = qwen_routed_expand::native_for_device_with;
    let _ = qwen_routed_output::native_for_device_with;
    let sources = [
        include_str!("../kernels/target_rows.seismic"),
        include_str!("../kernels/dense_rows.seismic"),
        include_str!("../kernels/routed_rows.seismic"),
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

#[test]
#[cfg(target_os = "macos")]
fn staged_attention_matches_monolith_for_gqa_rotary_extremes_and_overlap() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let f32e = f32_elements();
    let tensor_values = |shape: &[u64], values: Vec<f32>| tensor(&device, shape, &values);
    let hidden = tensor_values(&[2, 4], vec![0.7, -1.2, 0.3, 1.8, -0.9, 0.4, 1.5, -0.2]);
    let norm = tensor_values(&[4], vec![1.0, 0.8, 1.1, 0.9]);
    let patterned = |count: usize, scale: f32| {
        (0..count)
            .map(|i| (((i * 17 + 5) % 23) as f32 - 11.0) * scale)
            .collect::<Vec<_>>()
    };
    let query_gate = tensor_values(&[12, 4], patterned(48, 0.13));
    let key_weight = tensor_values(&[3, 4], patterned(12, 0.19));
    let value_weight = tensor_values(&[3, 4], patterned(12, 0.11));
    let output_weight = tensor_values(&[4, 6], patterned(24, 0.09));
    let head_norm = tensor_values(&[3], vec![1.0, 0.75, 1.25]);
    let coordinates = indices(&device, &[2, 4], &[3, 0, 0, 0, 7, 0, 0, 0]);
    let rotary = indices(&device, &[1], &[0]);
    let visible = indices(&device, &[2, 2, 2], &[0, 1, 0, 0, 0, 1, 0, 0]);
    let fresh = indices(&device, &[2, 2], &[0, 1, 0, 2]);
    let destinations = indices(&device, &[2], &[0, 1]);
    let old_key_values = vec![0.6, -0.4, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let old_value_values = vec![1.2, -0.7, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut oracle_key = tensor_values(&[3, 1, 3], old_key_values.clone());
    let mut oracle_value = tensor_values(&[3, 1, 3], old_value_values.clone());
    let oracle = qwen_attention_rows::native_for_device_with(
        &device,
        qwen_attention_rows::Elements {
            NW: f32e,
            QW: f32e,
            KW: f32e,
            VW: f32e,
            OW: f32e,
            A: f32e,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_attention_rows::Args {
        hidden: &hidden,
        input_norm: &norm,
        query_gate_weight: &query_gate,
        key_weight: &key_weight,
        value_weight: &value_weight,
        query_norm: &head_norm,
        key_norm: &head_norm,
        output_weight: &output_weight,
        coordinates: &coordinates,
        rotary_components: &rotary,
        visible: &visible,
        fresh: &fresh,
        destinations: &destinations,
        history_key: &mut oracle_key,
        history_value: &mut oracle_value,
        base: 10_000.0,
        epsilon: 1e-5,
        scale: 20.0,
    })
    .unwrap()
    .value;
    let normalized = qwen_attention_normalize::native_for_device_with(
        &device,
        qwen_attention_normalize::Elements { NW: f32e, A: f32e }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_attention_normalize::Args {
        hidden: &hidden,
        input_norm: &norm,
        epsilon: 1e-5,
    })
    .unwrap()
    .value;
    let projected = qwen_attention_project::native_for_device_with(
        &device,
        qwen_attention_project::Elements {
            QW: f32e,
            KW: f32e,
            VW: f32e,
            A: f32e,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_attention_project::Args {
        normalized: &normalized,
        query_norm: &head_norm,
        query_gate_weight: &query_gate,
        key_weight: &key_weight,
        value_weight: &value_weight,
    })
    .unwrap();
    let prepared = qwen_attention_prepare::native_for_device_with(
        &device,
        qwen_attention_prepare::Elements { A: f32e }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_attention_prepare::Args {
        query_gate: &projected.r0,
        key: &projected.r1,
        query_norm: &head_norm,
        key_norm: &head_norm,
        coordinates: &coordinates,
        rotary_components: &rotary,
        base: 10_000.0,
        epsilon: 1e-5,
    })
    .unwrap();
    let mut staged_key = tensor_values(&[3, 1, 3], old_key_values);
    let mut staged_value = tensor_values(&[3, 1, 3], old_value_values);
    let mut accumulator = tensor_values(&[2, 2, 3], vec![0.0; 12]);
    let gated = qwen_attention_attend::native_for_device_with(
        &device,
        qwen_attention_attend::Elements { A: f32e }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_attention_attend::Args {
        query: &prepared.r0,
        prepared_key: &prepared.r1,
        value: &projected.r2,
        gate: &prepared.r2,
        visible: &visible,
        fresh: &fresh,
        history_key: &staged_key,
        history_value: &staged_value,
        accumulator: &mut accumulator,
        scale: 20.0,
    })
    .unwrap()
    .value;
    let staged = qwen_attention_output::native_for_device_with(
        &device,
        qwen_attention_output::Elements { OW: f32e, A: f32e }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_attention_output::Args {
        hidden: &hidden,
        gated: &gated,
        prepared_key: &prepared.r1,
        value: &projected.r2,
        output_weight: &output_weight,
        destinations: &destinations,
        history_key: &mut staged_key,
        history_value: &mut staged_value,
    })
    .unwrap()
    .value;
    for (actual, expected) in read_f32(&staged).iter().zip(read_f32(&oracle)) {
        assert!((actual - expected).abs() <= 2e-4, "{actual} != {expected}");
    }
    assert_close(&read_f32(&staged_key), &read_f32(&oracle_key));
    assert_close(&read_f32(&staged_value), &read_f32(&oracle_value));
}

#[test]
#[cfg(target_os = "macos")]
fn packed_native_rows_match_small_host_oracles_at_two_partitions() {
    let catalog = seismic::DeviceCatalog::discover().unwrap();
    let device = catalog.open_backend(seismic::BackendName::Metal).unwrap();
    let f32e = f32_elements();
    let epsilon = 1e-5f32;

    // Attention: zero queries/keys make each declared fresh interval uniform;
    // identity value/output projections leave a compact exact host oracle.
    // Keep every symbolic extent positive: P = 1 and S = 1 make the head
    // width three. Zero-valued symbolic extents are outside Seismic's target
    // domain even when the corresponding loop would be empty.
    let hidden_values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let hidden = tensor(&device, &[2, 3], &hidden_values);
    let ones3 = tensor(&device, &[3], &[1.0, 1.0, 1.0]);
    let query_gate = tensor(&device, &[6, 3], &[0.0; 18]);
    let key_weight = tensor(&device, &[3, 3], &[0.0; 9]);
    let identity = tensor(
        &device,
        &[3, 3],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    );
    let coordinates = indices(&device, &[2, 4], &[0; 8]);
    let components = indices(&device, &[1], &[0]);
    let visible = indices(&device, &[2, 1, 2], &[0, 0, 0, 0]);
    let fresh = indices(&device, &[2, 2], &[0, 1, 0, 1]);
    let destinations = indices(&device, &[2], &[0, 1]);
    let mut history_key = tensor(&device, &[3, 1, 3], &[0.0; 9]);
    let mut history_value = tensor(&device, &[3, 1, 3], &[0.0; 9]);
    let attention = qwen_attention_rows::native_for_device_with(
        &device,
        qwen_attention_rows::Elements {
            NW: f32e,
            QW: f32e,
            KW: f32e,
            VW: f32e,
            OW: f32e,
            A: f32e,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_attention_rows::Args {
        hidden: &hidden,
        input_norm: &ones3,
        query_gate_weight: &query_gate,
        key_weight: &key_weight,
        value_weight: &identity,
        query_norm: &ones3,
        key_norm: &ones3,
        output_weight: &identity,
        coordinates: &coordinates,
        rotary_components: &components,
        visible: &visible,
        fresh: &fresh,
        destinations: &destinations,
        history_key: &mut history_key,
        history_value: &mut history_value,
        base: 10_000.0,
        epsilon,
        scale: 2.0f32.sqrt().recip(),
    })
    .unwrap()
    .value;
    let normalized = hidden_values
        .chunks_exact(3)
        .flat_map(|row| {
            let inverse = ((row[0] * row[0] + row[1] * row[1] + row[2] * row[2]) / 3.0 + epsilon)
                .sqrt()
                .recip();
            [row[0] * inverse, row[1] * inverse, row[2] * inverse]
        })
        .collect::<Vec<_>>();
    let expected_attention = vec![
        1.0 + 0.5 * normalized[0],
        2.0 + 0.5 * normalized[1],
        3.0 + 0.5 * normalized[2],
        4.0 + 0.5 * normalized[0],
        5.0 + 0.5 * normalized[1],
        6.0 + 0.5 * normalized[2],
    ];
    assert_close(&read_f32(&attention), &expected_attention);
    assert_close(&read_f32(&history_key), &[0.0; 9]);
    assert_close(&read_f32(&history_value)[0..6], &normalized);

    // Recurrent: two slots advance independently. Zero projected keys preserve
    // each accepted bank exactly, including the final bank of both slots.
    let segments = indices(&device, &[3, 2], &[0, 1, 1, 2, 2, 2]);
    let qkv_weight = tensor(&device, &[5, 3], &[0.0; 15]);
    let gate_weight = tensor(&device, &[1, 3], &[0.0; 3]);
    let head_weight = tensor(&device, &[1, 3], &[0.0; 3]);
    let convolution = tensor(&device, &[5, 2], &[0.0; 10]);
    let scalar_zero = tensor(&device, &[1], &[0.0]);
    let scalar_one = tensor(&device, &[1], &[1.0]);
    let output_weight = tensor(&device, &[3, 1], &[0.0; 3]);
    let window = tensor(
        &device,
        &[2, 1, 5],
        &[7.0, 8.0, 9.0, 10.0, 11.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let delta_values = [2.0f32, 3.0];
    let delta = tensor(&device, &[2, 1, 1, 1], &delta_values);
    let normalized = qwen_recurrent_normalize::native_for_device_with(
        &device,
        qwen_recurrent_normalize::Elements { NW: f32e, A: f32e }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_normalize::Args {
        hidden: &hidden,
        input_norm: &ones3,
        epsilon,
    })
    .unwrap()
    .value;
    let projection = qwen_recurrent_project::native_for_device_with(
        &device,
        qwen_recurrent_project::Elements {
            QW: f32e,
            GW: f32e,
            AW: f32e,
            BW: f32e,
            A: f32e,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_project::Args {
        normalized: &normalized,
        qkv_weight: &qkv_weight,
        gate_weight: &gate_weight,
        alpha_weight: &head_weight,
        beta_weight: &head_weight,
    })
    .unwrap()
    .value;
    let prepared = qwen_recurrent_prepare::native_for_device_with(
        &device,
        qwen_recurrent_prepare::Elements { RN: f32e, A: f32e }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_prepare::Args {
        projection: &projection,
        convolution: &convolution,
        rate: &scalar_zero,
        time_bias: &scalar_zero,
        recurrent_norm: &scalar_one,
        segments: &segments,
        window: &window,
        preparation_epsilon: epsilon,
    })
    .unwrap();
    let scanned = qwen_recurrent_scan::native_for_device_with(
        &device,
        qwen_recurrent_scan::Elements { A: f32e }, &seismic::NativeSpecialization::new(),
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
        qwen_recurrent_mix::Elements { RN: f32e, A: f32e }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_mix::Args {
        projection: &projection,
        mixed: &scanned.r1,
        recurrent_norm: &scalar_one,
        epsilon,
    })
    .unwrap()
    .value;
    let recurrent = qwen_recurrent_output::native_for_device_with(
        &device,
        qwen_recurrent_output::Elements { OW: f32e, A: f32e }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_recurrent_output::Args {
        hidden: &hidden,
        gated: &gated,
        output_weight: &output_weight,
    })
    .unwrap()
    .value;
    assert_close(&read_f32(&prepared.r0), &[0.0; 10]);
    assert_close(&read_f32(&scanned.r0), &delta_values);
    assert_close(&read_f32(&recurrent), &hidden_values);

    // Routed: unequal router logits exercise top-k order, normalization,
    // selected-expert SwiGLU accumulation, and shared-expert gating.
    let routed_hidden = tensor(&device, &[2, 1], &[1.0, 2.0]);
    let norm1 = tensor(&device, &[1], &[1.0]);
    let router = tensor(&device, &[3, 1], &[0.0, 1.0, 2.0]);
    let shared_router = tensor(&device, &[1], &[0.5]);
    let expert_gate = tensor(&device, &[3, 1, 1], &[0.5, 1.0, 1.5]);
    let expert_up = tensor(&device, &[3, 1, 1], &[1.0, 1.0, 1.0]);
    let expert_down = tensor(&device, &[3, 1, 1], &[1.0, 2.0, 3.0]);
    let shared_gate = tensor(&device, &[1, 1], &[0.25]);
    let shared_up = tensor(&device, &[1, 1], &[0.5]);
    let shared_down = tensor(&device, &[1, 1], &[2.0]);
    let source_rows = indices(&device, &[2], &[0, 1]);
    let normalized = qwen_routed_normalize::native_for_device_with(
        &device,
        qwen_routed_normalize::Elements { NW: f32e, A: f32e }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_normalize::Args {
        residual: &routed_hidden,
        norm: &norm1,
        source_rows: &source_rows,
        eps: epsilon,
    })
    .unwrap()
    .value;
    let logits = qwen_routed_logits::native_for_device_with(
        &device,
        qwen_routed_logits::Elements { A: f32e, RW: f32e }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_logits::Args {
        normalized: &normalized,
        router_weight: &router,
    })
    .unwrap()
    .value;
    let mut selected_routes = indices(&device, &[2, 2], &[0; 4]);
    let mut selected_scores = tensor(&device, &[2, 2], &[0.0; 4]);
    qwen_routed_select::native_for_device(&device, &seismic::NativeSpecialization::new())
        .unwrap()
        .call(qwen_routed_select::Args {
            logits: &logits,
            selected: 1,
            routes: &mut selected_routes,
            scores: &mut selected_scores,
        })
        .unwrap();
    let expanded = qwen_routed_expand::native_for_device_with(
        &device,
        qwen_routed_expand::Elements {
            A: f32e,
            EGW: f32e,
            EUW: f32e,
            SGW: f32e,
            SUW: f32e,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_expand::Args {
        normalized: &normalized,
        expert_gate: &expert_gate,
        expert_up: &expert_up,
        shared_gate: &shared_gate,
        shared_up: &shared_up,
        shared_control: &shared_router,
        routes: &selected_routes,
    })
    .unwrap();
    let routed = qwen_routed_output::native_for_device_with(
        &device,
        qwen_routed_output::Elements {
            A: f32e,
            EDW: f32e,
            SDW: f32e,
        }, &seismic::NativeSpecialization::new(),
    )
    .unwrap()
    .call(qwen_routed_output::Args {
        residual: &routed_hidden,
        source_rows: &source_rows,
        expert_product: &expanded.r0,
        shared_product: &expanded.r1,
        shared_coefficient: &expanded.r2,
        expert_down: &expert_down,
        shared_down: &shared_down,
        routes: &selected_routes,
        scores: &selected_scores,
    })
    .unwrap()
    .value;
    assert_eq!(read_i32(&selected_routes), vec![1, 2, 1, 2]);
    let probability = |logit: f32| logit.exp() / (1.0f32 + 1.0f32.exp() + 2.0f32.exp());
    let denominator = probability(1.0) + probability(2.0);
    let expected_scores = [
        probability(1.0) / denominator,
        probability(2.0) / denominator,
    ];
    assert_close(
        &read_f32(&selected_scores),
        &[
            expected_scores[0],
            expected_scores[1],
            expected_scores[0],
            expected_scores[1],
        ],
    );
    let unit = (1.0 + epsilon).sqrt().recip();
    let expert = |gate: f32, down: f32| (gate * unit / (1.0 + (-gate * unit).exp())) * unit * down;
    let selected = expected_scores[0] * expert(1.0, 2.0) + expected_scores[1] * expert(1.5, 3.0);
    let shared = (0.25 * unit / (1.0 + (-0.25 * unit).exp()))
        * (0.5 * unit)
        * 2.0
        * (1.0 / (1.0 + (-0.5 * unit).exp()));
    assert_close(
        &read_f32(&routed),
        &[1.0 + selected + shared, 2.0 + selected + shared],
    );
}
