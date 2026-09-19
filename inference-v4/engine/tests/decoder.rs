use seismic_engine::{
    models::qwen35::{decoder::Decoder, *},
    weights::{
        descriptor::{ArtifactIdentity, Stored, StoredTensor, Transform, WeightDescriptor},
        residency::Importer,
        source::FileSource,
    },
};
use seismic_lang::types::DType;
use seismic_runtime::{Device, plan::Settings};
use serde_json::Value;
use std::{collections::HashMap, rc::Rc, sync::Arc};

fn assert_reference_logits(actual: &[f32], reference: &Value, stage: &str) {
    let expected = reference["logits"].as_array().unwrap();
    assert_eq!(actual.len(), expected.len(), "{stage}: logit count");
    let mut maximum = 0f32;
    for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
        let expected = expected.as_f64().unwrap() as f32;
        maximum = maximum.max((actual - expected).abs());
        assert!(actual.is_finite() && (actual - expected).abs() <= 2e-4 + 0.002 * expected.abs(),
            "{stage} logit {index}: {actual} != {expected}");
    }
    eprintln!("decoder {stage}: max_abs={maximum:e}");
}

fn exercise(device: Device, settings: Settings, routed: bool) {
    let started = std::time::Instant::now();
    eprintln!("decoder fixture: preparing {} weights (routed={routed})", device.backend());
    let fixture: Value = serde_json::from_str(if routed {
        include_str!("../../validation/results/fixtures/qwen-routed-decoder-reference.json")
    } else {
        include_str!("../../validation/results/fixtures/qwen-decoder-reference.json")
    })
    .unwrap();
    let directory = std::env::temp_dir().join(format!(
        "seismic-decoder-{}-{}-{routed}",
        std::process::id(),
        device.backend()
    ));
    std::fs::create_dir_all(&directory).unwrap();
    let mut bytes = Vec::new();
    let mut ranges = HashMap::new();
    for (name, record) in fixture["weights"].as_object().unwrap() {
        let offset = bytes.len();
        for value in record["values"].as_array().unwrap() {
            bytes.extend((value.as_f64().unwrap() as f32).to_le_bytes());
        }
        ranges.insert(name.clone(), (offset, bytes.len() - offset));
    }
    let path = directory.join("weights.bin");
    std::fs::write(&path, &bytes).unwrap();
    let source = Arc::new(FileSource::open(&path).unwrap());
    let descriptor = |name: &str| WeightDescriptor {
        name: name.into(),
        shape: fixture["weights"][name]["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap())
            .collect(),
        transform: Transform::Identity,
    };
    let geometry = Geometry {
        activation_dtype: DType::BF16,
        hidden: 8,
        intermediate: 12,
        vocabulary: 32,
        context_limit: 8,
        layers: vec![
            MixerKind::Recurrent,
            MixerKind::Attention,
            MixerKind::Recurrent,
            MixerKind::Attention,
        ],
        attention_heads: 4,
        kv_heads: 2,
        attention_width: 16,
        rotary_width: 12,
        rotary_base: 1e6,
        rotary_sections: [3, 2, 1, 0],
        epsilon: 1e-6,
        convolution_width: 4,
        recurrent_key_heads: 2,
        recurrent_value_heads: 4,
        recurrent_width: 4,
        recurrent_head_mapping: HeadMapping::Grouped,
        experts: routed.then_some(ExpertGeometry {
            count: 7,
            selected: 3,
            intermediate: 12,
            shared_intermediate: 16,
            normalize_selected: true,
        }),
    };
    let blocks = geometry
        .layers
        .iter()
        .enumerate()
        .map(|(i, kind)| {
            let d = |name: &str| descriptor(&format!("b{i}.{name}"));
            let mixer = match kind {
                MixerKind::Attention => MixerWeights::Attention(Box::new(AttentionWeights {
                    query_gate: d("query_gate"),
                    key: d("key"),
                    value: d("value"),
                    query_norm: d("query_norm"),
                    key_norm: d("key_norm"),
                    output: d("output"),
                })),
                MixerKind::Recurrent => MixerWeights::Recurrent(Box::new(RecurrentWeights {
                    query_key_value: d("qkv"),
                    gate: d("gate"),
                    alpha: d("alpha"),
                    beta: d("beta"),
                    convolution: d("convolution"),
                    decay: d("rate"),
                    time_bias: d("time_bias"),
                    norm: d("recurrent_norm"),
                    output: d("output"),
                })),
            };
            BlockWeights {
                input_norm: d("input_norm"),
                mixer,
                feedforward_norm: d("ff_norm"),
                feedforward: if routed {
                    FeedForwardWeights::Routed(Box::new(RoutedFeedForwardWeights {
                        router: d("router"),
                        shared_router: d("shared_router"),
                        expert_gate: d("expert_gate"),
                        expert_up: d("expert_up"),
                        expert_down: d("expert_down"),
                        shared_gate: d("shared_gate"),
                        shared_up: d("shared_up"),
                        shared_down: d("shared_down"),
                    }))
                } else {
                    FeedForwardWeights::Dense(Box::new(DenseFeedForwardWeights {
                        gate: d("ff_gate"),
                        up: d("ff_up"),
                        down: d("ff_down"),
                    }))
                },
            }
        })
        .collect();
    let description = Description {
        artifact_identity: ArtifactIdentity([0; 32]),
        geometry,
        embedding: descriptor("embedding"),
        output_norm: descriptor("output_norm"),
        output: descriptor("embedding"),
        blocks,
    };
    let device = Rc::new(device);
    eprintln!("decoder fixture: creating automatic importer");
    let mut importer = Importer::new(device.clone(), settings.clone()).unwrap();
    eprintln!("decoder fixture: compiling logical decoder and importing weights");
    let mut decoder = Decoder::compile(
        device,
        &description,
        |descriptor, target| {
            let import_started = std::time::Instant::now();
            eprintln!("decoder fixture: import {} -> {target:?}", descriptor.name);
            let (offset, nbytes) = ranges[&descriptor.name];
            let stored = Stored::Dense(StoredTensor {
                source: source.clone(),
                offset: offset as u64,
                nbytes: nbytes as u64,
                dtype: DType::F32,
                shape: descriptor.shape.clone(),
            });
            let imported = importer
                .import(descriptor, &stored, target)
                .map_err(|e| e.to_string());
            eprintln!("decoder fixture: import {} finished after {:.3}s", descriptor.name, import_started.elapsed().as_secs_f64());
            imported
        },
        settings,
        8,
        2,
    )
    .unwrap();
    eprintln!("decoder fixture: decoder ready after {:.3}s", started.elapsed().as_secs_f64());
    assert!(decoder.compiled_kernel_count() < 4 * 19);
    let store = decoder.state_store().clone();
    let mut state = store.create().unwrap();
    let mut checkpoint = None;
    for (position, step) in fixture["steps"].as_array().unwrap().iter().enumerate() {
        let token = step["token"].as_u64().unwrap() as u32;
        if position == 0 {
            eprintln!("decoder fixture: token {position}, initial proposal then abort");
            let rejected = decoder.propose(&mut state, token).unwrap();
            rejected.abort();
            assert_eq!(state.position(), 0);
            assert_eq!(store.occupied_rows(), 0);
        }
        eprintln!("decoder fixture: token {position}, sequential proposal");
        let serial = decoder.propose(&mut state, token).unwrap();
        let serial_logits = serial.logits().to_vec();
        serial.abort();
        eprintln!("decoder fixture: token {position}, batched proposal");
        let (proposed, observation) = decoder.propose_batched(&mut state, token).unwrap();
        assert!(observation.host_seconds > 0.);
        let actual = proposed.logits().to_vec();
        assert_eq!(
            actual, serial_logits,
            "batched publication differs at {position}"
        );
        assert_reference_logits(&actual, step, &format!("position {position}"));
        proposed.commit().unwrap();
        assert_eq!(state.position(), position + 1);
        if position == 0 {
            checkpoint = Some(state.checkpoint());
        }
        if position == 1 {
            eprintln!("decoder fixture: token {position}, forked-state proposal");
            let mut branch = checkpoint.take().unwrap().fork();
            let proposed = decoder.propose(&mut branch, token).unwrap();
            assert_eq!(actual, proposed.logits());
            proposed.commit().unwrap();
            assert_eq!(branch.position(), 2);
        }
    }
    assert_eq!(store.occupied_rows(), 3);
    assert!(decoder.propose(&mut state, 32).is_err());
    drop(state);
    assert_eq!(store.occupied_rows(), 0);
    if !routed {
        let reference = fixture["steps"].as_array().unwrap();
        let prompt = [
            reference[0]["token"].as_u64().unwrap() as u32,
            reference[1]["token"].as_u64().unwrap() as u32,
        ];
        let continuation = reference[2]["token"].as_u64().unwrap() as u32;
        let mut state = store.create().unwrap();
        eprintln!("decoder fixture: two-row sequential prefill then abort");
        let rejected = decoder.prefill(&mut state, &prompt).unwrap();
        let serial_logits = rejected.logits().to_vec();
        assert_reference_logits(&serial_logits, &reference[1], "two-row prefill before abort");
        rejected.abort();
        assert_eq!(state.position(), 0, "aborted prefill must leave the sequence empty");
        assert_eq!(store.occupied_rows(), 0, "aborted prefill must release both rows");

        eprintln!("decoder fixture: two-row batched prefill then commit");
        let (prefilled, observation) = decoder.prefill_batched(&mut state, &prompt).unwrap();
        assert!(observation.host_seconds > 0.);
        assert_eq!(prefilled.logits(), serial_logits, "batched prefill publication differs");
        assert_reference_logits(prefilled.logits(), &reference[1], "two-row batched prefill");
        prefilled.commit().unwrap();
        assert_eq!(state.position(), 2, "prefill must commit both tokens");
        assert_eq!(store.occupied_rows(), 2);

        eprintln!("decoder fixture: decode continuation after two-row prefill");
        let (decoded, observation) = decoder.propose_batched(&mut state, continuation).unwrap();
        assert!(observation.host_seconds > 0.);
        assert_reference_logits(decoded.logits(), &reference[2], "decode after two-row prefill");
        decoded.commit().unwrap();
        assert_eq!(state.position(), 3);
        assert_eq!(store.occupied_rows(), 3);
        drop(state);
        assert_eq!(store.occupied_rows(), 0);
    }
    drop(decoder);
    drop(importer);
    drop(source);
    std::fs::remove_dir_all(directory).unwrap();
}
#[test]
#[ignore = "requires a Metal device"]
fn metal_dense_decoder() {
    exercise(Device::metal().unwrap(), Settings::default(), false);
}
#[test]
#[ignore = "requires a Metal device"]
fn metal_routed_decoder() {
    exercise(Device::metal().unwrap(), Settings::default(), true);
}
