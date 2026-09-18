use seismic_engine::{
    models::qwen35::{decoder::Decoder, *},
    weights::{
        descriptor::{ArtifactIdentity, Stored, StoredTensor, Transform, WeightDescriptor},
        residency::Importer,
        source::FileSource,
    },
};
use seismic_lang::{lower::Options, types::DType};
use seismic_runtime::{Candidate, Device, plan::Diagnostic};
use serde_json::Value;
use std::{collections::HashMap, rc::Rc, sync::Arc};
fn exercise(device: Device, candidate: Candidate, routed: bool) {
    let fixture: Value = serde_json::from_str(if routed {
        include_str!("../../validation/fixtures/qwen-routed-decoder-reference.json")
    } else {
        include_str!("../../validation/fixtures/qwen-decoder-reference.json")
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
    let mut importer = Importer::new(device.clone(), candidate.clone()).unwrap();
    let mut decoder = Decoder::compile_diagnostic(
        device,
        &description,
        |descriptor, target| {
            let (offset, nbytes) = ranges[&descriptor.name];
            let stored = Stored::Dense(StoredTensor {
                source: source.clone(),
                offset: offset as u64,
                nbytes: nbytes as u64,
                dtype: DType::F32,
                shape: descriptor.shape.clone(),
            });
            importer
                .import(descriptor, &stored, target)
                .map_err(|e| e.to_string())
        },
        Diagnostic {
            candidate,
            lowering: Options {
                piece: Some(4),
                ..Default::default()
            },
        },
        8,
        2,
    )
    .unwrap();
    assert!(decoder.compiled_kernel_count() < 4 * 19);
    let store = decoder.state_store().clone();
    let mut state = store.create().unwrap();
    let mut checkpoint = None;
    for (position, step) in fixture["steps"].as_array().unwrap().iter().enumerate() {
        let token = step["token"].as_u64().unwrap() as u32;
        if position == 0 {
            let rejected = decoder.propose(&mut state, token).unwrap();
            rejected.abort();
            assert_eq!(state.position(), 0);
            assert_eq!(store.occupied_rows(), 0);
        }
        let serial = decoder.propose(&mut state, token).unwrap();
        let serial_logits = serial.logits().to_vec();
        serial.abort();
        let (proposed, observation) = decoder.propose_batched(&mut state, token).unwrap();
        assert!(observation.host_seconds > 0.);
        let actual = proposed.logits().to_vec();
        assert_eq!(
            actual, serial_logits,
            "batched publication differs at {position}"
        );
        let expected = step["logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let mut maximum = 0f32;
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            maximum = maximum.max((a - e).abs());
            assert!(
                (a - e).abs() <= 2e-4 + 0.002 * e.abs(),
                "position {position} logit {i}: {a} != {e}"
            );
        }
        eprintln!("decoder position {position}: max_abs={maximum:e}");
        proposed.commit().unwrap();
        assert_eq!(state.position(), position + 1);
        if position == 0 {
            checkpoint = Some(state.checkpoint());
        }
        if position == 1 {
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
    drop(decoder);
    drop(importer);
    drop(source);
    std::fs::remove_dir_all(directory).unwrap();
}
#[test]
fn cpu_dense_decoder() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::Materialize,
        },
        false,
    );
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_dense_decoder() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
        false,
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_dense_decoder() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
        false,
    );
}

#[test]
fn cpu_routed_decoder() {
    exercise(
        Device::cpu(),
        Candidate::Cpu {
            loads: seismic_realization::LoadStrategy::Materialize,
        },
        true,
    );
}
#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn metal_routed_decoder() {
    exercise(
        Device::metal().unwrap(),
        Candidate::Metal(Default::default()),
        true,
    );
}
#[test]
#[ignore = "requires CUDA hardware"]
fn cuda_routed_decoder() {
    exercise(
        Device::cuda(0).unwrap(),
        Candidate::Cuda {
            options: seismic_realization::ScalarOptions {
                dispatch: seismic_realization::Dispatch::ParallelRoot,
                loads: seismic_realization::LoadStrategy::Materialize,
            },
            threads_per_block: 32,
        },
        true,
    );
}
