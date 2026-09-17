use seismic_engine::{
    models::qwen35::{gguf::inspect, FeedForwardWeights, HeadMapping, MixerWeights},
    weights::{
        descriptor::ArtifactIdentity,
        gguf::{ByteOrder, Directory, Encoding, Metadata, Scalar, Tensor, Value},
    },
};
fn directory(routed: bool) -> Directory {
    let architecture = if routed { "qwen35moe" } else { "qwen35" };
    let mut metadata = vec![Metadata {
        name: "general.architecture".into(),
        value: Value::Scalar(Scalar::String(architecture.into())),
    }];
    for (name, n) in [
        ("block_count", 2),
        ("full_attention_interval", 2),
        ("embedding_length", 8),
        ("feed_forward_length", 12),
        ("context_length", 64),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 4),
        ("attention.value_length", 4),
        ("rope.dimension_count", 4),
        ("ssm.conv_kernel", 3),
        ("ssm.group_count", 1),
        ("ssm.time_step_rank", 2),
        ("ssm.state_size", 4),
        ("ssm.inner_size", 8),
        ("expert_count", 4),
        ("expert_used_count", 2),
        ("expert_feed_forward_length", 6),
        ("expert_shared_feed_forward_length", 8),
    ] {
        metadata.push(Metadata {
            name: format!("{architecture}.{name}"),
            value: Value::Scalar(Scalar::Unsigned(n)),
        });
    }
    for (name, n) in [
        ("rope.freq_base", 10000.0),
        ("attention.layer_norm_rms_epsilon", 1e-6),
    ] {
        metadata.push(Metadata {
            name: format!("{architecture}.{name}"),
            value: Value::Scalar(Scalar::Float(n)),
        });
    }
    metadata.push(Metadata {
        name: format!("{architecture}.rope.dimension_sections"),
        value: Value::Array(vec![
            Scalar::Unsigned(1),
            Scalar::Unsigned(1),
            Scalar::Unsigned(0),
            Scalar::Unsigned(0),
        ]),
    });
    let mut tensors = Vec::new();
    let mut add = |name: String, shape: &[u64]| {
        tensors.push(Tensor {
            name,
            shape: shape.into(),
            encoding: Encoding::F32,
            offset: 0,
            nbytes: shape.iter().product::<u64>() * 4,
        })
    };
    add("token_embd.weight".into(), &[16, 8]);
    add("output_norm.weight".into(), &[8]);
    for (name, shape) in [
        ("attn_qkv.weight", vec![16, 8]),
        ("attn_gate.weight", vec![8, 8]),
        ("ssm_alpha.weight", vec![2, 8]),
        ("ssm_beta.weight", vec![2, 8]),
        ("ssm_conv1d.weight", vec![16, 3]),
        ("ssm_a", vec![2]),
        ("ssm_dt.bias", vec![2]),
        ("ssm_norm.weight", vec![4]),
        ("ssm_out.weight", vec![8, 8]),
    ] {
        add(format!("blk.0.{name}"), &shape);
    }
    for (name, shape) in [
        ("attn_q.weight", vec![16, 8]),
        ("attn_k.weight", vec![4, 8]),
        ("attn_v.weight", vec![4, 8]),
        ("attn_q_norm.weight", vec![4]),
        ("attn_k_norm.weight", vec![4]),
        ("attn_output.weight", vec![8, 8]),
    ] {
        add(format!("blk.1.{name}"), &shape);
    }
    for i in 0..2 {
        add(format!("blk.{i}.attn_norm.weight"), &[8]);
        add(format!("blk.{i}.post_attention_norm.weight"), &[8]);
        let roles = if routed {
            vec![
                ("ffn_gate_inp.weight", vec![4, 8]),
                ("ffn_gate_inp_shexp.weight", vec![8]),
                ("ffn_gate_exps.weight", vec![4, 6, 8]),
                ("ffn_up_exps.weight", vec![4, 6, 8]),
                ("ffn_down_exps.weight", vec![4, 8, 6]),
                ("ffn_gate_shexp.weight", vec![8, 8]),
                ("ffn_up_shexp.weight", vec![8, 8]),
                ("ffn_down_shexp.weight", vec![8, 8]),
            ]
        } else {
            vec![
                ("ffn_gate.weight", vec![12, 8]),
                ("ffn_up.weight", vec![12, 8]),
                ("ffn_down.weight", vec![8, 12]),
            ]
        };
        for (name, shape) in roles {
            add(format!("blk.{i}.{name}"), &shape);
        }
    }
    Directory {
        version: 3,
        byte_order: ByteOrder::Little,
        alignment: 32,
        data_offset: 0,
        metadata,
        tensors,
    }
}
fn set(d: &mut Directory, key: &str, value: Value) {
    let name = format!("qwen35.{key}");
    if let Some(m) = d.metadata.iter_mut().find(|m| m.name == name) {
        m.value = value;
    } else {
        d.metadata.push(Metadata { name, value });
    }
}
#[test]
fn dense_and_routed_roles_preserve_geometry_and_tied_output() {
    for routed in [false, true] {
        let d = directory(routed);
        let model = inspect(&d, ArtifactIdentity([0; 32])).unwrap();
        assert_eq!(model.geometry.recurrent_channels().unwrap(), 16);
        assert_eq!(model.geometry.recurrent_head_mapping, HeadMapping::Tiled);
        assert_eq!(model.output, model.embedding);
        assert!(matches!(&model.blocks[0].mixer, MixerWeights::Recurrent(_)));
        assert!(matches!(&model.blocks[1].mixer, MixerWeights::Attention(_)));
        assert_eq!(
            matches!(&model.blocks[0].feedforward, FeedForwardWeights::Routed(_)),
            routed
        );
    }
}
#[test]
fn rejects_invalid_geometry_and_missing_wrong_or_unbound_roles() {
    for (key, value) in [
        ("attention.head_count", Scalar::Unsigned(3)),
        ("ssm.inner_size", Scalar::Unsigned(9)),
        (
            "attention.layer_norm_rms_epsilon",
            Scalar::Float(f64::INFINITY),
        ),
        ("block_count", Scalar::Bool(true)),
        ("recurrent_key_heads", Scalar::Unsigned(u64::MAX)),
    ] {
        let mut d = directory(false);
        let key = if key == "recurrent_key_heads" {
            "ssm.group_count"
        } else {
            key
        };
        set(&mut d, key, Value::Scalar(value));
        assert!(inspect(&d, ArtifactIdentity([0; 32])).is_err(), "{key}");
    }
    let mut d = directory(false);
    d.tensors.pop();
    assert!(inspect(&d, ArtifactIdentity([0; 32])).is_err());
    let mut d = directory(false);
    d.tensors[2].shape[0] += 1;
    assert!(inspect(&d, ArtifactIdentity([0; 32])).is_err());
    let mut d = directory(false);
    let mut extra = d.tensors[2].clone();
    extra.name = "blk.0.unbound.weight".into();
    d.tensors.push(extra);
    assert!(inspect(&d, ArtifactIdentity([0; 32])).is_err());
}
#[test]
fn explicit_mixer_flags_and_speculative_blocks_keep_main_layer_order() {
    let mut d = directory(false);
    set(&mut d, "block_count", Value::Scalar(Scalar::Unsigned(3)));
    set(
        &mut d,
        "nextn_predict_layers",
        Value::Scalar(Scalar::Unsigned(1)),
    );
    set(
        &mut d,
        "attention.recurrent_layers",
        Value::Array(vec![Scalar::Bool(true), Scalar::Bool(false)]),
    );
    let mut extra = d.tensors[2].clone();
    extra.name = "blk.2.speculative.weight".into();
    d.tensors.push(extra);
    assert_eq!(
        inspect(&d, ArtifactIdentity([0; 32])).unwrap().blocks.len(),
        2
    );
    d.tensors.last_mut().unwrap().name = "blk.02.speculative.weight".into();
    assert!(inspect(&d, ArtifactIdentity([0; 32])).is_err());
    d.tensors.pop();
    set(
        &mut d,
        "attention.recurrent_layers",
        Value::Array(vec![Scalar::Unsigned(1), Scalar::Bool(false)]),
    );
    assert!(inspect(&d, ArtifactIdentity([0; 32])).is_err());
}
