use magnitude_engine::{
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

#[test]
fn local_gguf_loading_shares_artifact_identity_with_tokenizer_and_templates() {
    use magnitude_engine::{
        chat::{ChatRequest, PreparedChat, TemplateSelection},
        inputs::ByteBpeTokenizer,
        models::qwen35::loading::Model,
    };
    fn string(out: &mut Vec<u8>, value: &str) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value.as_bytes());
    }
    fn kind(value: &Scalar) -> u32 {
        match value {
            Scalar::String(_) => 8,
            Scalar::Bool(_) => 7,
            Scalar::Unsigned(_) => 10,
            Scalar::Signed(_) => 11,
            Scalar::Float(_) => 12,
        }
    }
    fn scalar(out: &mut Vec<u8>, value: &Scalar) {
        match value {
            Scalar::String(s) => string(out, s),
            Scalar::Bool(b) => out.push(u8::from(*b)),
            Scalar::Unsigned(n) => out.extend_from_slice(&n.to_le_bytes()),
            Scalar::Signed(n) => out.extend_from_slice(&n.to_le_bytes()),
            Scalar::Float(n) => out.extend_from_slice(&n.to_le_bytes()),
        }
    }
    let mut d = directory(false);
    let mut alphabet: Vec<u8> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut codes: Vec<u32> = alphabet.iter().map(|&b| u32::from(b)).collect();
    let mut next = 256;
    for b in 0..=255 {
        if !alphabet.contains(&b) {
            alphabet.push(b);
            codes.push(next);
            next += 1;
        }
    }
    let mut pieces = vec![Scalar::String(String::new()); 256];
    for (b, code) in alphabet.into_iter().zip(codes) {
        pieces[b as usize] = Scalar::String(char::from_u32(code).unwrap().to_string());
    }
    pieces.push(Scalar::String("<eos>".into()));
    for (name, value) in [
        (
            "tokenizer.ggml.model",
            Value::Scalar(Scalar::String("gpt2".into())),
        ),
        (
            "tokenizer.ggml.pre",
            Value::Scalar(Scalar::String("qwen35".into())),
        ),
        ("tokenizer.ggml.tokens", Value::Array(pieces)),
        (
            "tokenizer.ggml.token_type",
            Value::Array(
                (0..257)
                    .map(|i| Scalar::Unsigned(if i == 256 { 3 } else { 1 }))
                    .collect(),
            ),
        ),
        ("tokenizer.ggml.merges", Value::Array(vec![])),
        (
            "tokenizer.ggml.eos_token_id",
            Value::Scalar(Scalar::Unsigned(256)),
        ),
        (
            "tokenizer.chat_template",
            Value::Scalar(Scalar::String("{{ messages[0].content }}".into())),
        ),
    ] {
        d.metadata.push(Metadata {
            name: name.into(),
            value,
        });
    }
    let embedding = d
        .tensors
        .iter_mut()
        .find(|t| t.name == "token_embd.weight")
        .unwrap();
    embedding.shape[0] = 257;
    embedding.nbytes = 257 * 8 * 4;
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(d.tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(d.metadata.len() as u64).to_le_bytes());
    for item in &d.metadata {
        string(&mut bytes, &item.name);
        match &item.value {
            Value::Scalar(value) => {
                bytes.extend_from_slice(&kind(value).to_le_bytes());
                scalar(&mut bytes, value);
            }
            Value::Array(values) => {
                bytes.extend_from_slice(&9u32.to_le_bytes());
                bytes.extend_from_slice(&values.first().map_or(8, kind).to_le_bytes());
                bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in values {
                    scalar(&mut bytes, value);
                }
            }
        }
    }
    let mut offset = 0u64;
    for tensor in &d.tensors {
        string(&mut bytes, &tensor.name);
        bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for dimension in tensor.shape.iter().rev() {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&(tensor.encoding as u32).to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
        offset = (offset + tensor.nbytes).div_ceil(32) * 32;
    }
    bytes.resize(bytes.len().div_ceil(32) * 32 + offset as usize, 0);
    let path = std::env::temp_dir().join(format!(
        "seismic-qwen-loading-{}-{}.gguf",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    struct Temp(std::path::PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let temp = Temp(path);
    std::fs::write(&temp.0, bytes).unwrap();
    let model = Model::open(&temp.0).unwrap();
    assert_eq!(model.description().geometry.vocabulary, 257);
    assert_eq!(model.description().output, model.description().embedding);
    let identity = model.description().artifact_identity.to_string();
    // GGUF interpretation is retained; later pathname replacement cannot alter
    // its tokenizer/template metadata or the open weight source.
    let replacement = temp.0.with_extension("replacement");
    std::fs::write(&replacement, b"invalid replacement").unwrap();
    std::fs::rename(&replacement, &temp.0).unwrap();
    let tokenizer = ByteBpeTokenizer::new(model.tokenizer_config().unwrap()).unwrap();
    assert_eq!(tokenizer.artifact_identity(), identity);
    let prepared = PreparedChat::prepare(
        &model.templates().unwrap(),
        &tokenizer,
        &ChatRequest::new(
            vec![serde_json::json!({"role":"user","content":"hello"})],
            0,
        ),
        &TemplateSelection::default(),
    )
    .unwrap();
    assert_eq!(prepared.prompt(), "hello");
    assert_eq!(prepared.prompt_tokens(), 5);
    assert!(Model::open(&temp.0).is_err());
}
