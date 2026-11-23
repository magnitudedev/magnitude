use magnitude_artifacts::{
    gguf::{ByteOrder, Directory, Encoding, Metadata, Scalar, TensorDescriptor, Value},
    ArtifactIdentity, PackageIdentity, TokenId,
};
use magnitude_family_contracts::{
    BlockAttention, BlockLayout, DraftMethod, DraftVariant, HyperConnectionMixing, InputNorm,
    ModelDefinition, ModelFamily, Operator, OutputForm, OutputProjection, OutputRotary,
    QueryProjection, ResidualForm, Rotary, ScoreFunction, SinkhornInput, SublayerIndex, TapPoint,
    ValueSource,
};
use magnitude_family_deepseek41::DeepSeek41Family;
use magnitude_family_dflash::{inspect, Error};

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([41; 32]),
        projector: None,
    }
}

fn meta(name: &str, value: Value) -> Metadata {
    Metadata {
        name: name.into(),
        value,
    }
}

fn tensor(name: &str, shape: &[u64], encoding: Encoding) -> TensorDescriptor {
    TensorDescriptor {
        name: name.into(),
        shape: shape.to_vec(),
        encoding,
        offset: 0,
        nbytes: 0,
    }
}

/// Synthetic DeepSeek V4.1 target directory matching `deepseek41` metadata and tensors.
fn deepseek41_target_directory() -> Directory {
    let vocab = 1024u64;
    let hidden = 128u64;
    let layers = 4u64;
    let heads = 4u64;
    let width = 64u64;
    let q_rank = 32u64;
    let groups = 2u64;
    let o_rank = 16u64;
    let streams = 4u64;
    let mixes = (2 + streams) * streams;
    let experts = 8u64;
    let inter = 64u64;
    let cols = (heads * width) / groups;
    let ranks = groups * o_rank;

    let metadata = vec![
        meta("general.architecture", Value::Scalar(Scalar::String("deepseek41".into()))),
        meta("deepseek41.block_count", Value::Scalar(Scalar::Unsigned(layers))),
        meta("deepseek41.context_length", Value::Scalar(Scalar::Unsigned(4096))),
        meta("deepseek41.embedding_length", Value::Scalar(Scalar::Unsigned(hidden))),
        meta("deepseek41.vocab_size", Value::Scalar(Scalar::Unsigned(vocab))),
        meta("deepseek41.attention.head_count", Value::Scalar(Scalar::Unsigned(heads))),
        meta("deepseek41.attention.head_count_kv", Value::Scalar(Scalar::Unsigned(1))),
        meta("deepseek41.attention.key_length", Value::Scalar(Scalar::Unsigned(width))),
        meta("deepseek41.attention.value_length", Value::Scalar(Scalar::Unsigned(width))),
        meta("deepseek41.attention.q_lora_rank", Value::Scalar(Scalar::Unsigned(q_rank))),
        meta("deepseek41.attention.sliding_window", Value::Scalar(Scalar::Unsigned(128))),
        meta("deepseek41.attention.output_group_count", Value::Scalar(Scalar::Unsigned(groups))),
        meta("deepseek41.attention.output_lora_rank", Value::Scalar(Scalar::Unsigned(o_rank))),
        meta("deepseek41.attention.layer_norm_rms_epsilon", Value::Scalar(Scalar::Float(1e-6))),
        meta("deepseek41.attention.compress_ratios", Value::Array(vec![0_u64; layers as usize].into())),
        meta("deepseek41.attention.indexer.head_count", Value::Scalar(Scalar::Unsigned(4))),
        meta("deepseek41.attention.indexer.key_length", Value::Scalar(Scalar::Unsigned(32))),
        meta("deepseek41.attention.indexer.top_k", Value::Scalar(Scalar::Unsigned(16))),
        meta("deepseek41.rope.dimension_count", Value::Scalar(Scalar::Unsigned(32))),
        meta("deepseek41.rope.freq_base", Value::Scalar(Scalar::Float(10000.0))),
        meta("deepseek41.expert_count", Value::Scalar(Scalar::Unsigned(experts))),
        meta("deepseek41.expert_used_count", Value::Scalar(Scalar::Unsigned(2))),
        meta("deepseek41.expert_feed_forward_length", Value::Scalar(Scalar::Unsigned(inter))),
        meta("deepseek41.expert_shared_count", Value::Scalar(Scalar::Unsigned(1))),
        meta("deepseek41.expert_weights_scale", Value::Scalar(Scalar::Float(1.0))),
        meta("deepseek41.expert_weights_norm", Value::Scalar(Scalar::Bool(true))),
        meta("deepseek41.expert_gating_func", Value::Scalar(Scalar::Unsigned(4))),
        meta("deepseek41.hash_layer_count", Value::Scalar(Scalar::Unsigned(0))),
        meta("deepseek41.swiglu_clamp_exp", Value::Array(vec![7.0_f64; layers as usize].into())),
        meta("deepseek41.swiglu_clamp_shexp", Value::Array(vec![7.0_f64; layers as usize].into())),
        meta("deepseek41.hyper_connection.count", Value::Scalar(Scalar::Unsigned(streams))),
        meta("deepseek41.hyper_connection.sinkhorn_iterations", Value::Scalar(Scalar::Unsigned(20))),
        meta("deepseek41.hyper_connection.epsilon", Value::Scalar(Scalar::Float(1e-6))),
    ];

    let mut tensors = vec![
        tensor("token_embd.weight", &[vocab, hidden], Encoding::BF16),
        tensor("output.weight", &[vocab, hidden], Encoding::BF16),
        tensor("output_norm.weight", &[hidden], Encoding::F32),
    ];
    for layer in 0..layers {
        let p = format!("blk.{layer}.");
        for site in ["attn", "ffn"] {
            tensors.push(tensor(&format!("{p}hc_{site}_fn.weight"), &[mixes, streams * hidden], Encoding::F32));
            tensors.push(tensor(&format!("{p}hc_{site}_base.weight"), &[mixes], Encoding::F32));
            tensors.push(tensor(&format!("{p}hc_{site}_scale.weight"), &[3], Encoding::F32));
        }
        tensors.extend([
            tensor(&format!("{p}attn_norm.weight"), &[hidden], Encoding::F32),
            tensor(&format!("{p}attn_q_a.weight"), &[q_rank, hidden], Encoding::BF16),
            tensor(&format!("{p}attn_q_a_norm.weight"), &[q_rank], Encoding::F32),
            tensor(&format!("{p}attn_q_b.weight"), &[heads * width, q_rank], Encoding::BF16),
            tensor(&format!("{p}attn_kv.weight"), &[width, hidden], Encoding::BF16),
            tensor(&format!("{p}attn_kv_a_norm.weight"), &[width], Encoding::F32),
            tensor(&format!("{p}attn_sinks.weight"), &[heads], Encoding::F32),
            tensor(&format!("{p}attn_output_a.weight"), &[ranks, cols], Encoding::BF16),
            tensor(&format!("{p}attn_output_b.weight"), &[hidden, ranks], Encoding::BF16),
            tensor(&format!("{p}ffn_norm.weight"), &[hidden], Encoding::F32),
            tensor(&format!("{p}ffn_gate_inp.weight"), &[experts, hidden], Encoding::F32),
            tensor(&format!("{p}exp_probs_b.bias"), &[experts], Encoding::F32),
            tensor(&format!("{p}exp_probs_b_vl.bias"), &[experts], Encoding::F32),
            tensor(&format!("{p}ffn_gate_exps.weight"), &[experts, inter, hidden], Encoding::Mxfp4),
            tensor(&format!("{p}ffn_up_exps.weight"), &[experts, inter, hidden], Encoding::Mxfp4),
            tensor(&format!("{p}ffn_down_exps.weight"), &[experts, hidden, inter], Encoding::Mxfp4),
            tensor(&format!("{p}ffn_gate_shexp.weight"), &[inter, hidden], Encoding::BF16),
            tensor(&format!("{p}ffn_up_shexp.weight"), &[inter, hidden], Encoding::BF16),
            tensor(&format!("{p}ffn_down_shexp.weight"), &[hidden, inter], Encoding::BF16),
        ]);
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

/// Synthetic DeepSeek V4.1 DSpark GGUF directory matching `DeepseekV41DSparkModel`.
fn deepseek41_dspark_directory() -> Directory {
    let vocab = 1024u64;
    let hidden = 128u64;
    let layers = 3u64;
    let heads = 4u64;
    let width = 64u64;
    let q_rank = 32u64;
    let groups = 2u64;
    let o_rank = 16u64;
    let streams = 4u64;
    let mixes = (2 + streams) * streams;
    let experts = 4u64;
    let inter = 64u64;
    let markov_rank = 32u64;
    let cols = (heads * width) / groups;
    let ranks = groups * o_rank;

    let metadata = vec![
        meta("general.architecture", Value::Scalar(Scalar::String("dflash".into()))),
        meta("tokenizer.ggml.mask_token_id", Value::Scalar(Scalar::Unsigned(42))),
        meta("dflash.block_count", Value::Scalar(Scalar::Unsigned(layers))),
        meta("dflash.context_length", Value::Scalar(Scalar::Unsigned(4096))),
        meta("dflash.embedding_length", Value::Scalar(Scalar::Unsigned(hidden))),
        meta("dflash.feed_forward_length", Value::Scalar(Scalar::Unsigned(inter))),
        meta("dflash.vocab_size", Value::Scalar(Scalar::Unsigned(vocab))),
        meta("dflash.attention.head_count", Value::Scalar(Scalar::Unsigned(heads))),
        meta("dflash.attention.head_count_kv", Value::Scalar(Scalar::Unsigned(1))),
        meta("dflash.attention.key_length", Value::Scalar(Scalar::Unsigned(width))),
        meta("dflash.attention.value_length", Value::Scalar(Scalar::Unsigned(width))),
        meta("dflash.rope.dimension_count", Value::Scalar(Scalar::Unsigned(32))),
        meta("dflash.rope.freq_base", Value::Scalar(Scalar::Float(10000.0))),
        meta("dflash.attention.layer_norm_rms_epsilon", Value::Scalar(Scalar::Float(1e-6))),
        meta("dflash.attention.q_lora_rank", Value::Scalar(Scalar::Unsigned(q_rank))),
        meta("dflash.attention.sliding_window", Value::Scalar(Scalar::Unsigned(128))),
        meta("dflash.expert_count", Value::Scalar(Scalar::Unsigned(experts))),
        meta("dflash.expert_used_count", Value::Scalar(Scalar::Unsigned(2))),
        meta("dflash.expert_feed_forward_length", Value::Scalar(Scalar::Unsigned(inter))),
        meta("dflash.expert_shared_count", Value::Scalar(Scalar::Unsigned(1))),
        meta("dflash.expert_weights_scale", Value::Scalar(Scalar::Float(1.0))),
        meta("dflash.expert_weights_norm", Value::Scalar(Scalar::Bool(true))),
        meta("dflash.expert_gating_func", Value::Scalar(Scalar::Unsigned(4))),
        meta("dflash.swiglu_clamp_exp", Value::Array(vec![7.0_f64; layers as usize].into())),
        meta("dflash.swiglu_clamp_shexp", Value::Array(vec![7.0_f64; layers as usize].into())),
        meta("dflash.attention.output_group_count", Value::Scalar(Scalar::Unsigned(groups))),
        meta("dflash.attention.output_lora_rank", Value::Scalar(Scalar::Unsigned(o_rank))),
        meta("dflash.attention.compress_ratios", Value::Array(vec![0_u64; layers as usize].into())),
        meta("dflash.hyper_connection.count", Value::Scalar(Scalar::Unsigned(streams))),
        meta("dflash.hyper_connection.sinkhorn_iterations", Value::Scalar(Scalar::Unsigned(20))),
        meta("dflash.hyper_connection.epsilon", Value::Scalar(Scalar::Float(1e-6))),
        meta("dflash.hyper_connection.single_pass", Value::Scalar(Scalar::Bool(true))),
        meta("dflash.block_size", Value::Scalar(Scalar::Unsigned(6))),
        meta("dflash.target_layers", Value::Array(vec![1_u64, 3].into())),
        meta("dflash.sample_from_anchor", Value::Scalar(Scalar::Bool(true))),
        meta("dflash.has_confidence_head", Value::Scalar(Scalar::Bool(true))),
    ];

    let mut tensors = vec![
        tensor("fc.weight", &[hidden, 2 * hidden], Encoding::BF16),
        tensor("enc.output_norm.weight", &[hidden], Encoding::F32),
        tensor("output_norm.weight", &[hidden], Encoding::F32),
        tensor("markov_w1.weight", &[vocab, markov_rank], Encoding::BF16),
        tensor("markov_w2.weight", &[vocab, markov_rank], Encoding::BF16),
        tensor("conf_proj.weight", &[1, hidden + markov_rank], Encoding::F32),
    ];
    for layer in 0..layers {
        let p = format!("blk.{layer}.");
        for site in ["attn", "ffn"] {
            tensors.push(tensor(&format!("{p}hc_{site}_fn.weight"), &[mixes, streams * hidden], Encoding::F32));
            tensors.push(tensor(&format!("{p}hc_{site}_base.weight"), &[mixes], Encoding::F32));
            tensors.push(tensor(&format!("{p}hc_{site}_scale.weight"), &[3], Encoding::F32));
        }
        tensors.extend([
            tensor(&format!("{p}attn_norm.weight"), &[hidden], Encoding::F32),
            tensor(&format!("{p}attn_q_a.weight"), &[q_rank, hidden], Encoding::BF16),
            tensor(&format!("{p}attn_q_a_norm.weight"), &[q_rank], Encoding::F32),
            tensor(&format!("{p}attn_q_b.weight"), &[heads * width, q_rank], Encoding::BF16),
            tensor(&format!("{p}attn_kv.weight"), &[width, hidden], Encoding::BF16),
            tensor(&format!("{p}attn_kv_a_norm.weight"), &[width], Encoding::F32),
            tensor(&format!("{p}attn_sinks.weight"), &[heads], Encoding::F32),
            tensor(&format!("{p}attn_output_a.weight"), &[ranks, cols], Encoding::BF16),
            tensor(&format!("{p}attn_output_b.weight"), &[hidden, ranks], Encoding::BF16),
            tensor(&format!("{p}ffn_norm.weight"), &[hidden], Encoding::F32),
            tensor(&format!("{p}ffn_gate_inp.weight"), &[experts, hidden], Encoding::F32),
            tensor(&format!("{p}exp_probs_b.bias"), &[experts], Encoding::F32),
            tensor(&format!("{p}ffn_gate_exps.weight"), &[experts, inter, hidden], Encoding::Mxfp4),
            tensor(&format!("{p}ffn_up_exps.weight"), &[experts, inter, hidden], Encoding::Mxfp4),
            tensor(&format!("{p}ffn_down_exps.weight"), &[experts, hidden, inter], Encoding::Mxfp4),
            tensor(&format!("{p}ffn_gate_shexp.weight"), &[inter, hidden], Encoding::BF16),
            tensor(&format!("{p}ffn_up_shexp.weight"), &[inter, hidden], Encoding::BF16),
            tensor(&format!("{p}ffn_down_shexp.weight"), &[hidden, inter], Encoding::BF16),
        ]);
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

fn target_definition() -> ModelDefinition {
    DeepSeek41Family
        .inspect(&deepseek41_target_directory(), None, identity())
        .expect("synthetic deepseek41 target inspects cleanly")
}

#[test]
fn deepseek41_dspark_inspects_and_validates_against_mhc_target() {
    let mut target = target_definition();
    let draft = inspect(&deepseek41_dspark_directory(), &target, &|layer| {
        DeepSeek41Family.layer_entry(&target, layer)
    })
    .expect("DeepSeek V4.1 DSpark inspects cleanly");

    assert_eq!(draft.method.variant(), DraftVariant::DSpark);
    assert_eq!(draft.layout, BlockLayout::AnchorFirst);
    assert_eq!(draft.block_size, 6);
    assert_eq!(draft.max_proposals(), 6);
    assert_eq!(draft.mask_token, TokenId(42));
    assert_eq!(
        draft.taps,
        vec![
            TapPoint::Sublayer(SublayerIndex { block: 1, sublayer: 0 }),
            TapPoint::Sublayer(SublayerIndex { block: 3, sublayer: 0 }),
        ]
    );
    assert_eq!(draft.block_attention, vec![BlockAttention::Bidirectional; 3]);
    assert!(matches!(
        draft.residual,
        ResidualForm::HyperConnections(ref hc)
            if hc.streams == 4
                && matches!(
                    hc.mixing,
                    HyperConnectionMixing::Sinkhorn {
                        iterations: 20,
                        input: SinkhornInput::PreviousSite,
                        ..
                    }
                )
    ));
    let DraftMethod::DSpark { markov, confidence } = &draft.method else {
        panic!("expected DSpark method");
    };
    assert_eq!(markov.rank, 32);
    assert_eq!(confidence.weight.shape, [1, 128 + 32]);
    assert!(confidence.bias.is_none());

    assert_eq!(draft.blocks.len(), 3);
    for block in &draft.blocks {
        let [attn_sublayer, ffn_sublayer] = block.sublayers.as_slice() else {
            panic!("expected [attention, routed_ffn] sublayers");
        };
        assert!(matches!(attn_sublayer.input, InputNorm::HyperConnectionMix(_)));
        assert_eq!(attn_sublayer.output, OutputForm::Scatter);
        let Operator::Attention(attention) = &attn_sublayer.op else {
            panic!("expected MLA attention");
        };
        assert!(matches!(attention.query, QueryProjection::LowRank { .. }));
        assert!(matches!(attention.output, OutputProjection::GroupedLowRank { groups: 2, rank: 16, .. }));
        assert!(attention.sinks.is_some());
        assert!(matches!(
            attention.rotary,
            Rotary::Table { output: OutputRotary::Inverse, .. }
        ));
        assert!(matches!(
            &attention.fresh().rows,
            magnitude_family_contracts::HistoryRows::Own(
                magnitude_family_contracts::RowProducer::Projected(rows)
            ) if matches!(rows.value, ValueSource::StoredKey)
        ));

        assert!(matches!(ffn_sublayer.input, InputNorm::HyperConnectionMix(_)));
        assert_eq!(ffn_sublayer.output, OutputForm::Scatter);
        let Operator::RoutedFfn(routed) = &ffn_sublayer.op else {
            panic!("expected routed MoE feed-forward");
        };
        assert_eq!(routed.experts, 4);
        assert_eq!(routed.selected, 2);
        assert_eq!(routed.router.score, ScoreFunction::SqrtSoftplus);
        assert!(routed.router.media_bias.is_none());
        assert!(routed.shared.is_some());
    }

    target.draft = Some(draft);
    target.validate().expect("target + DSpark draft validates cleanly");
}

#[test]
fn deepseek41_dspark_rejects_nonzero_compress_ratios() {
    let target = target_definition();
    let mut dir = deepseek41_dspark_directory();
    dir.metadata
        .iter_mut()
        .find(|entry| entry.name == "dflash.attention.compress_ratios")
        .unwrap()
        .value = Value::Array(vec![0_u64, 4, 0].into());
    let err = inspect(&dir, &target, &|layer| {
        DeepSeek41Family.layer_entry(&target, layer)
    })
    .unwrap_err();
    assert!(matches!(err, Error::Geometry(_)), "unexpected error: {err:?}");
}
