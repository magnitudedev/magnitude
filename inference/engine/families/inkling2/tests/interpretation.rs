use magnitude_artifacts::{
    ArtifactIdentity, PackageIdentity,
    gguf::{ByteOrder, Directory, Encoding, Metadata, Scalar, TensorDescriptor, Value},
};
use magnitude_family_contracts::{
    AttentionSinks, BlockAttentionNorm, BlockAttentionQueries, BlockLayout, BlockTransition,
    CrossPassInput, DraftFusion, DraftMask, DraftMethod, EntryNorm, ExitNorm, FeedForwardUp,
    GluLimit, HeadNorm, HistoryRows, ImportTransform, InputNorm, MarkovHead, ModelFamily, Operator,
    OutputForm, ResidualForm, RowProducer, ScaleCodes, SublayerIndex, TapPoint,
};
use magnitude_family_dflash::inspect as inspect_dflash;
use magnitude_family_inkling2::{Inkling2Family, inspect_components};

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([42; 32]),
        projector: None,
    }
}

fn kv_str(name: &str, value: &str) -> Metadata {
    Metadata {
        name: name.into(),
        value: Value::Scalar(Scalar::String(value.into())),
    }
}

fn kv_u32(name: &str, value: u64) -> Metadata {
    Metadata {
        name: name.into(),
        value: Value::Scalar(Scalar::Unsigned(value)),
    }
}

fn kv_f32(name: &str, value: f64) -> Metadata {
    Metadata {
        name: name.into(),
        value: Value::Scalar(Scalar::Float(value)),
    }
}

fn kv_bool(name: &str, value: bool) -> Metadata {
    Metadata {
        name: name.into(),
        value: Value::Scalar(Scalar::Bool(value)),
    }
}

fn tensor(name: impl Into<String>, shape: Vec<u64>, encoding: Encoding) -> TensorDescriptor {
    TensorDescriptor {
        name: name.into(),
        shape,
        encoding,
        offset: 0,
        nbytes: 0,
    }
}

struct SyntheticSpec {
    phys_layers: u64,
    block_size: u64,
    hidden: u64,
    heads: u64,
    kv_heads: u64,
    head_dim: u64,
    experts: u64,
    selected: u64,
    inter: u64,
    mxfp8_start: u64,
    /// `(core_start, core_end, num_passes, gate_rank)`
    loop_cfg: Option<(u64, u64, u64, u64)>,
}

fn build_inkling2_directory(spec: &SyntheticSpec) -> Directory {
    let vocab_padded = 201_024u64;
    let vocab_unpadded = 200_058u64;
    let d_rel = 16u64;
    let rel_extent = 1024u64;
    let swa = 1024u64;
    let conv_k = 4u64;

    let visits: Vec<u64> = match spec.loop_cfg {
        None => (0..spec.phys_layers).collect(),
        Some((start, end, passes, _)) => {
            let mut v = Vec::new();
            v.extend(0..start);
            for _ in 0..passes {
                v.extend(start..end);
            }
            v.extend(end..spec.phys_layers);
            v
        }
    };
    let is_swa_phys: Vec<bool> = (0..spec.phys_layers)
        .map(|i| (i + 1) % spec.block_size != 0)
        .collect();
    let swa_pattern: Vec<bool> = visits.iter().map(|&p| is_swa_phys[p as usize]).collect();
    let heads_arr: Vec<u64> = vec![spec.heads; visits.len()];
    let kv_heads_arr: Vec<u64> = vec![spec.kv_heads; visits.len()];

    let mut metadata = vec![
        kv_str("general.architecture", "inkling2"),
        kv_u32("inkling2.block_count", visits.len() as u64),
        kv_u32("inkling2.context_length", 1_048_576),
        kv_u32("inkling2.embedding_length", spec.hidden),
        kv_u32("inkling2.feed_forward_length", spec.inter),
        Metadata {
            name: "inkling2.attention.head_count".into(),
            value: Value::Array(heads_arr.into()),
        },
        Metadata {
            name: "inkling2.attention.head_count_kv".into(),
            value: Value::Array(kv_heads_arr.into()),
        },
        kv_u32("inkling2.attention.key_length", spec.head_dim),
        kv_u32("inkling2.attention.value_length", spec.head_dim),
        kv_u32("inkling2.attention.key_length_swa", spec.head_dim),
        kv_u32("inkling2.attention.value_length_swa", spec.head_dim),
        kv_f32("inkling2.attention.layer_norm_rms_epsilon", 1e-6),
        kv_u32("inkling2.rope.dimension_count", 0),
        kv_u32("inkling2.expert_count", spec.experts),
        kv_u32("inkling2.expert_used_count", spec.selected),
        kv_u32("inkling2.expert_feed_forward_length", spec.inter),
        kv_f32("inkling2.expert_weights_scale", 8.0),
        kv_u32("inkling2.vocab_size", vocab_padded),
        kv_u32("inkling2.vocab_size_unpadded", vocab_unpadded),
        kv_u32("inkling2.attention.sliding_window", swa),
        Metadata {
            name: "inkling2.attention.sliding_window_pattern".into(),
            value: Value::Array(swa_pattern.into()),
        },
        kv_u32("inkling2.shortconv.l_cache", conv_k),
        kv_f32("inkling2.activation.situ_beta", 4.0),
        kv_f32("inkling2.activation.situ_linear_beta", 25.0),
        kv_u32("inkling2.attn_res.block_size", spec.block_size),
        kv_f32("inkling2.final_logit_softcapping", 30.0),
        kv_u32("inkling2.attention.relative_length", d_rel),
        kv_u32("inkling2.attention.relative_extent", rel_extent),
        kv_f32("inkling2.attention.qk_clip", 22.627417),
        kv_f32("inkling2.attention.relative_clip", 8.0),
        kv_u32("inkling2.attention.log_scale_floor", 8192),
        kv_f32("inkling2.attention.log_scale_alpha", 0.1),
    ];

    if let Some((start, end, passes, rank)) = spec.loop_cfg {
        metadata.push(kv_u32("inkling2.num_loops", passes));
        metadata.push(kv_u32("inkling2.loop.core_start", start));
        metadata.push(kv_u32("inkling2.loop.core_end", end));
        metadata.push(kv_u32("inkling2.loop.gate_rank", rank));
    }

    let h = spec.hidden;
    let q_rows = spec.heads * spec.head_dim;
    let kv_rows = spec.kv_heads * spec.head_dim;

    let mut tensors = vec![
        tensor("token_embd.weight", vec![vocab_padded, h], Encoding::BF16),
        tensor("token_embd_norm.weight", vec![h], Encoding::F32),
        tensor("output.weight", vec![vocab_padded, h], Encoding::BF16),
        tensor("output_norm.weight", vec![h], Encoding::F32),
        tensor("output_res_query.weight", vec![h], Encoding::F32),
    ];

    if let Some((start, end, _, _)) = spec.loop_cfg {
        let n_query = 2 * (end - start);
        tensors.push(tensor(
            "loop_res_query.weight",
            vec![n_query, h],
            Encoding::F32,
        ));
        tensors.push(tensor(
            "loop_kv_res_query.weight",
            vec![n_query, h],
            Encoding::F32,
        ));
    }

    for ip in 0..spec.phys_layers {
        let p = format!("blk.{ip}.");
        let swa_layer = is_swa_phys[ip as usize];
        let extent = if swa_layer { swa } else { rel_extent };

        tensors.push(tensor(format!("{p}attn_q.weight"), vec![q_rows, h], Encoding::BF16));
        tensors.push(tensor(format!("{p}attn_k.weight"), vec![kv_rows, h], Encoding::BF16));
        tensors.push(tensor(format!("{p}attn_v.weight"), vec![kv_rows, h], Encoding::BF16));
        tensors.push(tensor(format!("{p}attn_output.weight"), vec![h, q_rows], Encoding::BF16));
        tensors.push(tensor(format!("{p}attn_gate.weight"), vec![q_rows, h], Encoding::BF16));
        tensors.push(tensor(
            format!("{p}attn_q_clip.weight"),
            vec![spec.heads, spec.head_dim],
            Encoding::F32,
        ));
        tensors.push(tensor(
            format!("{p}attn_sink_proj.weight"),
            vec![spec.heads, h],
            Encoding::BF16,
        ));
        tensors.push(tensor(
            format!("{p}attn_rel.weight"),
            vec![spec.heads * d_rel, h],
            Encoding::BF16,
        ));
        tensors.push(tensor(
            format!("{p}attn_rel_bias.weight"),
            vec![extent, d_rel],
            Encoding::F32,
        ));
        tensors.push(tensor(
            format!("{p}attn_rel_clip.weight"),
            vec![spec.heads, d_rel],
            Encoding::F32,
        ));

        tensors.push(tensor(format!("{p}attn_k_conv.weight"), vec![kv_rows, conv_k], Encoding::F32));
        tensors.push(tensor(format!("{p}attn_v_conv.weight"), vec![kv_rows, conv_k], Encoding::F32));
        tensors.push(tensor(format!("{p}attn_post_conv.weight"), vec![h, conv_k], Encoding::F32));
        tensors.push(tensor(format!("{p}ffn_post_conv.weight"), vec![h, conv_k], Encoding::F32));

        if swa_layer {
            tensors.push(tensor(format!("{p}attn_norm.weight"), vec![h], Encoding::F32));
        } else {
            tensors.push(tensor(format!("{p}attn_q_inp_norm.weight"), vec![h], Encoding::F32));
            tensors.push(tensor(format!("{p}attn_k_inp_norm.weight"), vec![h], Encoding::F32));
            tensors.push(tensor(format!("{p}attn_v_inp_norm.weight"), vec![h], Encoding::F32));
            tensors.push(tensor(
                format!("{p}attn_k_conv_proj.weight"),
                vec![spec.kv_heads * conv_k, h],
                Encoding::BF16,
            ));
            tensors.push(tensor(
                format!("{p}attn_v_conv_proj.weight"),
                vec![spec.kv_heads * conv_k, h],
                Encoding::BF16,
            ));
        }

        if ip > 0 {
            let q_shape = if swa_layer { vec![1, h] } else { vec![3, h] };
            tensors.push(tensor(format!("{p}attn_res_query.weight"), q_shape, Encoding::F32));
        }
        tensors.push(tensor(format!("{p}ffn_res_query.weight"), vec![h], Encoding::F32));
        tensors.push(tensor(format!("{p}ffn_norm.weight"), vec![h], Encoding::F32));
        tensors.push(tensor(format!("{p}ffn_gate_inp.weight"), vec![spec.experts, h], Encoding::BF16));
        tensors.push(tensor(format!("{p}exp_probs_b.bias"), vec![spec.experts], Encoding::F32));

        if ip < spec.mxfp8_start {
            // Native NVFP4 + F32 per-expert scale.
            tensors.push(tensor(
                format!("{p}ffn_gate_up_exps.weight"),
                vec![spec.experts, 2 * spec.inter, h],
                Encoding::Nvfp4,
            ));
            tensors.push(tensor(
                format!("{p}ffn_gate_up_exps.scale"),
                vec![spec.experts],
                Encoding::F32,
            ));
            tensors.push(tensor(
                format!("{p}ffn_down_exps.weight"),
                vec![spec.experts, h, spec.inter],
                Encoding::Nvfp4,
            ));
            tensors.push(tensor(
                format!("{p}ffn_down_exps.scale"),
                vec![spec.experts],
                Encoding::F32,
            ));
        } else {
            // Native MXFP8 (F8E4M3 + I8 UE8M0 1x32 scale grid).
            tensors.push(tensor(
                format!("{p}ffn_gate_up_exps.weight"),
                vec![spec.experts, 2 * spec.inter, h],
                Encoding::F8E4M3,
            ));
            tensors.push(tensor(
                format!("{p}ffn_gate_up_exps.scale"),
                vec![spec.experts, 2 * spec.inter, h / 32],
                Encoding::I8,
            ));
            tensors.push(tensor(
                format!("{p}ffn_down_exps.weight"),
                vec![spec.experts, h, spec.inter],
                Encoding::F8E4M3,
            ));
            tensors.push(tensor(
                format!("{p}ffn_down_exps.scale"),
                vec![spec.experts, h, spec.inter / 32],
                Encoding::I8,
            ));
        }

        if let Some((start, end, _, rank)) = spec.loop_cfg {
            if ip >= start && ip < end {
                for m in ["attn", "k", "v", "mlp"] {
                    tensors.push(tensor(
                        format!("{p}loop_{m}_down.weight"),
                        vec![rank, 2 * h],
                        Encoding::BF16,
                    ));
                    tensors.push(tensor(
                        format!("{p}loop_{m}_gain.weight"),
                        vec![h, rank],
                        Encoding::BF16,
                    ));
                    tensors.push(tensor(
                        format!("{p}loop_{m}_forget.weight"),
                        vec![h, rank],
                        Encoding::BF16,
                    ));
                }
            }
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

#[test]
fn inkling2_small_non_looped_validates() {
    let dir = build_inkling2_directory(&SyntheticSpec {
        phys_layers: 48,
        block_size: 6,
        hidden: 4608,
        heads: 48,
        kv_heads: 8,
        head_dim: 128,
        experts: 384,
        selected: 8,
        inter: 2304,
        mxfp8_start: 40,
        loop_cfg: None,
    });
    assert!(Inkling2Family.recognizes(&dir));
    let def = inspect_components(&dir, None, identity()).expect("Inkling 2 Small validates");
    assert_eq!(def.decoder.blocks.len(), 48);
    assert_eq!(def.decoder.vocabulary, 201_024);
    assert_eq!(def.decoder.active_vocabulary(), 200_058);
    assert!(matches!(
        def.decoder.residual,
        ResidualForm::BlockAttention {
            block_size: 6,
            normalize_values: true,
            ..
        }
    ));
    assert!(matches!(
        def.decoder.entry.norm,
        Some(EntryNorm::Rms { text_only: true, .. })
    ));
    assert!(matches!(def.decoder.exit.norm, ExitNorm::BlockAttention { .. }));
    assert_eq!(def.decoder.exit.softcap, Some(30.0));

    // Check BlockCommit at ip % 6 == 0, local vs global attention, and mixed NVFP4/MXFP8.
    for (idx, block) in def.decoder.blocks.iter().enumerate() {
        assert_eq!(block.transition, None);
        let expected_out = if idx % 6 == 0 {
            OutputForm::BlockCommit
        } else {
            OutputForm::Residual
        };
        assert_eq!(block.sublayers[0].output, expected_out);
        let is_global = (idx + 1) % 6 == 0;
        let InputNorm::BlockAttention(ba) = &block.sublayers[0].input else {
            panic!("expected BlockAttention input");
        };
        if idx == 0 {
            assert!(matches!(ba.queries, BlockAttentionQueries::Partial));
        } else if is_global {
            assert!(matches!(ba.queries, BlockAttentionQueries::SplitQkv { .. }));
            assert!(matches!(ba.norm, BlockAttentionNorm::SplitQkv { .. }));
        } else {
            assert!(matches!(ba.queries, BlockAttentionQueries::Single(_)));
            assert!(matches!(ba.norm, BlockAttentionNorm::Shared(_)));
        }
        let Operator::Attention(attn) = &block.sublayers[0].op else {
            panic!("expected Attention");
        };
        assert!(attn.relative_bias.is_some());
        assert_eq!(attn.log_scale.is_some(), is_global);
        assert!(matches!(attn.sinks, AttentionSinks::Dynamic(_)));
        assert!(matches!(attn.query_norm, HeadNorm::L2Clip { scale: Some(_), .. }));
        let HistoryRows::Own(RowProducer::Projected(rows)) = &attn.history[0].rows else {
            panic!("expected ProjectedRows");
        };
        assert_eq!(rows.key_conv.as_ref().unwrap().dynamic.is_some(), is_global);
        assert_eq!(rows.value_conv.as_ref().unwrap().dynamic.is_some(), is_global);

        let Operator::RoutedFfn(moe) = &block.sublayers[1].op else {
            panic!("expected RoutedFfn");
        };
        assert!(matches!(
            moe.expert_up,
            FeedForwardUp::Gated {
                limit: Some(GluLimit::Softcap { .. }),
                ..
            }
        ));
        if idx < 40 {
            assert!(matches!(
                moe.expert_down.transforms.as_slice(),
                [ImportTransform::ScaleByTensor { .. }]
            ));
        } else {
            assert!(matches!(
                moe.expert_down.transforms.as_slice(),
                [ImportTransform::ScaleBlocks {
                    codes: ScaleCodes::Ue8m0,
                    tile: [1, 32],
                    ..
                }]
            ));
        }
    }
}

#[test]
fn inkling2_mini_looped_unrolls_and_shares_weights() {
    let dir = build_inkling2_directory(&SyntheticSpec {
        phys_layers: 24,
        block_size: 6,
        hidden: 2304,
        heads: 24,
        kv_heads: 8,
        head_dim: 128,
        experts: 128,
        selected: 8,
        inter: 1152,
        mxfp8_start: 20,
        loop_cfg: Some((6, 18, 3, 64)),
    });
    let def = inspect_components(&dir, None, identity()).expect("Inkling 2 Mini validates");
    assert_eq!(def.decoder.blocks.len(), 48);

    // Transitions: SnapshotPrelude at visit 6 (start of pass 0),
    // ResetToPreludeAndBank at visits 18 (pass 1) and 30 (pass 2).
    assert_eq!(def.decoder.blocks[6].transition, Some(BlockTransition::SnapshotPrelude));
    assert_eq!(def.decoder.blocks[18].transition, Some(BlockTransition::ResetToPreludeAndBank));
    assert_eq!(def.decoder.blocks[30].transition, Some(BlockTransition::ResetToPreludeAndBank));
    for il in 0..48 {
        if ![6, 18, 30].contains(&il) {
            assert_eq!(def.decoder.blocks[il].transition, None, "visit {il}");
        }
    }

    // Core layer 6 is visited at logical indices 6 (pass 0), 18 (pass 1), 30 (pass 2).
    // All 3 visits must reuse identical WeightDescriptors!
    for offset in 0..12 {
        let v0 = &def.decoder.blocks[6 + offset];
        let v1 = &def.decoder.blocks[18 + offset];
        let v2 = &def.decoder.blocks[30 + offset];
        assert_eq!(v0.sublayers[0].op, v1.sublayers[0].op);
        assert_eq!(v0.sublayers[0].op, v2.sublayers[0].op);
        assert_eq!(v0.sublayers[1].op, v1.sublayers[1].op);
        assert_eq!(v0.sublayers[1].op, v2.sublayers[1].op);

        // Pass 0 has no cross-pass messages; passes 1 & 2 have SplitQkv on attn and Single on MoE,
        // even on local SWA layers!
        let InputNorm::BlockAttention(a0) = &v0.sublayers[0].input else { panic!() };
        let InputNorm::BlockAttention(a1) = &v1.sublayers[0].input else { panic!() };
        let InputNorm::BlockAttention(a2) = &v2.sublayers[0].input else { panic!() };
        assert!(matches!(a0.cross_pass, CrossPassInput::None));
        assert!(matches!(a1.cross_pass, CrossPassInput::SplitQkv { .. }));
        assert!(matches!(a2.cross_pass, CrossPassInput::SplitQkv { .. }));

        let InputNorm::BlockAttention(m0) = &v0.sublayers[1].input else { panic!() };
        let InputNorm::BlockAttention(m1) = &v1.sublayers[1].input else { panic!() };
        assert!(matches!(m0.cross_pass, CrossPassInput::None));
        assert!(matches!(m1.cross_pass, CrossPassInput::Single(_)));
    }

    // Also verify DSpark targeting Inkling 2 Mini!
    let h = 2304u64;
    let draft_layers = 3u64;
    let draft_heads = 24u64;
    let draft_kv_heads = 8u64;
    let draft_head_dim = 128u64;
    let draft_inter = 6144u64;
    let markov_rank = 256u64;
    let target_layers: Vec<u64> = vec![5, 23, 41, 48];
    let n_taps = target_layers.len() as u64;

    let mut d_tensors = vec![
        tensor("fuse_weight.weight", vec![draft_layers + 1, n_taps], Encoding::BF16),
        tensor("mask_embd.weight", vec![h], Encoding::BF16),
        tensor("output_norm.weight", vec![h], Encoding::BF16),
        tensor("markov_w1.weight", vec![201_024, markov_rank], Encoding::BF16),
        tensor(
            "markov_rnn_proj.weight",
            vec![3 * markov_rank, 2 * markov_rank + 2 * h],
            Encoding::BF16,
        ),
        tensor("markov_rnn_proj.bias", vec![3 * markov_rank], Encoding::BF16),
        tensor("markov_hidden_out.weight", vec![h, markov_rank], Encoding::BF16),
        tensor("markov_hidden_out.bias", vec![h], Encoding::BF16),
        tensor("markov_dflare_norm.weight", vec![h], Encoding::BF16),
    ];
    for i in 0..draft_layers {
        let p = format!("blk.{i}.");
        let q_rows = draft_heads * draft_head_dim;
        let kv_rows = draft_kv_heads * draft_head_dim;
        d_tensors.extend([
            tensor(format!("{p}attn_norm.weight"), vec![h], Encoding::BF16),
            tensor(format!("{p}dflare_norm.weight"), vec![h], Encoding::BF16),
            tensor(format!("{p}attn_q.weight"), vec![q_rows, h], Encoding::BF16),
            tensor(format!("{p}attn_k.weight"), vec![kv_rows, h], Encoding::BF16),
            tensor(format!("{p}attn_v.weight"), vec![kv_rows, h], Encoding::BF16),
            tensor(format!("{p}attn_output.weight"), vec![h, q_rows], Encoding::BF16),
            tensor(format!("{p}attn_q_norm.weight"), vec![draft_head_dim], Encoding::BF16),
            tensor(format!("{p}attn_k_norm.weight"), vec![draft_head_dim], Encoding::BF16),
            tensor(format!("{p}attn_k_ctx.weight"), vec![kv_rows, h], Encoding::BF16),
            tensor(format!("{p}attn_v_ctx.weight"), vec![kv_rows, h], Encoding::BF16),
            tensor(format!("{p}attn_k_norm_ctx.weight"), vec![draft_head_dim], Encoding::BF16),
            tensor(format!("{p}ffn_norm.weight"), vec![h], Encoding::BF16),
            tensor(format!("{p}ffn_gate.weight"), vec![draft_inter, h], Encoding::BF16),
            tensor(format!("{p}ffn_up.weight"), vec![draft_inter, h], Encoding::BF16),
            tensor(format!("{p}ffn_down.weight"), vec![h, draft_inter], Encoding::BF16),
        ]);
    }
    let dspark_dir = Directory {
        version: 3,
        byte_order: ByteOrder::Little,
        alignment: 32,
        data_offset: 0,
        metadata: vec![
            kv_str("general.architecture", "dflash"),
            kv_u32("dflash.block_count", draft_layers),
            kv_u32("dflash.context_length", 1_048_576),
            kv_u32("dflash.embedding_length", h),
            kv_u32("dflash.feed_forward_length", draft_inter),
            kv_u32("dflash.attention.head_count", draft_heads),
            kv_u32("dflash.attention.head_count_kv", draft_kv_heads),
            kv_u32("dflash.attention.key_length", draft_head_dim),
            kv_u32("dflash.attention.value_length", draft_head_dim),
            kv_f32("dflash.attention.layer_norm_rms_epsilon", 1e-6),
            kv_bool("dflash.attention.causal", false),
            kv_u32("dflash.rope.dimension_count", draft_head_dim),
            kv_f32("dflash.rope.freq_base", 10_000.0),
            kv_u32("dflash.block_size", 8),
            Metadata {
                name: "dflash.target_layers".into(),
                value: Value::Array(target_layers.into()),
            },
            kv_bool("dflash.sample_from_anchor", false),
            kv_u32("dflash.markov_rank", markov_rank),
            kv_u32("dflash.vocab_size", 201_024),
            kv_u32("dflash.vocab_size_unpadded", 200_058),
            kv_u32("dflash.attention.sliding_window", 1024),
            Metadata {
                name: "dflash.attention.sliding_window_pattern".into(),
                value: Value::Array(vec![true, true, false].into()),
            },
            kv_f32("dflash.final_logit_softcapping", 30.0),
        ],
        tensors: d_tensors,
    };

    let draft = inspect_dflash(&dspark_dir, &def, &|layer| {
        Inkling2Family.layer_entry(&def, layer)
    })
    .expect("Inkling 2 DSpark validates");
    assert_eq!(draft.blocks.len(), 3);
    assert_eq!(draft.block_size, 8);
    assert_eq!(draft.layout, BlockLayout::MaskSlots);
    assert_eq!(
        draft.taps,
        vec![
            TapPoint::Sublayer(SublayerIndex { block: 5, sublayer: 0 }),
            TapPoint::Sublayer(SublayerIndex { block: 23, sublayer: 0 }),
            TapPoint::Sublayer(SublayerIndex { block: 41, sublayer: 0 }),
            TapPoint::Exit,
        ]
    );
    assert!(matches!(draft.mask, DraftMask::Learned(_)));
    assert!(matches!(
        &draft.fusion,
        DraftFusion::SoftmaxMix { norms, .. } if norms.len() == 4
    ));
    assert_eq!(draft.context_projections.as_ref().map(Vec::len), Some(3));
    assert!(matches!(
        draft.method,
        DraftMethod::DSpark {
            markov: MarkovHead::RnnHidden(_),
            confidence: None,
        }
    ));
}

#[test]
fn inkling2_vision_projector_validates() {
    let depth = 23u64;
    let vh = 1024u64;
    let vinter = 4096u64;
    let heads = 16u64;
    let patch = 20u64;
    let merge = 2u64;
    let side = 32u64;
    let out_h = 2304u64;

    let mut tensors = vec![
        tensor("v.patch_embd.weight", vec![vh, 3, patch, patch], Encoding::BF16),
        tensor("v.position_embd.weight", vec![side * side, vh], Encoding::BF16),
        tensor("v.class_embd", vec![1, vh], Encoding::BF16),
        tensor("v.class_position_embd.weight", vec![1, vh], Encoding::BF16),
        tensor("v.pre_ln.weight", vec![vh], Encoding::F32),
        tensor("v.pre_ln.bias", vec![vh], Encoding::F32),
        tensor("mm.1.weight", vec![vinter, merge * merge * vh], Encoding::BF16),
        tensor("mm.1.bias", vec![vinter], Encoding::BF16),
        tensor("mm.2.weight", vec![out_h, vinter], Encoding::BF16),
        tensor("mm.2.bias", vec![out_h], Encoding::BF16),
    ];
    for i in 0..depth {
        let p = format!("v.blk.{i}.");
        tensors.extend([
            tensor(format!("{p}ln1.weight"), vec![vh], Encoding::F32),
            tensor(format!("{p}ln1.bias"), vec![vh], Encoding::F32),
            tensor(format!("{p}attn_q.weight"), vec![vh, vh], Encoding::BF16),
            tensor(format!("{p}attn_q.bias"), vec![vh], Encoding::BF16),
            tensor(format!("{p}attn_k.weight"), vec![vh, vh], Encoding::BF16),
            tensor(format!("{p}attn_k.bias"), vec![vh], Encoding::BF16),
            tensor(format!("{p}attn_v.weight"), vec![vh, vh], Encoding::BF16),
            tensor(format!("{p}attn_v.bias"), vec![vh], Encoding::BF16),
            tensor(format!("{p}attn_out.weight"), vec![vh, vh], Encoding::BF16),
            tensor(format!("{p}attn_out.bias"), vec![vh], Encoding::BF16),
            tensor(format!("{p}ls1.weight"), vec![vh], Encoding::F32),
            tensor(format!("{p}ln2.weight"), vec![vh], Encoding::F32),
            tensor(format!("{p}ln2.bias"), vec![vh], Encoding::F32),
            tensor(format!("{p}ffn_up.weight"), vec![vinter, vh], Encoding::BF16),
            tensor(format!("{p}ffn_up.bias"), vec![vinter], Encoding::BF16),
            tensor(format!("{p}ffn_down.weight"), vec![vh, vinter], Encoding::BF16),
            tensor(format!("{p}ffn_down.bias"), vec![vh], Encoding::BF16),
            tensor(format!("{p}ls2.weight"), vec![vh], Encoding::F32),
        ]);
    }
    let proj_dir = Directory {
        version: 3,
        byte_order: ByteOrder::Little,
        alignment: 32,
        data_offset: 0,
        metadata: vec![
            kv_str("general.architecture", "clip"),
            kv_str("general.type", "mmproj"),
            kv_bool("clip.has_vision_encoder", true),
            kv_str("clip.projector_type", "inkling2v"),
            kv_u32("clip.vision.projection_dim", out_h),
            kv_u32("clip.vision.image_size", side * patch),
            kv_u32("clip.vision.patch_size", patch),
            kv_u32("clip.vision.embedding_length", vh),
            kv_u32("clip.vision.feed_forward_length", vinter),
            kv_u32("clip.vision.block_count", depth),
            kv_u32("clip.vision.attention.head_count", heads),
            Metadata {
                name: "clip.vision.image_mean".into(),
                value: Value::Array(vec![0.5f64, 0.5, 0.5].into()),
            },
            Metadata {
                name: "clip.vision.image_std".into(),
                value: Value::Array(vec![0.5f64, 0.5, 0.5].into()),
            },
            kv_f32("clip.vision.attention.layer_norm_epsilon", 1e-6),
            kv_u32("clip.vision.spatial_merge_size", merge),
        ],
        tensors,
    };

    let vision = magnitude_family_inkling2::projector::describe_projector(&proj_dir, out_h)
        .expect("Inkling 2 vision projector validates");
    assert_eq!(vision.blocks.len(), 23);
    assert!(vision.blocks.iter().all(|b| b.ls1.is_some() && b.ls2.is_some()));
}
