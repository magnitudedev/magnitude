use magnitude_artifacts::{
    gguf::{ByteOrder, Directory, Encoding, Metadata, Scalar, TensorDescriptor, Value},
    ArtifactIdentity, PackageIdentity,
};
use magnitude_artifacts::{
    media::{DType, PreparedMedia, PreparedTensor},
    ImageProcessor, InputLayout, TokenId,
};
use magnitude_model_contracts::{
    FeedForwardGeometry, FeedForwardWeights, MixerGeometry, MixerWeights, ModelInputAdapter,
    RecurrentHeadMapping, TokenPlan,
};
use magnitude_model_qwen35::inputs::{QwenImageTokens, QwenInputAdapter};
use magnitude_model_qwen35::{describe_projector, inspect_components, recognize, Architecture};

fn directory(routed: bool) -> Directory {
    let architecture = if routed { "qwen35moe" } else { "qwen35" };
    let mut metadata = vec![Metadata {
        name: "general.architecture".into(),
        value: Value::Scalar(Scalar::String(architecture.into())),
    }];
    for (name, value) in [
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
            value: Value::Scalar(Scalar::Unsigned(value)),
        });
    }
    for (name, value) in [
        ("rope.freq_base", 10_000.0),
        ("attention.layer_norm_rms_epsilon", 1e-6),
    ] {
        metadata.push(Metadata {
            name: format!("{architecture}.{name}"),
            value: Value::Scalar(Scalar::Float(value)),
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
        tensors.push(TensorDescriptor {
            name,
            shape: shape.into(),
            encoding: Encoding::F32,
            offset: 0,
            nbytes: shape.iter().product::<u64>() * 4,
        });
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
    for block in 0..2 {
        add(format!("blk.{block}.attn_norm.weight"), &[8]);
        add(format!("blk.{block}.post_attention_norm.weight"), &[8]);
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
            add(format!("blk.{block}.{name}"), &shape);
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

fn mtp_directory() -> Directory {
    let mut directory = directory(false);
    let block_count = directory
        .metadata
        .iter_mut()
        .find(|metadata| metadata.name == "qwen35.block_count")
        .unwrap();
    block_count.value = Value::Scalar(Scalar::Unsigned(3));
    directory.metadata.push(Metadata {
        name: "qwen35.nextn_predict_layers".into(),
        value: Value::Scalar(Scalar::Unsigned(1)),
    });
    let mut add = |name: &str, shape: &[u64]| {
        directory.tensors.push(TensorDescriptor {
            name: format!("blk.2.{name}"),
            shape: shape.into(),
            encoding: Encoding::F32,
            offset: 0,
            nbytes: shape.iter().product::<u64>() * 4,
        });
    };
    for (name, shape) in [
        ("nextn.enorm.weight", vec![8]),
        ("nextn.hnorm.weight", vec![8]),
        ("nextn.eh_proj.weight", vec![8, 16]),
        ("attn_norm.weight", vec![8]),
        ("attn_q.weight", vec![16, 8]),
        ("attn_k.weight", vec![4, 8]),
        ("attn_v.weight", vec![4, 8]),
        ("attn_q_norm.weight", vec![4]),
        ("attn_k_norm.weight", vec![4]),
        ("attn_output.weight", vec![8, 8]),
        ("post_attention_norm.weight", vec![8]),
        ("ffn_gate.weight", vec![12, 8]),
        ("ffn_up.weight", vec![12, 8]),
        ("ffn_down.weight", vec![8, 12]),
        ("nextn.shared_head_norm.weight", vec![8]),
    ] {
        add(name, &shape);
    }
    directory
}

fn routed_mtp_directory() -> Directory {
    let mut directory = directory(true);
    directory
        .metadata
        .retain(|item| item.name != "qwen35moe.feed_forward_length");
    directory
        .metadata
        .iter_mut()
        .find(|item| item.name == "qwen35moe.block_count")
        .unwrap()
        .value = Value::Scalar(Scalar::Unsigned(3));
    directory.metadata.push(Metadata {
        name: "qwen35moe.nextn_predict_layers".into(),
        value: Value::Scalar(Scalar::Unsigned(1)),
    });
    for (name, shape) in [
        ("nextn.enorm.weight", vec![8]),
        ("nextn.hnorm.weight", vec![8]),
        ("nextn.eh_proj.weight", vec![8, 16]),
        ("attn_norm.weight", vec![8]),
        ("attn_q.weight", vec![16, 8]),
        ("attn_k.weight", vec![4, 8]),
        ("attn_v.weight", vec![4, 8]),
        ("attn_q_norm.weight", vec![4]),
        ("attn_k_norm.weight", vec![4]),
        ("attn_output.weight", vec![8, 8]),
        ("post_attention_norm.weight", vec![8]),
        ("ffn_gate_inp.weight", vec![4, 8]),
        ("ffn_gate_inp_shexp.weight", vec![8]),
        ("ffn_gate_exps.weight", vec![4, 6, 8]),
        ("ffn_up_exps.weight", vec![4, 6, 8]),
        ("ffn_down_exps.weight", vec![4, 8, 6]),
        ("ffn_gate_shexp.weight", vec![8, 8]),
        ("ffn_up_shexp.weight", vec![8, 8]),
        ("ffn_down_shexp.weight", vec![8, 8]),
        ("nextn.shared_head_norm.weight", vec![8]),
    ] {
        directory.tensors.push(TensorDescriptor {
            name: format!("blk.2.{name}"),
            shape: shape.clone(),
            encoding: Encoding::F32,
            offset: 0,
            nbytes: shape.iter().product::<u64>() * 4,
        });
    }
    directory
}

#[test]
fn routed_mtp_head_uses_expert_roles_without_dense_intermediate_metadata() {
    let model = inspect_components(
        &routed_mtp_directory(),
        None,
        PackageIdentity {
            target: ArtifactIdentity([9; 32]),
            projector: None,
        },
    )
    .unwrap();
    assert_eq!(model.geometry.blocks.len(), 2);
    let head = model.head.unwrap();
    assert_eq!(head.depth(), 1);
    assert!(matches!(
        head.blocks[0].feedforward,
        FeedForwardWeights::Routed(_)
    ));
    assert!(matches!(
        head.blocks[0].feedforward_geometry,
        FeedForwardGeometry::Routed(_)
    ));
}

fn projector() -> Directory {
    let scalar = |name: &str, value: Scalar| Metadata {
        name: name.into(),
        value: Value::Scalar(value),
    };
    let mut metadata = vec![
        scalar("general.architecture", Scalar::String("clip".into())),
        scalar("general.type", Scalar::String("mmproj".into())),
        scalar(
            "general.base_model.0.name",
            Scalar::String("fixture".into()),
        ),
        scalar("clip.has_vision_encoder", Scalar::Bool(true)),
        scalar(
            "clip.projector_type",
            Scalar::String("qwen3vl_merger".into()),
        ),
        scalar("clip.vision.block_count", Scalar::Unsigned(1)),
        scalar("clip.vision.embedding_length", Scalar::Unsigned(4)),
        scalar("clip.vision.feed_forward_length", Scalar::Unsigned(8)),
        scalar("clip.vision.attention.head_count", Scalar::Unsigned(2)),
        scalar("clip.vision.patch_size", Scalar::Unsigned(2)),
        scalar("clip.vision.spatial_merge_size", Scalar::Unsigned(2)),
        scalar("clip.vision.image_size", Scalar::Unsigned(4)),
        scalar("clip.vision.projection_dim", Scalar::Unsigned(6)),
        scalar(
            "clip.vision.attention.layer_norm_epsilon",
            Scalar::Float(1e-6),
        ),
        scalar("clip.use_gelu", Scalar::Bool(true)),
        Metadata {
            name: "clip.vision.image_mean".into(),
            value: Value::Array(vec![Scalar::Float(0.5); 3]),
        },
        Metadata {
            name: "clip.vision.image_std".into(),
            value: Value::Array(vec![Scalar::Float(0.5); 3]),
        },
        Metadata {
            name: "clip.vision.is_deepstack_layers".into(),
            value: Value::Array(vec![Scalar::Bool(false)]),
        },
    ];
    metadata.shrink_to_fit();
    let mut tensors = Vec::new();
    let mut add = |name: &str, shape: &[u64]| {
        tensors.push(TensorDescriptor {
            name: name.into(),
            shape: shape.into(),
            encoding: Encoding::F32,
            offset: 0,
            nbytes: shape.iter().product::<u64>() * 4,
        });
    };
    for (name, shape) in [
        ("v.patch_embd.weight", vec![4, 3, 2, 2]),
        ("v.patch_embd.weight.1", vec![4, 3, 2, 2]),
        ("v.patch_embd.bias", vec![4]),
        ("v.position_embd.weight", vec![4, 4]),
        ("v.blk.0.ln1.weight", vec![4]),
        ("v.blk.0.ln1.bias", vec![4]),
        ("v.blk.0.attn_qkv.weight", vec![12, 4]),
        ("v.blk.0.attn_qkv.bias", vec![12]),
        ("v.blk.0.attn_out.weight", vec![4, 4]),
        ("v.blk.0.attn_out.bias", vec![4]),
        ("v.blk.0.ln2.weight", vec![4]),
        ("v.blk.0.ln2.bias", vec![4]),
        ("v.blk.0.ffn_up.weight", vec![8, 4]),
        ("v.blk.0.ffn_up.bias", vec![8]),
        ("v.blk.0.ffn_down.weight", vec![4, 8]),
        ("v.blk.0.ffn_down.bias", vec![4]),
        ("v.post_ln.weight", vec![4]),
        ("v.post_ln.bias", vec![4]),
        ("mm.0.weight", vec![16, 16]),
        ("mm.0.bias", vec![16]),
        ("mm.2.weight", vec![6, 16]),
        ("mm.2.bias", vec![6]),
    ] {
        add(name, &shape);
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
fn recognizes_only_qwen35_architectures() {
    assert_eq!(recognize(&directory(false)).unwrap(), Architecture::Dense);
    assert_eq!(recognize(&directory(true)).unwrap(), Architecture::Routed);
    let mut unsupported = directory(false);
    unsupported.metadata[0].value = Value::Scalar(Scalar::String("llama".into()));
    assert!(recognize(&unsupported).is_err());
}

#[test]
fn binds_dense_and_routed_geometry_to_semantic_roles() {
    for routed in [false, true] {
        let model = inspect_components(
            &directory(routed),
            None,
            PackageIdentity {
                target: ArtifactIdentity([0; 32]),
                projector: None,
            },
        )
        .unwrap();
        let MixerGeometry::Recurrent(recurrent) = &model.geometry.blocks[0].mixer else {
            panic!("first block must be recurrent")
        };
        assert_eq!(recurrent.channels().unwrap(), 16);
        assert_eq!(recurrent.head_mapping, RecurrentHeadMapping::Tiled);
        assert_eq!(model.output, model.embedding);
        assert!(matches!(&model.blocks[0].mixer, MixerWeights::Recurrent(_)));
        assert!(matches!(&model.blocks[1].mixer, MixerWeights::Attention(_)));
        assert_eq!(
            matches!(&model.blocks[0].feedforward, FeedForwardWeights::Routed(_)),
            routed
        );
        assert_eq!(
            matches!(
                &model.geometry.blocks[0].feedforward,
                FeedForwardGeometry::Routed(_)
            ),
            routed
        );
    }
}

#[test]
fn rejects_wrong_shapes_and_unbound_family_roles() {
    let mut wrong_shape = directory(false);
    wrong_shape.tensors[2].shape[0] += 1;
    assert!(inspect_components(
        &wrong_shape,
        None,
        PackageIdentity {
            target: ArtifactIdentity([0; 32]),
            projector: None
        }
    )
    .is_err());

    let mut unbound = directory(false);
    let mut extra = unbound.tensors[2].clone();
    extra.name = "blk.0.unbound.weight".into();
    unbound.tensors.push(extra);
    assert!(inspect_components(
        &unbound,
        None,
        PackageIdentity {
            target: ArtifactIdentity([0; 32]),
            projector: None
        }
    )
    .is_err());
}

#[test]
fn binds_every_mtp_role_outside_the_target_geometry() {
    let model = inspect_components(
        &mtp_directory(),
        None,
        PackageIdentity {
            target: ArtifactIdentity([3; 32]),
            projector: None,
        },
    )
    .unwrap();
    assert_eq!(model.geometry.blocks.len(), 2);
    let head = model.head.unwrap();
    assert_eq!(head.depth(), 1);
    assert_eq!(head.blocks[0].combine.shape, [8, 16]);

    let mut malformed = mtp_directory();
    malformed
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "blk.2.nextn.eh_proj.weight")
        .unwrap()
        .shape[1] = 15;
    assert!(inspect_components(
        &malformed,
        None,
        PackageIdentity {
            target: ArtifactIdentity([3; 32]),
            projector: None
        }
    )
    .is_err());

    let mut unqualified_depth = mtp_directory();
    unqualified_depth
        .metadata
        .iter_mut()
        .find(|metadata| metadata.name == "qwen35.nextn_predict_layers")
        .unwrap()
        .value = Value::Scalar(Scalar::Unsigned(2));
    assert!(inspect_components(
        &unqualified_depth,
        None,
        PackageIdentity {
            target: ArtifactIdentity([3; 32]),
            projector: None
        }
    )
    .is_err());
}

#[test]
fn projector_description_validates_roles_and_fused_qkv_ranges() {
    let projector = projector();
    let vision = describe_projector(&projector).unwrap();
    assert_eq!(vision.geometry.table_side, 2);
    assert_eq!(vision.geometry.temporal_patch, 2);
    assert_eq!(vision.preprocessing.processor, "qwen3vl_merger");
    assert_eq!(vision.preprocessing.min_pixels, 65_536);
    assert_eq!(vision.preprocessing.max_pixels, 16_777_216);
    let processor = vision.image_processor_config().unwrap();
    assert_eq!(processor.patch, 2);
    assert_eq!(processor.merge, 2);
    assert_eq!(processor.temporal_patch, 2);
    assert_eq!(vision.blocks[0].attention.qkv.query.start, 0);
    assert_eq!(vision.blocks[0].attention.qkv.key.start, 4);
    assert_eq!(vision.blocks[0].attention.qkv.value.start, 8);

    let mut target = directory(false);
    target.metadata.push(Metadata {
        name: "general.name".into(),
        value: Value::Scalar(Scalar::String("fixture".into())),
    });
    let model = inspect_components(
        &target,
        Some(&projector),
        PackageIdentity {
            target: ArtifactIdentity([1; 32]),
            projector: Some(ArtifactIdentity([2; 32])),
        },
    )
    .unwrap();
    assert!(model.vision.is_some());
    assert_eq!(
        model.artifact_identity.projector,
        Some(ArtifactIdentity([2; 32]))
    );

    // A derived target (a quantization) names its own file and its base
    // model; the projector pairs with the base model.
    let mut derived = directory(false);
    for (name, value) in [
        ("general.name", "fixture-Q4_K_M"),
        ("general.base_model.0.name", "fixture"),
    ] {
        derived.metadata.push(Metadata {
            name: name.into(),
            value: Value::Scalar(Scalar::String(value.into())),
        });
    }
    let identity = PackageIdentity {
        target: ArtifactIdentity([1; 32]),
        projector: Some(ArtifactIdentity([2; 32])),
    };
    assert!(inspect_components(&derived, Some(&projector), identity.clone()).is_ok());
    derived.metadata.pop();
    derived.metadata.push(Metadata {
        name: "general.base_model.0.name".into(),
        value: Value::Scalar(Scalar::String("other".into())),
    });
    assert!(inspect_components(&derived, Some(&projector), identity).is_err());

    let mut malformed = projector;
    malformed.tensors.push(TensorDescriptor {
        name: "v.unbound.weight".into(),
        shape: vec![1],
        encoding: Encoding::F32,
        offset: 0,
        nbytes: 4,
    });
    assert!(describe_projector(&malformed).is_err());
}

#[test]
fn projector_pixel_bounds_override_processor_defaults() {
    let mut projector = projector();
    projector.metadata.extend([
        Metadata {
            name: "clip.vision.image_min_pixels".into(),
            value: Value::Scalar(Scalar::Unsigned(1_024)),
        },
        Metadata {
            name: "clip.vision.image_max_pixels".into(),
            value: Value::Scalar(Scalar::Unsigned(4_096)),
        },
    ]);
    let vision = describe_projector(&projector).unwrap();
    assert_eq!(vision.preprocessing.min_pixels, 1_024);
    assert_eq!(vision.preprocessing.max_pixels, 4_096);
}

#[test]
fn projector_deepstack_metadata_is_optional_but_rejects_enabled_layers() {
    let mut without_deepstack = projector();
    without_deepstack
        .metadata
        .retain(|metadata| metadata.name != "clip.vision.is_deepstack_layers");
    assert!(describe_projector(&without_deepstack).is_ok());

    let mut enabled = projector();
    enabled
        .metadata
        .iter_mut()
        .find(|metadata| metadata.name == "clip.vision.is_deepstack_layers")
        .unwrap()
        .value = Value::Array(vec![Scalar::Bool(true)]);
    assert!(describe_projector(&enabled).is_err());
}

#[test]
fn projector_rejects_malformed_shapes_for_every_role_class() {
    for name in [
        "v.patch_embd.weight",
        "v.patch_embd.bias",
        "v.position_embd.weight",
        "v.blk.0.ln1.weight",
        "v.blk.0.attn_qkv.weight",
        "v.blk.0.attn_out.weight",
        "v.blk.0.ln2.bias",
        "v.blk.0.ffn_up.weight",
        "v.blk.0.ffn_down.weight",
        "v.post_ln.weight",
        "mm.0.weight",
        "mm.2.weight",
    ] {
        let mut malformed = projector();
        malformed
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == name)
            .unwrap()
            .shape[0] += 1;
        assert!(
            describe_projector(&malformed).is_err(),
            "malformed projector role {name:?} was accepted"
        );
    }
}

fn prepared_qwen_media(model: &magnitude_model_contracts::ModelDefinition) -> PreparedMedia {
    let vision = model.vision.as_ref().unwrap();
    let processor = ImageProcessor::new(vision.image_processor_config().unwrap()).unwrap();
    PreparedMedia::new(
        processor.identity().into(),
        vec![
            PreparedTensor::new(
                "pixel_values".into(),
                DType::F32,
                vec![16, 24],
                vec![0; 16 * 24 * 4],
            )
            .unwrap(),
            PreparedTensor::new(
                "image_grid_thw".into(),
                DType::I64,
                vec![1, 3],
                [1i64, 4, 4]
                    .into_iter()
                    .flat_map(i64::to_le_bytes)
                    .collect(),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

#[test]
fn qwen_adapter_closes_token_and_vision_coordinates() {
    let model = inspect_components(
        &directory(false),
        Some(&projector()),
        PackageIdentity {
            target: ArtifactIdentity([1; 32]),
            projector: Some(ArtifactIdentity([2; 32])),
        },
    )
    .unwrap();
    let adapter = QwenInputAdapter::new(Some(
        QwenImageTokens::new(TokenId(99), TokenId(98), TokenId(100)).unwrap(),
    ));
    let tokens = vec![TokenId(98), TokenId(99), TokenId(100), TokenId(7)];
    let plan = TokenPlan::new(
        tokens.clone(),
        InputLayout::new(tokens.len(), vec![]).unwrap(),
    )
    .unwrap();
    let media = prepared_qwen_media(&model);
    let input = adapter.prepare(&model, plan, &[media.clone()]).unwrap();
    assert_eq!(input.tokens().len(), 7);
    assert_eq!(input.layout().spans().len(), 1);
    assert_eq!(
        (
            input.layout().spans()[0].start,
            input.layout().spans()[0].end
        ),
        (1, 5)
    );
    assert_eq!(
        input.coordinates_at(0, 7).unwrap(),
        vec![
            [0, 0, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 1],
            [1, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
        ]
    );
    assert_eq!(input.continuation(), 5);
    assert_eq!(input.vision()[0].grid(), [1, 4, 4]);
    assert_eq!(input.vision()[0].pixels().shape(), [16, 24]);
    assert_eq!(
        input.vision()[0].spatial().patch_order(),
        &(0..16).collect::<Vec<_>>()
    );

    let expanded = input.tokens().to_vec();
    let expanded = TokenPlan::new(
        expanded.clone(),
        InputLayout::new(expanded.len(), vec![]).unwrap(),
    )
    .unwrap();
    assert!(adapter.prepare(&model, expanded, &[media]).is_err());
}

#[test]
fn qwen_adapter_rejects_media_with_wrong_processor_identity() {
    let model = inspect_components(
        &directory(false),
        Some(&projector()),
        PackageIdentity {
            target: ArtifactIdentity([1; 32]),
            projector: Some(ArtifactIdentity([2; 32])),
        },
    )
    .unwrap();
    let adapter = QwenInputAdapter::new(Some(
        QwenImageTokens::new(TokenId(99), TokenId(98), TokenId(100)).unwrap(),
    ));
    let tokens = vec![TokenId(98), TokenId(99), TokenId(100)];
    let plan = TokenPlan::new(
        tokens.clone(),
        InputLayout::new(tokens.len(), vec![]).unwrap(),
    )
    .unwrap();
    let original = prepared_qwen_media(&model);
    let altered = PreparedMedia::new("b".repeat(64), original.tensors().to_vec()).unwrap();
    assert!(adapter.prepare(&model, plan, &[altered]).is_err());
}
