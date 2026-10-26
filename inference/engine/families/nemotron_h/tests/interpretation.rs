//! Interpretation of every catalog Nemotron-H header, plus negative cases.
//!
//! The headers are the locked catalog components' (`headers::target`).

use magnitude_artifacts::{
    gguf::{Array, Directory, Encoding, Metadata, Scalar, TensorDescriptor, Value},
    ArtifactIdentity, InputLayout, PackageIdentity, TokenId, UnsupportedReason,
};
use magnitude_family_common::{headers, TextInput};
use magnitude_family_contracts::{
    ActivationFunction, AttentionGate, ExitNorm, ExpertSelection, FeedForwardUp, HeadNorm,
    ImportTransform, InputNorm, InputPreparationError, KeyValue, ModelDefinition,
    ModelInputAdapter, Operator, Rotary, ScoreFunction, SharedExpertGate, TokenPlan,
    WeightDescriptor,
};
use magnitude_family_nemotron_h::{inspect_components, recognize, Error};

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    }
}

fn inspect(directory: &Directory) -> Result<ModelDefinition, Error> {
    inspect_components(directory, None, identity())
}

const LIGHTNING: [&str; 3] = [
    "nemotron-3.5-lightning-30b-a3b:gguf:nvfp4-qat",
    "nemotron-3.5-lightning-30b-a3b:gguf:q4",
    "nemotron-3.5-lightning-30b-a3b:gguf:q8",
];
const SUPER: [&str; 2] = [
    "nemotron-3-super-120b-a12b:gguf:mxfp4",
    "nemotron-3-super-120b-a12b:gguf:q4",
];
const ULTRA: &str = "nemotron-3-ultra-550b-a55b:gguf:mxfp4";

/// NVIDIA's 24GB Bolt release (IQ2_XXS experts, NVFP4 state space).
fn bolt() -> Directory {
    headers::target("nemotron-3.5-bolt-74b-a9b:gguf:iq2-qat")
}

/// Unsloth's Super VL UD-Q4_K_XL, all three shards' tensors, which states
/// only `layer_norm_epsilon`.
fn super_vl() -> Directory {
    headers::target("nemotron-3.5-super-vl-120b-a12b:gguf:q4")
}

/// The layer pattern, one character per sublayer (`M` state space, `*`
/// attention, `E` routed), with `|` between blocks.
fn pattern(definition: &ModelDefinition) -> String {
    definition
        .decoder
        .blocks
        .iter()
        .map(|block| {
            block
                .sublayers
                .iter()
                .map(|sublayer| match sublayer.op {
                    Operator::StateSpace(_) => 'M',
                    Operator::Attention(_) => '*',
                    Operator::RoutedFfn(_) => 'E',
                    ref other => panic!("unexpected operator {}", other.name()),
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("|")
}

fn scale_tensor(weight: &WeightDescriptor) -> Option<&str> {
    weight
        .transforms
        .iter()
        .find_map(|transform| match transform {
            ImportTransform::ScaleByTensor { tensor, .. } => Some(tensor.as_str()),
            _ => None,
        })
}

struct Expected {
    pattern: &'static str,
    hidden: u64,
    heads: u64,
    state_heads: u64,
    norm_group: u64,
    experts: u64,
    selected: u64,
    intermediate: u64,
    shared: u64,
    latent: Option<u64>,
    scale: f64,
}

const LIGHTNING_PATTERN: &str = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";
const SUPER_PATTERN: &str =
    "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME";
const ULTRA_PATTERN: &str = "MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";
/// Blocks as the family must form them: a mixer takes the feed-forward
/// layer after it, and a mixer followed by a mixer stands alone.
fn blocks_of(layers: &str) -> String {
    let mut blocks = Vec::new();
    let mut layers = layers.chars().peekable();
    while let Some(mixer) = layers.next() {
        let mut block = mixer.to_string();
        if layers.peek() == Some(&'E') {
            block.push(layers.next().unwrap());
        }
        blocks.push(block);
    }
    blocks.join("|")
}

fn check(definition: &ModelDefinition, expected: &Expected) {
    let decoder = &definition.decoder;
    assert_eq!(definition.family.0, "nemotron_h_moe");
    assert!(definition.head.is_none() && definition.vision.is_none());
    assert!(definition.deferred_forms().is_empty());
    assert_eq!(decoder.hidden, expected.hidden);
    assert_eq!(decoder.vocabulary, 131_072);
    assert_eq!(decoder.context_limit, 1_048_576);
    assert_eq!(pattern(definition), blocks_of(expected.pattern));
    assert_ne!(decoder.exit.output.name, decoder.entry.embedding.name);
    for (_, sublayer) in decoder.sublayers() {
        let InputNorm::Rms(norm) = &sublayer.input else {
            panic!("every sublayer is RMS pre-normalized");
        };
        assert!(norm.weight.name.ends_with("attn_norm.weight"));
        assert!((norm.epsilon - 1e-5).abs() < 1e-10);
        match &sublayer.op {
            Operator::Attention(attention) => {
                assert_eq!(
                    (attention.heads, attention.kv_heads, attention.width),
                    (expected.heads, 2, 128)
                );
                assert_eq!(attention.gate, AttentionGate::None);
                assert_eq!(attention.query_norm, HeadNorm::None);
                assert_eq!(attention.rotary, Rotary::None);
                assert!((attention.scale - 1.0 / 128f64.sqrt()).abs() < 1e-15);
                assert!(matches!(
                    attention.key_value,
                    KeyValue::Owned {
                        key_norm: HeadNorm::None,
                        ..
                    }
                ));
            }
            Operator::StateSpace(space) => {
                assert_eq!(
                    (space.heads, space.head_width, space.state, space.groups),
                    (expected.state_heads, 64, 128, 8)
                );
                assert_eq!(space.norm_group, expected.norm_group);
                let inner = expected.state_heads * 64;
                let channels = inner + 2 * 8 * 128;
                assert!(space.projection.name.ends_with("ssm_in.weight"));
                assert_eq!(
                    space.projection.shape,
                    [inner + channels + expected.state_heads, expected.hidden]
                );
                assert!(space.projection.transforms.is_empty());
                assert_eq!(space.decay.shape, [expected.state_heads]);
                assert_eq!(space.decay.transforms, [ImportTransform::Flatten]);
                assert_eq!(space.norm.weight.shape, [inner]);
            }
            Operator::RoutedFfn(routed) => {
                assert_eq!(
                    (routed.experts, routed.selected, routed.intermediate),
                    (expected.experts, expected.selected, expected.intermediate)
                );
                assert_eq!(routed.router.score, ScoreFunction::Sigmoid);
                assert!(matches!(
                    routed.router.selection,
                    ExpertSelection::TopK { bias: Some(_) }
                ));
                assert_eq!(
                    routed.router.normalization,
                    magnitude_family_contracts::RouteNormalization::SumPlusEpsilon(1e-20)
                );
                assert_eq!(routed.router.scale, expected.scale);
                assert!(matches!(
                    routed.expert_up,
                    FeedForwardUp::Plain {
                        activation: ActivationFunction::ReluSquared,
                        ..
                    }
                ));
                assert_eq!(routed.latent.as_ref().map(|l| l.width), expected.latent);
                let shared = routed.shared.as_ref().unwrap();
                assert_eq!(shared.intermediate, expected.shared);
                assert_eq!(shared.gate, SharedExpertGate::None);
                assert_eq!(shared.up.activation(), ActivationFunction::ReluSquared);
            }
            other => panic!("unexpected operator {}", other.name()),
        }
    }
}

fn lightning() -> Expected {
    Expected {
        pattern: LIGHTNING_PATTERN,
        hidden: 2688,
        heads: 32,
        state_heads: 64,
        norm_group: 512,
        experts: 128,
        selected: 6,
        intermediate: 1856,
        shared: 3712,
        latent: None,
        scale: 2.5,
    }
}

#[test]
fn every_lightning_variant_binds_with_its_prediction_block_skipped() {
    for file in LIGHTNING {
        let definition = inspect(&headers::target(file)).unwrap_or_else(|error| panic!("{file}: {error}"));
        check(&definition, &lightning());
        // One block per mixer (23 state space, 6 attention); 6 lone state
        // space blocks.
        assert_eq!(definition.decoder.blocks.len(), 29);
        let lone = pattern(&definition)
            .split('|')
            .filter(|block| *block == "M")
            .count();
        assert_eq!(lone, 6);
    }
}

#[test]
fn super_and_ultra_bind_latent_experts() {
    for file in SUPER {
        let definition = inspect(&headers::target(file)).unwrap_or_else(|error| panic!("{file}: {error}"));
        check(
            &definition,
            &Expected {
                pattern: SUPER_PATTERN,
                hidden: 4096,
                heads: 32,
                state_heads: 128,
                norm_group: 1024,
                experts: 512,
                selected: 22,
                intermediate: 2688,
                shared: 5376,
                latent: Some(1024),
                scale: 5.0,
            },
        );
    }
    let definition = inspect(&headers::target(ULTRA)).unwrap();
    check(
        &definition,
        &Expected {
            pattern: ULTRA_PATTERN,
            hidden: 8192,
            heads: 64,
            state_heads: 256,
            norm_group: 2048,
            experts: 512,
            selected: 22,
            intermediate: 5120,
            shared: 10240,
            latent: Some(2048),
            scale: 5.0,
        },
    );
    // Lone state-space blocks: 8 in Super, 12 in Ultra.
    let lone = pattern(&definition)
        .split('|')
        .filter(|block| *block == "M")
        .count();
    assert_eq!(lone, 12);
}

#[test]
fn a_header_without_an_rms_epsilon_normalizes_with_its_layer_norm_epsilon() {
    let mut directory = lightning_q8();
    directory
        .metadata
        .retain(|entry| entry.name != "nemotron_h_moe.attention.layer_norm_rms_epsilon");
    check(&inspect(&directory).unwrap(), &lightning());

    // Super VL states only its layer-norm epsilon
    // (`super_vl_binds_like_super_with_its_single_epsilon`).
    assert!(super_vl()
        .value("nemotron_h_moe.attention.layer_norm_rms_epsilon")
        .is_none());

    let mut directory = super_vl();
    directory
        .metadata
        .retain(|entry| entry.name != "nemotron_h_moe.attention.layer_norm_epsilon");
    let error = inspect(&directory).unwrap_err();
    assert_eq!(
        error,
        Error::MissingMetadata("nemotron_h_moe.attention.layer_norm_epsilon".into())
    );
    assert_eq!(
        error.reasons(),
        [UnsupportedReason::MissingMetadata {
            key: "nemotron_h_moe.attention.layer_norm_epsilon".into()
        }]
    );
}

#[test]
fn super_vl_binds_like_super_with_its_single_epsilon() {
    // Its header states only `layer_norm_epsilon`; the RMS norms take it.
    let definition = inspect(&super_vl()).unwrap();
    check(
        &definition,
        &Expected {
            pattern: SUPER_PATTERN,
            hidden: 4096,
            heads: 32,
            state_heads: 128,
            norm_group: 1024,
            experts: 512,
            selected: 22,
            intermediate: 2688,
            shared: 5376,
            latent: Some(1024),
            scale: 5.0,
        },
    );
}

/// Bolt's routed layers in order: (experts, expert width).
const BOLT_ROUTED: [(u64, u64); 40] = [
    (240, 2048), (240, 1536), (240, 2048), (304, 2304), (240, 2304), (240, 2048), (240, 2048),
    (304, 2304), (304, 2048), (432, 2560), (432, 2304), (240, 2048), (432, 2304), (304, 2560),
    (304, 2304), (432, 2304), (368, 2304), (368, 2048), (368, 2304), (432, 2560), (368, 2560),
    (496, 2560), (432, 2560), (240, 2304), (368, 2304), (368, 2560), (304, 2048), (240, 2304),
    (240, 2304), (240, 2048), (240, 2048), (368, 2048), (368, 2304), (432, 2304), (432, 2560),
    (496, 2560), (432, 2560), (432, 2560), (496, 2560), (368, 2560),
];

#[test]
fn bolt_binds_heterogeneous_experts_and_scaled_state_space() {
    let definition = inspect(&bolt()).unwrap();
    let decoder = &definition.decoder;
    assert!(definition.head.is_none() && definition.vision.is_none());
    assert_eq!((decoder.hidden, decoder.vocabulary), (4096, 131_072));
    assert_eq!(decoder.context_limit, 262_144);
    assert_eq!(pattern(&definition), blocks_of(SUPER_PATTERN));
    let mut routed = Vec::new();
    for (_, sublayer) in decoder.sublayers() {
        match &sublayer.op {
            Operator::RoutedFfn(ffn) => {
                assert_eq!(ffn.selected, 9);
                assert_eq!(ffn.router.score, ScoreFunction::Sigmoid);
                assert_eq!(ffn.router.weight.shape, [ffn.experts, 4096]);
                assert_eq!(
                    ffn.expert_up.up().shape,
                    [ffn.experts, ffn.intermediate, 1024]
                );
                assert_eq!(ffn.latent.as_ref().map(|latent| latent.width), Some(1024));
                assert_eq!(ffn.shared.as_ref().unwrap().intermediate, 5376);
                routed.push((ffn.experts, ffn.intermediate));
            }
            Operator::StateSpace(space) => {
                // NVFP4 projections carry their second-level scales.
                let projection = scale_tensor(&space.projection).unwrap();
                assert!(projection.ends_with("ssm_in.scale"));
                let output = scale_tensor(&space.output).unwrap();
                assert!(output.ends_with("ssm_out.scale"));
            }
            Operator::Attention(_) => {}
            other => panic!("unexpected operator {}", other.name()),
        }
    }
    assert_eq!(routed, BOLT_ROUTED);
}

#[test]
fn rejects_softmax_gating_and_oversized_layer_selections() {
    let mut directory = bolt();
    let gating = directory
        .metadata
        .iter_mut()
        .find(|item| item.name == "nemotron_h_moe.expert_gating_func")
        .unwrap();
    gating.value = Value::Scalar(Scalar::Unsigned(1));
    assert!(matches!(inspect(&directory), Err(Error::Metadata(_))));

    // Layer 1 routes over 240 experts; selecting more is impossible.
    let mut directory = bolt();
    edit_counts(&mut directory, "expert_used_count", |values| values[1] = 241);
    assert!(
        matches!(inspect(&directory), Err(Error::Structure(message)) if message.contains("routed layer 1"))
    );
}

#[test]
fn nvfp4_second_level_scales_bind_to_their_weights() {
    let definition = inspect(&headers::target(LIGHTNING[0])).unwrap();
    // Every second-level scale scales the weight it belongs to (the
    // executor applies it on that projection's accumulator).
    assert_eq!(
        scale_tensor(&definition.decoder.exit.output),
        Some("output.scale")
    );
    let ExitNorm::Rms(norm) = &definition.decoder.exit.norm else {
        panic!("Nemotron-H exits through an RMS norm");
    };
    assert_eq!(scale_tensor(&norm.weight), None);
    assert_eq!(scale_tensor(&definition.decoder.entry.embedding), None);
    let routed = definition
        .decoder
        .sublayers()
        .find_map(|(_, sublayer)| match &sublayer.op {
            Operator::RoutedFfn(routed) => Some(routed),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        scale_tensor(routed.expert_up.up()),
        Some("blk.1.ffn_up_exps.scale")
    );
    assert_eq!(
        scale_tensor(&routed.expert_down),
        Some("blk.1.ffn_down_exps.scale")
    );
    assert_eq!(routed.expert_scale, None);
    let shared = routed.shared.as_ref().unwrap();
    assert_eq!(
        scale_tensor(shared.up.up()),
        Some("blk.1.ffn_up_shexp.scale")
    );
    assert_eq!(
        scale_tensor(&shared.down),
        Some("blk.1.ffn_down_shexp.scale")
    );
    assert_eq!(scale_tensor(&routed.router.weight), None);
}

#[test]
fn recognizes_only_the_nemotron_h_architecture() {
    for file in LIGHTNING.iter().chain(&SUPER).chain([&ULTRA]) {
        assert!(recognize(&headers::target(file)).is_ok(), "{file}");
    }
    let draft = headers::draft("nemotron-3.5-lightning-30b-a3b:gguf:q4");
    assert!(matches!(recognize(&draft), Err(Error::Architecture(_))));
    let qwen = headers::target("qwen3.5-4b:gguf:q4");
    assert!(matches!(recognize(&qwen), Err(Error::Architecture(_))));
}

/// Edit the per-layer counts `nemotron_h_moe.<key>`, whatever integer type
/// the header stores them as.
fn edit_counts(directory: &mut Directory, key: &str, edit: impl FnOnce(&mut Vec<u64>)) {
    let name = format!("nemotron_h_moe.{key}");
    let item = directory
        .metadata
        .iter_mut()
        .find(|item| item.name == name)
        .unwrap_or_else(|| panic!("{name} is absent"));
    let Value::Array(array) = &mut item.value else {
        panic!("{name} is not an array");
    };
    let mut values = array
        .iter()
        .map(|value| value.unsigned().unwrap())
        .collect::<Vec<_>>();
    edit(&mut values);
    *array = Array::of(values.into_iter().map(Scalar::Unsigned)).unwrap();
}

fn lightning_q8() -> Directory {
    headers::target(LIGHTNING[2])
}

#[test]
fn rejects_a_missing_tensor() {
    let mut directory = lightning_q8();
    directory
        .tensors
        .retain(|tensor| tensor.name != "blk.0.ssm_conv1d.bias");
    assert_eq!(
        inspect(&directory).unwrap_err(),
        Error::Weight("missing Nemotron-H weight \"blk.0.ssm_conv1d.bias\"".into())
    );
}

#[test]
fn rejects_a_wrong_shape() {
    let mut directory = lightning_q8();
    let tensor = directory
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "blk.5.attn_k.weight")
        .unwrap();
    tensor.shape = vec![384, 2688];
    assert!(
        matches!(inspect(&directory), Err(Error::Weight(message)) if message.contains("blk.5.attn_k.weight"))
    );
}

#[test]
fn rejects_unknown_metadata_and_unbound_tensors() {
    let mut directory = lightning_q8();
    directory.metadata.push(Metadata {
        name: "nemotron_h_moe.expert_router_temperature".into(),
        value: Value::Scalar(Scalar::Unsigned(1)),
    });
    assert_eq!(
        inspect(&directory).unwrap_err(),
        Error::UnknownMetadata(vec!["nemotron_h_moe.expert_router_temperature".into()])
    );

    let mut directory = lightning_q8();
    directory.tensors.push(TensorDescriptor {
        name: "blk.0.ssm_extra.weight".into(),
        shape: vec![8],
        encoding: Encoding::F32,
        offset: 0,
        nbytes: 32,
    });
    assert_eq!(
        inspect(&directory).unwrap_err(),
        Error::UnboundWeights(vec!["blk.0.ssm_extra.weight".into()])
    );
}

#[test]
fn rejects_a_misshapen_second_level_scale() {
    let mut directory = headers::target(LIGHTNING[0]);
    let scale = directory
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "blk.1.ffn_up_exps.scale")
        .unwrap();
    scale.shape = vec![1];
    assert!(
        matches!(inspect(&directory), Err(Error::Weight(message)) if message.contains("blk.1.ffn_up_exps.scale"))
    );
}

#[test]
fn rejects_layer_sequences_that_form_no_blocks() {
    // A feed-forward layer first, with no mixer before it.
    let mut directory = lightning_q8();
    for key in ["feed_forward_length", "attention.head_count_kv"] {
        edit_counts(&mut directory, key, |values| values.swap(0, 1));
    }
    assert!(matches!(inspect(&directory), Err(Error::Structure(_))));
}

#[test]
fn a_feed_forward_layer_without_a_router_is_dense_squared_relu() {
    let mut directory = lightning_q8();
    directory.tensors.retain(|tensor| {
        !tensor.name.starts_with("blk.1.") || tensor.name == "blk.1.attn_norm.weight"
    });
    for (name, shape) in [
        ("blk.1.ffn_up.weight", vec![1856, 2688]),
        ("blk.1.ffn_down.weight", vec![2688, 1856]),
    ] {
        directory.tensors.push(TensorDescriptor {
            name: name.into(),
            nbytes: shape.iter().product::<u64>() * 4,
            shape,
            encoding: Encoding::F32,
            offset: 0,
        });
    }
    let definition = inspect(&directory).unwrap();
    let Operator::DenseFfn(dense) = &definition.decoder.blocks[0].sublayers[1].op else {
        panic!("layer 1 is dense");
    };
    assert_eq!(dense.intermediate, 1856);
    assert!(matches!(
        dense.up,
        FeedForwardUp::Plain {
            activation: ActivationFunction::ReluSquared,
            ..
        }
    ));
}

#[test]
fn text_input_takes_absolute_positions_and_rejects_media_and_foreign_tokens() {
    let definition = inspect(&lightning_q8()).unwrap();
    let adapter = TextInput::new(&definition);
    let plan = |tokens: Vec<u32>| {
        let count = tokens.len();
        TokenPlan::new(
            tokens.into_iter().map(TokenId).collect(),
            InputLayout::new(count, Vec::new()).unwrap(),
        )
        .unwrap()
    };
    let prepared = adapter
        .prepare(&definition, plan(vec![1, 10, 131_071]), &[])
        .unwrap();
    assert_eq!(prepared.coordinates(), &[[0; 3], [1; 3], [2; 3]]);
    assert_eq!(
        adapter
            .prepare(&definition, plan(vec![131_072]), &[])
            .unwrap_err(),
        InputPreparationError::TokenDomain
    );
}

#[test]
fn a_projector_other_than_nemotron_omni_is_rejected() {
    let directory = lightning_q8();
    let error = inspect_components(&directory, Some(&directory), identity()).unwrap_err();
    assert!(matches!(error, Error::Projector(_)));
    assert!(matches!(
        error.reasons().as_slice(),
        [UnsupportedReason::Unreadable]
    ));
}
