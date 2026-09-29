//! Interpretation of every catalog Nemotron-H header, plus negative cases.
//!
//! The headers are the metadata and full tensor directories of the locked
//! catalog files (`inference/validation/results/catalog-headers`).

use magnitude_artifacts::{
    gguf::{Directory, Encoding, Metadata, Scalar, TensorDescriptor, Value},
    ArtifactIdentity, InputLayout, PackageIdentity, TokenId,
};
use magnitude_family_common::{headers::directory as catalog, TextInput};
use magnitude_family_contracts::{
    ActivationFunction, AttentionGate, ExitNorm, ExpertSelection, FeedForwardUp, HeadNorm,
    ImportTransform,
    InputNorm, InputPreparationError, KeyValue, ModelDefinition, ModelInputAdapter, Operator,
    Rotary, ScoreFunction, SharedExpertGate, TokenPlan, WeightDescriptor,
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
    "nemotron-3.5-lightning-30b-a3b__target-gguf_nvfp4-qat.json",
    "nemotron-3.5-lightning-30b-a3b__target-gguf_q4.json",
    "nemotron-3.5-lightning-30b-a3b__target-gguf_q8.json",
];
const SUPER: [&str; 2] = [
    "nemotron-3-super-120b-a12b__target-gguf_mxfp4-allshards.json",
    "nemotron-3-super-120b-a12b__target-gguf_q4-allshards.json",
];
const ULTRA: &str = "nemotron-3-ultra-550b-a55b__target-gguf_mxfp4-allshards.json";

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
            ImportTransform::ScaleByTensor { tensor } => Some(tensor.as_str()),
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
        let definition = inspect(&catalog(file)).unwrap_or_else(|error| panic!("{file}: {error}"));
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
        let definition = inspect(&catalog(file)).unwrap_or_else(|error| panic!("{file}: {error}"));
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
    let definition = inspect(&catalog(ULTRA)).unwrap();
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
fn nvfp4_second_level_scales_bind_to_their_weights() {
    let definition = inspect(&catalog(LIGHTNING[0])).unwrap();
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
        assert!(recognize(&catalog(file)).is_ok(), "{file}");
    }
    let draft = catalog("nemotron-3.5-lightning-30b-a3b__draft.json");
    assert!(matches!(recognize(&draft), Err(Error::Architecture(_))));
    let qwen = catalog("qwen3.5-4b__target-gguf_q4.json");
    assert!(matches!(recognize(&qwen), Err(Error::Architecture(_))));
}

fn lightning_q8() -> Directory {
    catalog(LIGHTNING[2])
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
        name: "nemotron_h_moe.expert_gating_func".into(),
        value: Value::Scalar(Scalar::Unsigned(1)),
    });
    assert!(matches!(inspect(&directory), Err(Error::Metadata(_))));

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
        Error::Weight(
            "Nemotron-H artifact contains unbound weight role \"blk.0.ssm_extra.weight\"".into()
        )
    );
}

#[test]
fn rejects_a_misshapen_second_level_scale() {
    let mut directory = catalog(LIGHTNING[0]);
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
        let name = format!("nemotron_h_moe.{key}");
        let item = directory
            .metadata
            .iter_mut()
            .find(|item| item.name == name)
            .unwrap();
        let Value::Array(values) = &mut item.value else {
            unreachable!()
        };
        values.swap(0, 1);
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
fn a_projector_is_rejected() {
    let directory = lightning_q8();
    assert!(matches!(
        inspect_components(&directory, Some(&directory), identity()),
        Err(Error::Structure(_))
    ));
}
