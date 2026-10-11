//! Interpretation of every LFM2 catalog variant's real GGUF header
//! (`inference/validation/results/catalog-headers/`).

use magnitude_artifacts::{
    gguf::{Directory, Scalar, Value},
    ArtifactIdentity, InputLayout, PackageIdentity, TokenId,
};
use magnitude_family_contracts::{
    AttentionGate, Block, ExitNorm, ExpertSelection, HeadNorm, ImportTransform, InputNorm,
    ModelDefinition, ModelInputAdapter, Operator, OutputForm, Rotary, RowRange, ScoreFunction,
    TokenPlan,
};
use magnitude_family_common::{headers, TextInput};
use magnitude_family_lfm2::{inspect_components, recognize, Architecture};

struct Variant {
    model: String,
    role: String,
    directory: Directory,
}

/// Every LFM2 catalog target. The dump records integers without their GGUF
/// type: scalar integers are stored unsigned, integer arrays (per-layer
/// `head_count_kv`) as INT32, as gguf-py writes them.
fn catalog() -> Vec<Variant> {
    headers::index()
        .into_iter()
        .filter(|(model, role, _)| model.starts_with("lfm2") && role.starts_with("target"))
        .map(|(model, role, file)| {
            let mut directory = headers::directory(&file);
            for entry in &mut directory.metadata {
                if let Value::Array(values) = &mut entry.value {
                    for value in values {
                        if let Scalar::Unsigned(integer) = *value {
                            *value = Scalar::Signed(integer as i64);
                        }
                    }
                }
            }
            Variant {
                model,
                role,
                directory,
            }
        })
        .collect()
}

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([0; 32]),
        projector: None,
    }
}

fn inspect(directory: &Directory) -> Result<ModelDefinition, String> {
    inspect_components(directory, None, identity()).map_err(|error| error.to_string())
}

fn variant(model: &str) -> Directory {
    catalog()
        .into_iter()
        .find(|variant| variant.model == model && variant.role == "target-gguf_q4")
        .expect("catalog variant")
        .directory
}

/// Geometry the catalog models are documented to have
/// (`specs/26-09-26/model-family-support/briefs/lfm2.md` §2).
struct Expected {
    architecture: Architecture,
    layers: usize,
    attention: &'static [usize],
    dense_intermediate: u64,
    leading_dense: usize,
    rope_base: f64,
}

fn expected(model: &str) -> Expected {
    match model {
        "lfm2.5-2.6b" => Expected {
            architecture: Architecture::Dense,
            layers: 30,
            attention: &[2, 5, 9, 13, 17, 21, 24, 27],
            dense_intermediate: 10752,
            leading_dense: 30,
            rope_base: 1.0e7,
        },
        "lfm2.5-8b-a1b" => Expected {
            architecture: Architecture::Routed,
            layers: 24,
            attention: &[2, 6, 10, 14, 18, 21],
            dense_intermediate: 7168,
            leading_dense: 2,
            rope_base: 5.0e6,
        },
        other => panic!("unexpected catalog model {other}"),
    }
}

fn check_block(block: &Block, index: usize, expected: &Expected, directory: &Directory) {
    let [mixer, feed_forward] = block.sublayers.as_slice() else {
        panic!("block {index} has two sublayers");
    };
    for (sublayer, norm) in [(mixer, "attn_norm"), (feed_forward, "ffn_norm")] {
        let InputNorm::Rms(input) = &sublayer.input else {
            panic!("block {index} sublayers are RMS pre-normalized");
        };
        assert_eq!(input.weight.name, format!("blk.{index}.{norm}.weight"));
        assert_eq!(input.epsilon, 9.999999747378752e-06);
        assert_eq!(sublayer.output, OutputForm::Residual);
    }
    match &mixer.op {
        Operator::Attention(attention) => {
            assert!(expected.attention.contains(&index), "layer {index} is attention");
            assert_eq!(
                (attention.heads, attention.kv_heads, attention.width),
                (32, 8, 64)
            );
            assert_eq!(attention.gate, AttentionGate::None);
            assert_eq!(attention.query.shape, [2048, 2048]);
            assert!(matches!(&attention.query_norm, HeadNorm::Rms(norm) if norm.weight.shape == [64]));
            assert_eq!(attention.scale, 1.0 / 8.0);
            let Rotary::Table { pairs, .. } = &attention.rotary else {
                panic!("NEOX rotation is a pair table");
            };
            assert_eq!(pairs.len(), 32);
            assert_eq!(pairs[0].frequency, 1.0);
            assert_eq!(pairs[1].frequency, expected.rope_base.powf(-2.0 / 64.0));
            assert!(pairs.iter().all(|pair| pair.amplitude == 1.0));
        }
        Operator::ShortConv(conv) => {
            assert!(!expected.attention.contains(&index), "layer {index} is short conv");
            assert_eq!((conv.channels, conv.width), (2048, 3));
            let fused = format!("blk.{index}.shortconv.in_proj.weight");
            let stored = &directory.tensor(&fused).expect("in_proj").shape;
            // B, C, X in stored row order.
            for (chunk, weight) in [&conv.input_gate, &conv.output_gate, &conv.value]
                .into_iter()
                .enumerate()
            {
                assert_eq!(weight.name, fused);
                assert_eq!(
                    weight.transforms,
                    [ImportTransform::Rows(RowRange {
                        start: chunk as u64 * 2048,
                        rows: 2048
                    })]
                );
                assert_eq!(weight.transformed_shape(stored).unwrap(), weight.shape);
            }
            assert_eq!(conv.convolution.shape, [2048, 3]);
        }
        other => panic!("layer {index} mixer is {}", other.name()),
    }
    match &feed_forward.op {
        Operator::DenseFfn(dense) => {
            assert!(index < expected.leading_dense);
            assert_eq!(dense.intermediate, expected.dense_intermediate);
        }
        Operator::RoutedFfn(routed) => {
            assert!(index >= expected.leading_dense);
            assert_eq!(
                (routed.experts, routed.selected, routed.intermediate),
                (32, 4, 1792)
            );
            assert_eq!(routed.router.score, ScoreFunction::Sigmoid);
            assert_eq!(
                routed.router.normalization,
                magnitude_family_contracts::RouteNormalization::SumPlusEpsilon(1e-6)
            );
            assert_eq!(routed.router.scale, 1.0);
            assert!(matches!(
                &routed.router.selection,
                ExpertSelection::TopK { bias: Some(bias) }
                    if bias.name == format!("blk.{index}.exp_probs_b.bias")
            ));
            assert!(routed.shared.is_none() && routed.expert_scale.is_none());
        }
        other => panic!("layer {index} feed-forward is {}", other.name()),
    }
}

#[test]
fn every_catalog_variant_binds_its_documented_geometry() {
    let catalog = catalog();
    assert_eq!(catalog.len(), 8, "four quantizations of two models");
    for Variant {
        model,
        role,
        directory,
    } in &catalog
    {
        let expected = expected(model);
        assert_eq!(recognize(directory).unwrap(), expected.architecture);
        let definition =
            inspect(directory).unwrap_or_else(|error| panic!("{model} {role}: {error}"));
        assert!(definition.deferred_forms().is_empty());
        let decoder = &definition.decoder;
        assert_eq!(
            (decoder.hidden, decoder.vocabulary, decoder.context_limit),
            (2048, 128000, 128000)
        );
        assert_eq!(definition.inputs.coordinate_axes, 1);
        assert!(decoder.entry.norm.is_none());
        // Tied output; the final norm is the tensor named `token_embd_norm`.
        assert_eq!(decoder.exit.output.name, "token_embd.weight");
        assert!(matches!(
            &decoder.exit.norm,
            ExitNorm::Rms(norm) if norm.weight.name == "token_embd_norm.weight"
        ));
        assert_eq!(decoder.blocks.len(), expected.layers);
        for (index, block) in decoder.blocks.iter().enumerate() {
            check_block(block, index, &expected, directory);
        }
    }
}

fn without_tensor(mut directory: Directory, name: &str) -> Directory {
    headers::remove_tensor(&mut directory, name);
    directory
}

fn with_metadata(mut directory: Directory, name: &str, value: Value) -> Directory {
    headers::set(&mut directory, name, value);
    directory
}

#[test]
fn recognizes_only_lfm2_architectures() {
    // The DSpark drafts (`dflash`) belong to the draft family.
    let directory = with_metadata(
        variant("lfm2.5-2.6b"),
        "general.architecture",
        Value::Scalar(Scalar::String("dflash".into())),
    );
    assert!(recognize(&directory).is_err());
}

#[test]
fn rejects_missing_unbound_and_misshapen_weights() {
    let missing = without_tensor(variant("lfm2.5-2.6b"), "blk.4.shortconv.conv.weight");
    assert!(inspect(&missing).unwrap_err().contains("blk.4.shortconv.conv.weight"));

    let missing_bias = without_tensor(variant("lfm2.5-8b-a1b"), "blk.7.exp_probs_b.bias");
    assert!(inspect(&missing_bias).unwrap_err().contains("exp_probs_b"));

    let mut unbound = variant("lfm2.5-2.6b");
    let mut extra = unbound.tensors[1].clone();
    extra.name = "blk.0.shortconv.bias".into();
    unbound.tensors.push(extra);
    assert!(inspect(&unbound).unwrap_err().contains("unbound weight role"));

    let mut misshapen = variant("lfm2.5-2.6b");
    let in_proj = misshapen
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "blk.0.shortconv.in_proj.weight")
        .unwrap();
    in_proj.shape = vec![4096, 2048];
    assert!(inspect(&misshapen).unwrap_err().contains("expected [6144, 2048]"));

    // An untied output projection is bound when present.
    let mut untied = variant("lfm2.5-2.6b");
    let mut output = untied.tensor("token_embd.weight").unwrap().clone();
    output.name = "output.weight".into();
    untied.tensors.push(output);
    assert_eq!(inspect(&untied).unwrap().decoder.exit.output.name, "output.weight");
}

#[test]
fn rejects_unknown_and_malformed_metadata() {
    let sliding = with_metadata(
        variant("lfm2.5-2.6b"),
        "lfm2.attention.sliding_window",
        Value::Scalar(Scalar::Unsigned(4096)),
    );
    assert!(inspect(&sliding).unwrap_err().contains("not understood"));

    let scaled = with_metadata(
        variant("lfm2.5-8b-a1b"),
        "lfm2moe.expert_weights_scale",
        Value::Scalar(Scalar::Float(2.5)),
    );
    assert!(inspect(&scaled).unwrap_err().contains("not understood"));

    // Expert keys belong to the routed architecture only.
    let routed_key = with_metadata(
        variant("lfm2.5-2.6b"),
        "lfm2.expert_count",
        Value::Scalar(Scalar::Unsigned(32)),
    );
    assert!(inspect(&routed_key).unwrap_err().contains("not understood"));

    let scalar_kv = with_metadata(
        variant("lfm2.5-2.6b"),
        "lfm2.attention.head_count_kv",
        Value::Scalar(Scalar::Unsigned(8)),
    );
    assert!(inspect(&scalar_kv).unwrap_err().contains("per-layer array"));

    let short_kv = with_metadata(
        variant("lfm2.5-2.6b"),
        "lfm2.attention.head_count_kv",
        Value::Array(vec![Scalar::Unsigned(0); 29]),
    );
    assert!(inspect(&short_kv).unwrap_err().contains("one entry per layer"));

    let mut negative = vec![Scalar::Signed(0); 30];
    negative[2] = Scalar::Signed(-8);
    let negative_kv = with_metadata(
        variant("lfm2.5-2.6b"),
        "lfm2.attention.head_count_kv",
        Value::Array(negative),
    );
    assert!(inspect(&negative_kv).unwrap_err().contains("nonnegative"));

    let softmax_weight = with_metadata(
        variant("lfm2.5-8b-a1b"),
        "lfm2moe.expert_gating_func",
        Value::Scalar(Scalar::Unsigned(3)),
    );
    assert!(inspect(&softmax_weight).unwrap_err().contains("expert_gating_func"));

    let vocabulary = with_metadata(
        variant("lfm2.5-2.6b"),
        "lfm2.vocab_size",
        Value::Scalar(Scalar::Unsigned(127999)),
    );
    assert!(inspect(&vocabulary).unwrap_err().contains("vocab_size"));

    let key_length = with_metadata(
        variant("lfm2.5-2.6b"),
        "lfm2.attention.key_length",
        Value::Scalar(Scalar::Unsigned(128)),
    );
    assert!(inspect(&key_length).unwrap_err().contains("key_length"));
}

#[test]
fn the_layer_pattern_follows_the_key_head_array() {
    // The array alone decides each layer's operator: reading layer 2 as a
    // conv layer demands conv weights the file does not have.
    let mut pattern = vec![Scalar::Unsigned(0); 30];
    for layer in [2, 5, 9, 13, 17, 21, 24, 27] {
        pattern[layer] = Scalar::Unsigned(8);
    }
    pattern[2] = Scalar::Unsigned(0);
    let directory = with_metadata(
        variant("lfm2.5-2.6b"),
        "lfm2.attention.head_count_kv",
        Value::Array(pattern),
    );
    // Layer 2 is now read as a conv layer, which has no in_proj.
    assert!(inspect(&directory)
        .unwrap_err()
        .contains("blk.2.shortconv.in_proj.weight"));
}

#[test]
fn a_projector_is_rejected() {
    let directory = variant("lfm2.5-2.6b");
    let error = inspect_components(&directory, Some(&directory), identity()).unwrap_err();
    assert!(error.to_string().contains("no projector"));
}

#[test]
fn text_rows_take_their_absolute_positions() {
    let definition = inspect(&variant("lfm2.5-8b-a1b")).unwrap();
    let tokens = vec![TokenId(124894), TokenId(7), TokenId(9)];
    let plan = TokenPlan::new(tokens, InputLayout::new(3, Vec::new()).unwrap()).unwrap();
    let prepared = TextInput::new(&definition).prepare(&definition, plan, &[]).unwrap();
    assert_eq!(prepared.coordinates(), &[[0; 3], [1; 3], [2; 3]]);
    assert_eq!(prepared.coordinates_at(3, 2).unwrap(), [[3; 3], [4; 3]]);
}
