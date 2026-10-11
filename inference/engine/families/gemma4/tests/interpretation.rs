//! Interpretation of every catalog Gemma 4 header
//! (`inference/validation/results/catalog-headers/`), plus negative cases
//! derived from them.

use magnitude_artifacts::{
    gguf::{Directory, Metadata, Scalar, Value},
    ArtifactIdentity, InputLayout, PackageIdentity, TokenId,
};
use magnitude_family_common::headers::directory as header;
use magnitude_family_contracts::{
    ActivationFunction, CellReduction, EmbeddingScale, FeedForwardUp, HistoryDomain,
    ImportTransform, InputNorm, KeyValue, MediaRowAttention, ModelDefinition, ModelInputAdapter,
    Operator, OutputForm, PositionSampling, Rotary, SublayerIndex, TokenPlan, ValueNorm,
    ValueSource, VisionActivation, VisionAttentionScale, VisionResize, VisionStem, VisionUp,
};
use magnitude_family_gemma4::{
    describe_projector, inputs, inspect_components, recognize,
};

fn target(model: &str) -> Directory {
    header(&format!("gemma-4-{model}-it-qat__target-gguf_q4.json"))
}

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    }
}

fn inspect(directory: &Directory) -> Result<ModelDefinition, String> {
    inspect_components(directory, None, identity()).map_err(|error| error.to_string())
}

fn attention(model: &ModelDefinition, layer: u32) -> &magnitude_family_contracts::Attention {
    model
        .decoder
        .attention(SublayerIndex {
            block: layer,
            sublayer: 0,
        })
        .expect("attention sublayer")
}

#[test]
fn recognizes_only_the_gemma4_architecture() {
    assert!(recognize(&target("e2b")).is_ok());
    let mut other = target("e2b");
    other
        .metadata
        .iter_mut()
        .find(|entry| entry.name == "general.architecture")
        .unwrap()
        .value = Value::Scalar(Scalar::String("gemma3".into()));
    assert!(recognize(&other).is_err());
}

/// Per-layer-input models: E2B and E4B share history into their last layers
/// and project values on every owning layer.
#[test]
fn per_layer_input_models_bind_every_catalog_weight() {
    for (model, layers, owning, window_period, window, sources) in [
        ("e2b", 35u32, 15u32, 5u32, 512u64, (13u32, 14u32)),
        ("e4b", 42, 24, 6, 512, (22, 23)),
    ] {
        let definition = inspect(&target(model)).unwrap();
        let decoder = &definition.decoder;
        assert_eq!(decoder.blocks.len() as u32, layers, "{model}");
        assert_eq!(decoder.entry.scale, EmbeddingScale::SqrtHidden);
        assert_eq!(decoder.exit.softcap, Some(30.0));
        assert_eq!(decoder.exit.output.name, "token_embd.weight");
        assert_eq!(decoder.vocabulary, 262_144);
        let entry = decoder.entry.per_layer.as_ref().expect("per-layer entry");
        assert_eq!((entry.width, entry.layers), (256, layers as u64));
        assert_eq!(entry.table.shape, [262_144, 256 * layers as u64]);
        assert_eq!(entry.media_row, 0);
        assert_eq!(entry.table_scale, 16.0);
        assert_eq!(entry.combine_scale, std::f64::consts::FRAC_1_SQRT_2);
        for layer in 0..layers {
            let block = &decoder.blocks[layer as usize];
            assert_eq!(block.sublayers.len(), 3, "{model} layer {layer}");
            let attention = attention(&definition, layer);
            let full = (layer + 1) % window_period == 0;
            assert_eq!(attention.width, if full { 512 } else { 256 });
            assert_eq!(attention.scale, 1.0);
            assert_eq!(attention.media_rows, MediaRowAttention::Causal);
            match &attention.key_value {
                KeyValue::Owned {
                    value,
                    value_norm,
                    domain,
                    ..
                } => {
                    assert!(layer < owning);
                    assert!(matches!(value, ValueSource::Projected(_)));
                    assert!(matches!(value_norm, ValueNorm::RmsUnweighted(_)));
                    let expected = if full {
                        HistoryDomain::Token
                    } else {
                        HistoryDomain::Window { tokens: window }
                    };
                    assert_eq!(*domain, expected);
                }
                KeyValue::Shared { source } => {
                    assert!(layer >= owning);
                    let expected = if full { sources.1 } else { sources.0 };
                    assert_eq!(source.block, expected, "{model} layer {layer}");
                }
            }
            let Operator::PerLayerInput(input) = &block.sublayers[2].op else {
                panic!("{model} layer {layer} lacks its per-layer input");
            };
            assert_eq!(input.layer, layer as u64);
            assert_eq!(block.sublayers[2].input, InputNorm::None);
            assert!(matches!(block.sublayers[1].output, OutputForm::PostNorm(_)));
            assert!(matches!(
                &block.sublayers[2].output,
                OutputForm::ScaledPostNorm { layer_scale, .. }
                    if layer_scale.name == format!("blk.{layer}.layer_output_scale.weight")
            ));
        }
    }
}

#[test]
fn e2b_feed_forward_doubles_in_its_sharing_layers() {
    let definition = inspect(&target("e2b")).unwrap();
    for (layer, width) in [(0usize, 6144u64), (14, 6144), (15, 12288), (34, 12288)] {
        let Operator::DenseFfn(dense) = &definition.decoder.blocks[layer].sublayers[1].op else {
            panic!("dense feed-forward");
        };
        assert_eq!(dense.intermediate, width);
        assert_eq!(dense.up.activation(), ActivationFunction::GeluTanh);
    }
}

#[test]
fn full_layers_rotate_a_quarter_of_their_pairs_over_the_whole_head() {
    let definition = inspect(&target("e2b")).unwrap();
    let window = attention(&definition, 0);
    let Rotary::Table { pairs, divisors } = &window.rotary else {
        panic!("table rotary");
    };
    assert_eq!(pairs.len(), 128);
    assert_eq!(pairs[0].frequency, 1.0);
    assert!((pairs[1].frequency - 10_000f64.powf(-2.0 / 256.0)).abs() < 1e-15);
    assert!(divisors.is_none());

    let full = attention(&definition, 4);
    let Rotary::Table { pairs, divisors } = &full.rotary else {
        panic!("table rotary");
    };
    // Pair p rotates dimensions (p, p + 256); only the first 64 turn.
    assert_eq!(pairs.len(), 256);
    assert!((pairs[1].frequency - 1e6f64.powf(-2.0 / 512.0)).abs() < 1e-15);
    assert!(pairs[..64].iter().all(|pair| pair.frequency > 0.0));
    assert!(pairs[64..].iter().all(|pair| pair.frequency == 0.0));
    let divisors = divisors.as_ref().expect("stored divisors");
    assert_eq!(divisors.weight.name, "rope_freqs.weight");
    assert_eq!(divisors.bases.len(), 256);
    assert!((divisors.bases[100] - 1e6f64.powf(-200.0 / 512.0)).abs() < 1e-15);
    // No weight is reordered.
    let KeyValue::Owned { key, .. } = &full.key_value else {
        panic!("owning layer");
    };
    assert!(full.query.transforms.is_empty() && key.transforms.is_empty());
}

/// 12B, 26B and 31B take full-layer values from their raw keys.
#[test]
fn values_are_keys_on_full_layers_without_a_value_projection() {
    for (model, layers, period, window) in [
        ("12b", 48u32, 6u32, 1024u64),
        ("26b-a4b", 30, 6, 1024),
        ("31b", 60, 6, 1024),
    ] {
        let definition = inspect(&target(model)).unwrap();
        assert!(definition.decoder.entry.per_layer.is_none());
        for layer in 0..layers {
            let full = (layer + 1) % period == 0;
            let KeyValue::Owned { value, domain, .. } = &attention(&definition, layer).key_value
            else {
                panic!("{model}: every layer owns its history");
            };
            assert_eq!(matches!(value, ValueSource::Key), full, "{model} {layer}");
            let expected = if full {
                HistoryDomain::Token
            } else {
                HistoryDomain::Window { tokens: window }
            };
            assert_eq!(*domain, expected);
        }
    }
}

/// Image rows see each other only on the window layers of the models whose
/// released configurations set `use_bidirectional_attention = "vision"`
/// (12B, 26B-A4B, 31B); E2B and E4B, the per-layer-input models, are causal
/// everywhere, and full layers are causal in every model.
#[test]
fn image_rows_are_bidirectional_only_on_window_layers_of_the_larger_models() {
    for (model, bidirectional_windows) in [
        ("e2b", false),
        ("e4b", false),
        ("12b", true),
        ("26b-a4b", true),
        ("31b", true),
    ] {
        let definition = inspect(&target(model)).unwrap();
        for (index, sublayer) in definition.decoder.sublayers() {
            let Operator::Attention(attention) = &sublayer.op else {
                continue;
            };
            // Sharing layers carry no domain; every catalog window layer is
            // 256 wide and every full layer 512.
            let window = matches!(
                &attention.key_value,
                KeyValue::Owned {
                    domain: HistoryDomain::Window { .. },
                    ..
                }
            ) || attention.width == 256;
            let expected = if window && bidirectional_windows {
                MediaRowAttention::Bidirectional
            } else {
                MediaRowAttention::Causal
            };
            assert_eq!(attention.media_rows, expected, "{model} {index:?}");
        }
    }
}

#[test]
fn routed_layers_are_parallel_dense_and_routed_branches() {
    let definition = inspect(&target("26b-a4b")).unwrap();
    for block in &definition.decoder.blocks {
        let [_, feed_forward] = block.sublayers.as_slice() else {
            panic!("attention and feed-forward sublayers");
        };
        assert_eq!(feed_forward.input, InputNorm::None);
        let OutputForm::ScaledPostNorm { norm, .. } = &feed_forward.output else {
            panic!("scaled post norm");
        };
        assert!(norm.weight.name.ends_with("post_ffw_norm.weight"));
        let Operator::Parallel(branches) = &feed_forward.op else {
            panic!("parallel branches");
        };
        let [dense, routed] = branches.as_slice() else {
            panic!("two branches");
        };
        let Operator::DenseFfn(dense_op) = &dense.op else {
            panic!("dense branch");
        };
        assert_eq!(dense_op.intermediate, 2112);
        let Operator::RoutedFfn(routed_op) = &routed.op else {
            panic!("routed branch");
        };
        assert_eq!(
            (
                routed_op.experts,
                routed_op.selected,
                routed_op.intermediate
            ),
            (128, 8, 704)
        );
        assert_eq!(
            routed_op.router.normalization,
            magnitude_family_contracts::RouteNormalization::Sum
        );
        let magnitude_family_contracts::RouterInput::Residual(router_norm) =
            &routed_op.router.input
        else {
            panic!("router reads its own residual norm");
        };
        assert_eq!(
            router_norm.weight.transforms,
            [ImportTransform::Scale {
                factor: 1.0 / 2816f64.sqrt()
            }]
        );
        let FeedForwardUp::Gated { gate, up, .. } = &routed_op.expert_up else {
            panic!("gated experts");
        };
        assert_eq!(gate.name, up.name);
        assert_eq!(gate.shape, [128, 704, 2816]);
        assert_eq!(
            up.transforms,
            [ImportTransform::Rows(
                magnitude_family_contracts::RowRange {
                    start: 704,
                    rows: 704
                }
            )]
        );
        assert!(routed_op.expert_scale.is_some());
    }
}

#[test]
fn rejects_missing_tensors_unknown_metadata_wrong_shapes_and_unbound_roles() {
    let mut missing = target("e2b");
    missing
        .tensors
        .retain(|tensor| tensor.name != "blk.3.attn_k_norm.weight");
    assert_eq!(
        inspect(&missing).unwrap_err(),
        "missing Gemma weight \"blk.3.attn_k_norm.weight\""
    );

    let mut unknown = target("e2b");
    unknown.metadata.push(Metadata {
        name: "gemma4.attn_logit_softcapping".into(),
        value: Value::Scalar(Scalar::Float(50.0)),
    });
    assert_eq!(
        inspect(&unknown).unwrap_err(),
        "Gemma artifact carries uninterpreted metadata gemma4.attn_logit_softcapping"
    );

    let mut wrong = target("e2b");
    let query = wrong
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "blk.4.attn_q.weight")
        .unwrap();
    query.shape = vec![2048, 1536];
    assert!(inspect(&wrong)
        .unwrap_err()
        .starts_with("Gemma weight \"blk.4.attn_q.weight\": expected [4096, 1536]"));

    let mut unbound = target("e2b");
    let mut extra = unbound.tensors[0].clone();
    extra.name = "blk.20.attn_k.weight".into();
    unbound.tensors.push(extra);
    assert_eq!(
        inspect(&unbound).unwrap_err(),
        "Gemma artifact contains unbound weight role \"blk.20.attn_k.weight\""
    );

    let mut pattern = target("e2b");
    let flags = pattern
        .metadata
        .iter_mut()
        .find(|entry| entry.name == "gemma4.attention.sliding_window_pattern")
        .unwrap();
    flags.value = Value::Array(vec![Scalar::Bool(true); 34]);
    assert_eq!(
        inspect(&pattern).unwrap_err(),
        "the Gemma window pattern must hold one flag per layer"
    );
}

#[test]
fn encoder_projectors_bind_sandwich_blocks_pooling_and_axial_rotary() {
    // (model, decoder width, blocks, tower width, heads, clipped, standardized)
    for (model, output, depth, hidden, heads, clipped, standardized) in [
        ("e2b", 1536, 16, 768, 12, true, false),
        ("e4b", 2560, 16, 768, 12, true, false),
        ("26b-a4b", 2816, 27, 1152, 16, false, true),
        ("31b", 5376, 27, 1152, 16, false, true),
    ] {
        let projector = header(&format!("gemma-4-{model}-it-qat__projector.json"));
        let vision = describe_projector(&projector, output).unwrap();
        assert_eq!((vision.hidden, vision.output_hidden), (hidden, output), "{model}");
        assert_eq!(vision.blocks.len(), depth, "{model}");
        assert_eq!(vision.preprocessing.merge, 3);
        assert_eq!(
            vision.preprocessing.resize,
            VisionResize::PatchBudget { max_patches: 2520 }
        );
        assert_eq!(vision.preprocessing.mean, [0.5; 3]);
        assert_eq!(
            vision.stem.positions().sampling,
            PositionSampling::Axes { length: 10240 }
        );
        let block = &vision.blocks[0];
        let attention = &block.attention;
        let width = hidden / heads;
        assert_eq!((attention.heads, attention.width), (heads, width));
        assert_eq!(attention.scale, VisionAttentionScale::Unit);
        assert_eq!(attention.rotary_base, 100.0);
        assert!(attention.value_norm.is_some() && block.attention_post_norm.is_some());
        assert_eq!(attention.query.clamp.is_some(), clipped, "{model}");
        assert!(matches!(block.feedforward.up, VisionUp::Gated { .. }));
        assert_eq!(block.feedforward.activation, VisionActivation::GeluTanh);
        // HF's axial rotary halves (column pairs, then row pairs) are
        // reordered into the pair-(i, i + W/2) layout on q, k and their norms.
        let quarter = width / 4;
        let order: Vec<u64> = (0..quarter)
            .chain(2 * quarter..3 * quarter)
            .chain(quarter..2 * quarter)
            .chain(3 * quarter..4 * quarter)
            .collect();
        let permuted = [ImportTransform::PermuteRows { order }];
        assert_eq!(attention.query.weight.transforms, permuted);
        assert_eq!(attention.key.weight.transforms, permuted);
        assert_eq!(attention.query_norm.as_ref().unwrap().weight.transforms, permuted);
        assert!(attention.value.weight.transforms.is_empty());
        assert_eq!(vision.merger.standardize.is_some(), standardized, "{model}");
        assert!(matches!(
            vision.merger.reduction,
            CellReduction::Average { scale } if scale == (hidden as f64).sqrt()
        ));
    }
    let model = inspect_components(
        &target("e2b"),
        Some(&header("gemma-4-e2b-it-qat__projector.json")),
        identity(),
    )
    .unwrap();
    assert!(model.vision.is_some());
}

#[test]
fn the_unified_projector_is_a_normalized_patch_projection() {
    let projector = header("gemma-4-12b-it-qat__projector.json");
    let vision = describe_projector(&projector, 3840).unwrap();
    assert!(vision.blocks.is_empty());
    assert_eq!((vision.preprocessing.patch, vision.preprocessing.merge), (48, 1));
    assert_eq!(
        vision.preprocessing.resize,
        VisionResize::PatchBudget { max_patches: 280 }
    );
    assert_eq!(vision.patch_row_width().unwrap(), 6912);
    assert!(matches!(vision.stem, VisionStem::NormalizedPatch { .. }));
    assert_eq!(
        vision.stem.positions().sampling,
        PositionSampling::Axes { length: 1120 }
    );
    assert_eq!(describe_projector(&projector, 2560).unwrap_err().to_string(),
        "projector width differs from the decoder width");
}

#[test]
fn text_rows_take_their_position_on_one_axis() {
    let definition = inspect(&target("e2b")).unwrap();
    let tokens = vec![TokenId(2), TokenId(105), TokenId(9)];
    let plan = |tokens: Vec<TokenId>| {
        let layout = InputLayout::new(tokens.len(), Vec::new()).unwrap();
        TokenPlan::new(tokens, layout).unwrap()
    };
    let prepared = inputs::input_adapter(None)
        .prepare(&definition, plan(tokens.clone()), &[])
        .unwrap();
    assert_eq!(prepared.tokens(), tokens.as_slice());
    assert_eq!(prepared.coordinates(), &[[0; 3], [1; 3], [2; 3]]);
    assert_eq!(prepared.continuation(), 3);
    assert_eq!(definition.inputs.coordinate_axes, 1);
}

#[test]
fn feed_forward_up_is_gated_gelu_tanh() {
    let definition = inspect(&target("e4b")).unwrap();
    let Operator::DenseFfn(dense) = &definition.decoder.blocks[0].sublayers[1].op else {
        panic!("dense feed-forward");
    };
    assert!(matches!(
        dense.up,
        FeedForwardUp::Gated {
            activation: ActivationFunction::GeluTanh,
            ..
        }
    ));
}
