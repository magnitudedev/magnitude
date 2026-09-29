use magnitude_artifacts::{
    gguf::{Directory, Scalar, Value},
    ArtifactIdentity, InputLayout, PackageIdentity, TokenId,
};
use magnitude_family_contracts::{
    AttentionGate, CellReduction, ExitNorm, GateFunction, GateGranularity, HeadNorm,
    HistoryDomain, ImportTransform, InputNorm, KeyValue, ModelDefinition, ModelFamily, Operator,
    OutputForm, PositionSampling, Rotary, TokenPlan, ValueSource, VisionAttentionSpan, VisionNorm,
    VisionResampling, VisionResize, WeightDescriptor,
};
use magnitude_family_common::headers;
use magnitude_family_muse_glimmer::{inspect_components, recognize, MuseGlimmerFamily};

/// Every catalog variant of the target.
const VARIANTS: [&str; 4] = [
    "muse-glimmer-30b__target-gguf_q4.json",
    "muse-glimmer-30b__target-gguf_q5.json",
    "muse-glimmer-30b__target-gguf_q6.json",
    "muse-glimmer-30b__target-gguf_q8.json",
];

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([0; 32]),
        projector: None,
    }
}

fn inspect(directory: &Directory) -> Result<ModelDefinition, String> {
    inspect_components(directory, None, identity()).map_err(|error| error.to_string())
}

/// The stored-row order of a head whose adjacent rotary pairs become
/// `(i, i + 64)`.
fn paired() -> Vec<u64> {
    (0..64).map(|i| 2 * i).chain((0..64).map(|i| 2 * i + 1)).collect()
}

fn transforms(weight: &WeightDescriptor) -> &[ImportTransform] {
    &weight.transforms
}

#[test]
fn every_catalog_variant_binds_to_the_released_geometry() {
    let epsilon = f64::from(1e-5f32);
    for variant in VARIANTS {
        let directory = headers::directory(variant);
        assert!(recognize(&directory), "{variant}");
        let model = inspect(&directory).unwrap_or_else(|error| panic!("{variant}: {error}"));
        assert_eq!(model.family.0, "muse-glimmer");
        assert!(model.deferred_forms().is_empty());
        let decoder = &model.decoder;
        assert_eq!(
            (decoder.hidden, decoder.vocabulary, decoder.context_limit),
            (6656, 202048, 131072)
        );
        assert_eq!(decoder.entry.norm.map(|norm| norm.epsilon), Some(epsilon));
        assert_eq!(decoder.exit.output.name, "output.weight");
        assert_eq!(decoder.exit.softcap, Some(20.0));
        let ExitNorm::Rms(norm) = &decoder.exit.norm else {
            panic!("final RMS norm");
        };
        assert_eq!(
            transforms(&norm.weight),
            [ImportTransform::Scale {
                factor: f64::from(0.19611613f32)
            }]
        );
        assert_eq!(decoder.blocks.len(), 52);
        let mut windowed = 0;
        for (layer, block) in decoder.blocks.iter().enumerate() {
            let [attention, feed_forward] = block.sublayers.as_slice() else {
                panic!("layer {layer} is attention then feed-forward");
            };
            for sublayer in [attention, feed_forward] {
                assert!(matches!(&sublayer.input, InputNorm::Rms(norm) if norm.epsilon == epsilon));
                assert!(matches!(&sublayer.output, OutputForm::PostNorm(norm) if norm.epsilon == 1e-8));
            }
            assert!(matches!(&feed_forward.op, Operator::DenseFfn(dense) if dense.intermediate == 19968));
            let Operator::Attention(attention) = &attention.op else {
                panic!("layer {layer} mixes with attention");
            };
            assert_eq!(
                (attention.heads, attention.kv_heads, attention.width),
                (32, 2, 128)
            );
            assert!(matches!(
                &attention.gate,
                AttentionGate::Separate {
                    weight,
                    function: GateFunction::Sigmoid,
                    granularity: GateGranularity::Element,
                } if weight.shape == [4096, 6656] && weight.transforms.is_empty()
            ));
            let KeyValue::Owned {
                key,
                value: ValueSource::Projected(value),
                key_norm: HeadNorm::Rms(key_norm),
                domain,
                ..
            } = &attention.key_value
            else {
                panic!("layer {layer} owns projected history");
            };
            let HeadNorm::Rms(query_norm) = &attention.query_norm else {
                panic!("layer {layer} normalizes its queries");
            };
            assert!(value.transforms.is_empty());
            // Full layers are the last of every four; they have no rotary
            // and keep the whole history.
            if layer % 4 == 3 {
                assert_eq!(*domain, HistoryDomain::Token);
                assert_eq!(attention.rotary, Rotary::None);
                for weight in [&attention.query, key, &query_norm.weight, &key_norm.weight] {
                    assert!(weight.transforms.is_empty(), "{}", weight.name);
                }
            } else {
                windowed += 1;
                assert_eq!(*domain, HistoryDomain::Window { tokens: 2048 });
                let Rotary::Table { pairs, .. } = &attention.rotary else {
                    panic!("layer {layer} rotates");
                };
                assert_eq!(pairs.len(), 64);
                for (p, pair) in pairs.iter().enumerate() {
                    assert_eq!(pair.frequency, 500_000f64.powf(-2.0 * p as f64 / 128.0));
                    assert_eq!(pair.amplitude, 1.0);
                }
                for weight in [&attention.query, key, &query_norm.weight, &key_norm.weight] {
                    assert_eq!(
                        transforms(weight),
                        [ImportTransform::PermuteRows { order: paired() }],
                        "{}",
                        weight.name
                    );
                }
            }
        }
        assert_eq!(windowed, 39);
    }
}

#[test]
fn a_per_layer_window_pattern_and_partial_rotary_are_read_from_the_header() {
    let mut directory = headers::directory(VARIANTS[0]);
    let flags = (0..52).map(|layer| Scalar::Bool(layer % 2 == 0)).collect();
    headers::set(
        &mut directory,
        "muse-glimmer.attention.sliding_window_pattern",
        Value::Array(flags),
    );
    headers::set(
        &mut directory,
        "muse-glimmer.rope.dimension_count",
        Value::Scalar(Scalar::Unsigned(64)),
    );
    let model = inspect(&directory).unwrap();
    for (layer, block) in model.decoder.blocks.iter().enumerate() {
        let Operator::Attention(attention) = &block.sublayers[0].op else {
            unreachable!()
        };
        if layer % 2 == 0 {
            assert!(matches!(&attention.rotary, Rotary::Table { pairs, .. } if pairs.len() == 32));
            let order = (0..32)
                .map(|i| 2 * i)
                .chain((0..32).map(|i| 2 * i + 1))
                .chain(64..128)
                .collect();
            assert_eq!(
                attention.query.transforms,
                [ImportTransform::PermuteRows { order }]
            );
        } else {
            assert_eq!(attention.rotary, Rotary::None);
        }
    }
}

#[test]
fn rejects_missing_misshapen_unbound_and_unknown_roles() {
    let base = headers::directory(VARIANTS[0]);
    let rejected = |edit: &dyn Fn(&mut Directory), reason: &str| {
        let mut directory = base.clone();
        edit(&mut directory);
        let error = inspect(&directory).expect_err(reason);
        assert!(error.contains(reason), "{error:?} does not name {reason:?}");
    };
    rejected(
        &|d| headers::remove_tensor(d, "blk.9.post_ffw_norm.weight"),
        "missing weight \"blk.9.post_ffw_norm.weight\"",
    );
    rejected(
        &|d| headers::reshape_tensor(d, "blk.4.attn_gate.weight", &[32, 6656]),
        "weight \"blk.4.attn_gate.weight\": expected [4096, 6656]",
    );
    rejected(
        &|d| headers::reshape_tensor(d, "blk.4.attn_k_norm.weight", &[64]),
        "blk.4.attn_k_norm.weight",
    );
    rejected(
        &|d| headers::add_tensor(d, "blk.0.attn_sinks.weight", &[32]),
        "unbound weight role \"blk.0.attn_sinks.weight\"",
    );
    rejected(
        &|d| {
            headers::set(
                d,
                "muse-glimmer.rope.scaling.type",
                Value::Scalar(Scalar::String("yarn".into())),
            )
        },
        "unknown metadata \"muse-glimmer.rope.scaling.type\"",
    );
    rejected(
        &|d| {
            headers::set(
                d,
                "muse-glimmer.attention.head_count",
                Value::Array(vec![Scalar::Unsigned(32); 52]),
            )
        },
        "muse-glimmer.attention.head_count must be a positive integer",
    );
    rejected(
        &|d| {
            headers::set(
                d,
                "muse-glimmer.attention.sliding_window_pattern",
                Value::Array(vec![Scalar::Bool(true); 51]),
            )
        },
        "one flag per layer",
    );
    rejected(
        &|d| {
            d.metadata
                .retain(|entry| entry.name != "muse-glimmer.logit_scale")
        },
        "missing metadata muse-glimmer.logit_scale",
    );
}

#[test]
fn recognizes_only_the_target() {
    for other in [
        "muse-glimmer-30b__draft.json",
        "muse-glimmer-30b__projector.json",
        "laguna-s-2.1__target-gguf_q4-allshards.json",
    ] {
        assert!(!recognize(&headers::directory(other)), "{other}");
    }
}

#[test]
fn the_projector_windows_sparse_layers_and_shuffles_channel_outer() {
    let directory = headers::directory(VARIANTS[0]);
    let projector = headers::directory("muse-glimmer-30b__projector.json");
    let model = inspect_components(&directory, Some(&projector), identity()).unwrap();
    let vision = model.vision.unwrap();
    assert_eq!((vision.hidden, vision.output_hidden), (1536, 6656));
    assert_eq!(vision.window, Some(32));
    assert_eq!(vision.stem.frames(), 1);
    assert_eq!(
        vision.stem.positions().sampling,
        PositionSampling::PixelCenters { side: 32 }
    );
    assert_eq!(
        vision.preprocessing.resize,
        VisionResize::CellBudget { max_cells: 4096 }
    );
    assert_eq!(vision.preprocessing.resampling, VisionResampling::Lanczos);
    // Blocks 3, 7, …, 47 and the last attend globally; the other 37 within
    // their window.
    let global = vision
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| block.attention.span == VisionAttentionSpan::Full)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert_eq!(
        global,
        (3..48).step_by(4).chain([49]).collect::<Vec<_>>()
    );
    // llama.cpp's adjacent rotary pairs are reordered on q/k rows and biases.
    let order: Vec<u64> = (0..48).map(|i| 2 * i).chain((0..48).map(|i| 2 * i + 1)).collect();
    let attention = &vision.blocks[0].attention;
    for linear in [&attention.query, &attention.key] {
        assert_eq!(transforms(&linear.weight), [ImportTransform::PermuteRows { order: order.clone() }]);
        assert_eq!(
            transforms(linear.bias.as_ref().unwrap()),
            [ImportTransform::PermuteRows { order: order.clone() }]
        );
    }
    assert!(attention.value.weight.transforms.is_empty());
    assert_eq!(vision.merger.reduction, CellReduction::Interleave);
    assert_eq!(vision.merger.stages.len(), 3);
    assert!(matches!(
        vision.merger.output_norm,
        Some(VisionNorm::Rms { weight: None, epsilon }) if epsilon == f64::from(1e-5f32)
    ));
}

#[test]
fn text_input_takes_absolute_positions_and_refuses_media() {
    let family = MuseGlimmerFamily;
    let model = family
        .inspect(&headers::directory(VARIANTS[0]), None, identity())
        .unwrap();
    struct NoMarkers;
    impl magnitude_family_contracts::MarkerTokens for NoMarkers {
        fn marker(&self, text: &str) -> Result<TokenId, magnitude_family_contracts::FamilyError> {
            panic!("text input reads no marker {text:?}")
        }
    }
    assert_eq!(family.media_placeholder(&model), None);
    let adapter = family.input_adapter(&model, &NoMarkers).unwrap();
    let tokens = vec![TokenId(200000), TokenId(7)];
    let plan = TokenPlan::new(tokens.clone(), InputLayout::new(2, Vec::new()).unwrap()).unwrap();
    let input = adapter.prepare(&model, plan, &[]).unwrap();
    assert_eq!(input.tokens(), tokens);
    assert_eq!(input.coordinates(), [[0; 3], [1; 3]]);
}
