use magnitude_artifacts::{
    gguf::{Directory, Encoding, Metadata, Scalar, TensorDescriptor, Value},
    ArtifactIdentity, PackageIdentity, TokenId,
};
use magnitude_family_contracts::{
    AttentionGate, BlockLayout, DraftDefinition, DraftEmbedding, DraftMethod, DraftVariant, HeadNorm, HistoryDomain, ImportTransform, KeyValue, ModelDefinition, ModelFamily, Operator,
    Rotary, SublayerIndex, TapPoint,
};
use magnitude_family_common::headers;
use magnitude_family_dflash::{inspect, recognize, recognizes, Error};
use magnitude_family_lfm2::Lfm2Family;
use magnitude_family_llama::LlamaFamily;
use magnitude_family_muse_glimmer::MuseGlimmerFamily;
use magnitude_family_nemotron_h::NemotronHFamily;
use magnitude_family_qwen35::Qwen35Family;

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    }
}

/// Every catalog draft with its target's family and header.
const PAIRS: [(&str, &str); 6] = [
    ("qwen3.6-35b-a3b__draft.json", "qwen3.6-35b-a3b__target-gguf_q4.json"),
    ("muse-glimmer-30b__draft.json", "muse-glimmer-30b__target-gguf_q4.json"),
    (
        "nemotron-3.5-lightning-30b-a3b__draft.json",
        "nemotron-3.5-lightning-30b-a3b__target-gguf_nvfp4-qat.json",
    ),
    ("lfm2.5-2.6b__draft.json", "lfm2.5-2.6b__target-gguf_q4.json"),
    ("lfm2.5-8b-a1b__draft.json", "lfm2.5-8b-a1b__target-gguf_q4.json"),
    ("minicpm5-2b__draft.json", "minicpm5-2b__target-gguf_q4.json"),
];

fn family(target: &str) -> &'static dyn ModelFamily {
    match target.split("__").next().unwrap() {
        "qwen3.6-35b-a3b" => &Qwen35Family,
        "muse-glimmer-30b" => &MuseGlimmerFamily,
        "nemotron-3.5-lightning-30b-a3b" => &NemotronHFamily,
        "lfm2.5-2.6b" | "lfm2.5-8b-a1b" => &Lfm2Family,
        "minicpm5-2b" => &LlamaFamily,
        other => panic!("no family for {other}"),
    }
}

fn target(file: &str) -> ModelDefinition {
    family(file)
        .inspect(&headers::directory(file), None, identity())
        .unwrap_or_else(|error| panic!("{file}: {error}"))
}

/// Bind a draft through its target family's layer mapping (Nemotron-H's GGUF
/// layers are single sublayers; every other family's are blocks).
fn draft_of(draft: &str, target_file: &str) -> Result<DraftDefinition, Error> {
    let definition = target(target_file);
    let directory = headers::directory(draft);
    let family = family(target_file);
    inspect(&directory, &definition, &|layer| {
        family.layer_entry(&definition, layer)
    })
}

fn block_taps(layers: &[u32]) -> Vec<TapPoint> {
    layers
        .iter()
        .map(|&block| TapPoint::Sublayer(SublayerIndex { block, sublayer: 0 }))
        .collect()
}

fn domains(draft: &DraftDefinition) -> Vec<HistoryDomain> {
    draft
        .blocks
        .iter()
        .map(|block| match &block.sublayers[0].op {
            Operator::Attention(attention) => match &attention.key_value {
                KeyValue::Owned { domain, .. } => domain.clone(),
                KeyValue::Shared { .. } => panic!("a draft layer owns its history"),
            },
            _ => panic!("a draft layer starts with attention"),
        })
        .collect()
}

#[test]
fn every_catalog_draft_binds_against_its_target() {
    for (draft, target_file) in PAIRS {
        let definition = draft_of(draft, target_file).unwrap_or_else(|error| panic!("{draft}: {error}"));
        let mut model = target(target_file);
        model.draft = Some(definition.clone());
        model.validate().unwrap_or_else(|error| panic!("{draft}: {error:?}"));
        assert!(model.deferred_forms().is_empty(), "{draft}");
        for block in &definition.blocks {
            let [mixer, feed_forward] = block.sublayers.as_slice() else {
                panic!("{draft}: a draft layer is [attention, feed-forward]");
            };
            let Operator::Attention(attention) = &mixer.op else {
                panic!("{draft}: attention first");
            };
            assert!(matches!(attention.gate, AttentionGate::None));
            assert!(matches!(attention.query_norm, HeadNorm::Rms(_)));
            assert!(matches!(attention.rotary, Rotary::Table { .. }));
            assert!(matches!(feed_forward.op, Operator::DenseFfn(_)));
        }
    }
}

#[test]
fn qwen_dflash_taps_eight_layers_with_five_sliding_windows() {
    let draft = draft_of(PAIRS[0].0, PAIRS[0].1).unwrap();
    assert_eq!(draft.method, DraftMethod::DFlash);
    assert_eq!(draft.layout, BlockLayout::MaskSlots);
    assert_eq!((draft.block_size, draft.max_proposals()), (16, 15));
    assert_eq!(draft.mask_token, TokenId(248077));
    assert_eq!(draft.taps, block_taps(&[2, 7, 12, 17, 23, 28, 33, 38]));
    assert_eq!(draft.fusion.shape, [2048, 8 * 2048]);
    assert_eq!(draft.embedding, DraftEmbedding::Target);
    // A draft's window of W keeps |q − k| ≤ W: W + 1 positions.
    let window = HistoryDomain::Window { tokens: 4097 };
    assert_eq!(
        domains(&draft),
        [window.clone(), window.clone(), window.clone(), window.clone(), window, HistoryDomain::Token]
    );
    let Operator::Attention(attention) = &draft.blocks[0].sublayers[0].op else {
        unreachable!()
    };
    assert_eq!((attention.heads, attention.kv_heads, attention.width), (32, 8, 128));
    let Rotary::Table { pairs, divisors: None } = &attention.rotary else {
        panic!("plain rotary table");
    };
    assert_eq!(pairs.len(), 64);
    assert_eq!(pairs[1].frequency, 1e7f64.powf(-2.0 / 128.0));
    assert!(pairs.iter().all(|pair| pair.amplitude == 1.0));
}

#[test]
fn muse_dflash_windows_every_layer() {
    let draft = draft_of(PAIRS[1].0, PAIRS[1].1).unwrap();
    assert_eq!(draft.taps, block_taps(&[2, 14, 26, 38, 50]));
    assert_eq!(domains(&draft), vec![HistoryDomain::Window { tokens: 2049 }; 5]);
    assert_eq!(draft.fusion.shape, [6656, 5 * 6656]);
    assert_eq!(draft.mask_token, TokenId(201818));
}

#[test]
fn nemotron_dflash_has_yarn_its_own_embedding_and_second_level_scales() {
    let draft = draft_of(PAIRS[2].0, PAIRS[2].1).unwrap();
    assert!(matches!(draft.embedding, DraftEmbedding::Own(ref table) if table.shape == [131072, 2688]));
    assert_eq!(domains(&draft), vec![HistoryDomain::Token; 6]);
    assert_eq!(draft.taps.len(), 6);
    assert_eq!(draft.taps.last(), Some(&TapPoint::Exit));
    assert_eq!(
        draft.fusion.transforms,
        [ImportTransform::ScaleByTensor {
            tensor: "fc.scale".into()
        }]
    );
    let Operator::Attention(attention) = &draft.blocks[0].sublayers[0].op else {
        unreachable!()
    };
    assert_eq!((attention.kv_heads, attention.width), (2, 128));
    let Rotary::Table { pairs, .. } = &attention.rotary else {
        panic!("rotary table");
    };
    let amplitude = 1.0 + 0.1 * 128f64.ln();
    assert!(pairs.iter().all(|pair| pair.amplitude == amplitude));
    // YaRN keeps the fastest pair and interpolates the slowest by 1/128.
    assert_eq!(pairs[0].frequency, 1.0);
    let slowest = 10_000f64.powf(-2.0 * 63.0 / 128.0);
    assert!((pairs[63].frequency - slowest / 128.0).abs() < 1e-18);
    let Operator::DenseFfn(ffn) = &draft.blocks[0].sublayers[1].op else {
        unreachable!()
    };
    assert_eq!(
        ffn.down.transforms,
        [ImportTransform::ScaleByTensor {
            tensor: "blk.0.ffn_down.scale".into()
        }]
    );
}

#[test]
fn lfm2_dspark_samples_from_the_anchor_with_markov_and_confidence() {
    for (draft, target_file) in &PAIRS[3..5] {
        let definition = draft_of(draft, target_file).unwrap();
        assert_eq!(definition.layout, BlockLayout::AnchorFirst);
        assert_eq!((definition.block_size, definition.max_proposals()), (9, 9));
        assert_eq!(definition.mask_token, TokenId(125017));
        let DraftMethod::DSpark { markov, confidence } = &definition.method else {
            panic!("{draft}: DSpark");
        };
        assert_eq!(markov.rank, 256);
        assert_eq!(markov.embedding.shape, [128000, 256]);
        assert_eq!(markov.projection.shape, [128000, 256]);
        assert_eq!(confidence.weight.shape, [2048 + 256]);
        assert!(confidence.weight.transforms.is_empty());
        assert_eq!(confidence.bias.shape, [1]);
        let Operator::Attention(attention) = &definition.blocks[0].sublayers[0].op else {
            unreachable!()
        };
        assert_eq!(attention.width, 64);
    }
    assert_eq!(
        draft_of(PAIRS[3].0, PAIRS[3].1).unwrap().taps,
        block_taps(&[3, 10, 18, 22, 28])
    );
    assert_eq!(
        draft_of(PAIRS[4].0, PAIRS[4].1).unwrap().taps,
        block_taps(&[3, 7, 11, 15, 19])
    );
}

#[test]
fn minicpm5_dspark_flattens_its_stored_confidence_row() {
    let draft = draft_of(PAIRS[5].0, PAIRS[5].1).unwrap();
    assert_eq!(draft.layout, BlockLayout::AnchorFirst);
    assert_eq!((draft.block_size, draft.mask_token), (7, TokenId(75982)));
    assert_eq!(draft.taps, block_taps(&[2, 11, 21, 31, 40]));
    let DraftMethod::DSpark { markov, confidence } = &draft.method else {
        panic!("DSpark");
    };
    assert_eq!(markov.embedding.shape, [130560, 256]);
    assert_eq!(confidence.weight.shape, [2304]);
    assert_eq!(confidence.weight.transforms, [ImportTransform::Flatten]);
}

#[test]
fn drafts_and_targets_are_recognized_by_disjoint_families() {
    let targets: [&dyn ModelFamily; 5] = [
        &Qwen35Family,
        &MuseGlimmerFamily,
        &NemotronHFamily,
        &Lfm2Family,
        &LlamaFamily,
    ];
    for (_, role, file) in headers::index() {
        if file.contains("allshards") {
            continue;
        }
        let directory = headers::directory(&file);
        let draft = role == "draft";
        let architecture = directory
            .value("general.architecture")
            .and_then(Value::string)
            .unwrap()
            .to_owned();
        // The DeepSeek V4 DSpark uses DS4 blocks (deferred with its target).
        if architecture == "dflash" && file.starts_with("deepseek") {
            assert!(!recognizes(&directory), "{file}");
            continue;
        }
        assert_eq!(recognizes(&directory), draft, "{file}");
        if draft {
            for family in targets {
                assert!(!family.recognizes(&directory), "{} claims {file}", family.name());
            }
        }
    }
}

fn qwen_draft() -> (Directory, ModelDefinition) {
    (headers::directory(PAIRS[0].0), target(PAIRS[0].1))
}

fn inspect_qwen(directory: &Directory, model: &ModelDefinition) -> Result<DraftDefinition, Error> {
    inspect(directory, model, &|layer| Qwen35Family.layer_entry(model, layer))
}

#[test]
fn a_missing_weight_is_refused() {
    let (mut directory, model) = qwen_draft();
    directory.tensors.retain(|tensor| tensor.name != "blk.3.attn_k_norm.weight");
    assert_eq!(
        inspect_qwen(&directory, &model).unwrap_err(),
        Error::MissingWeight("blk.3.attn_k_norm.weight".into())
    );
}

#[test]
fn an_unbound_or_foreign_weight_is_refused() {
    let (mut directory, model) = qwen_draft();
    directory.tensors.push(TensorDescriptor {
        name: "output.weight".into(),
        shape: vec![248320, 2048],
        encoding: Encoding::F32,
        offset: 0,
        nbytes: 0,
    });
    assert_eq!(
        recognize(&directory).unwrap_err(),
        Error::UnboundWeight("output.weight".into())
    );
    // A layer beyond `block_count` is a known role no layer binds.
    let (mut directory, _) = qwen_draft();
    directory.tensors.push(TensorDescriptor {
        name: "blk.6.attn_norm.weight".into(),
        shape: vec![2048],
        encoding: Encoding::F32,
        offset: 0,
        nbytes: 0,
    });
    assert_eq!(
        inspect_qwen(&directory, &model).unwrap_err(),
        Error::UnboundWeight("blk.6.attn_norm.weight".into())
    );
}

#[test]
fn unknown_metadata_is_refused() {
    let (mut directory, model) = qwen_draft();
    directory.metadata.push(Metadata {
        name: "dflash.rope.dimension_count".into(),
        value: Value::Scalar(Scalar::Unsigned(64)),
    });
    assert_eq!(
        inspect_qwen(&directory, &model).unwrap_err(),
        Error::UnknownMetadata("dflash.rope.dimension_count".into())
    );
}

#[test]
fn a_wrong_shape_is_refused() {
    let (mut directory, model) = qwen_draft();
    let tensor = directory
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "fc.weight")
        .unwrap();
    tensor.shape = vec![2048, 7 * 2048];
    assert_eq!(
        inspect_qwen(&directory, &model).unwrap_err(),
        Error::WeightShape {
            name: "fc.weight".into(),
            expected: vec![2048, 8 * 2048],
            received: vec![2048, 7 * 2048],
        }
    );
}

#[test]
fn a_draft_for_another_target_is_refused() {
    // The Qwen draft against the MiniCPM5 target: the taps fit, the width
    // does too, but the mask token is outside its vocabulary.
    let directory = headers::directory(PAIRS[0].0);
    let model = target(PAIRS[5].1);
    assert!(matches!(
        inspect(&directory, &model, &|layer| LlamaFamily.layer_entry(&model, layer)),
        Err(Error::Metadata { .. })
    ));
    // The Muse draft against the Qwen target: the width differs.
    let directory = headers::directory(PAIRS[1].0);
    let model = target(PAIRS[0].1);
    assert!(matches!(
        inspect_qwen(&directory, &model),
        Err(Error::Target(_))
    ));
}

#[test]
fn a_tap_beyond_the_target_is_refused() {
    let (mut directory, model) = qwen_draft();
    let layers = directory
        .metadata
        .iter_mut()
        .find(|item| item.name == "dflash.target_layers")
        .unwrap();
    layers.value = Value::Array(vec![Scalar::Unsigned(2), Scalar::Unsigned(41)]);
    assert!(matches!(
        inspect_qwen(&directory, &model),
        Err(Error::Target(_))
    ));
}

/// The published Qwen3.6 draft stores its tapped layers as INT32.
#[test]
fn signed_tap_indices_bind_as_unsigned_ones_do() {
    let (mut directory, model) = qwen_draft();
    let unsigned = inspect_qwen(&directory, &model).unwrap();
    let layers = directory
        .metadata
        .iter_mut()
        .find(|item| item.name == "dflash.target_layers")
        .unwrap();
    let Value::Array(values) = &layers.value else {
        panic!("target layers are an array");
    };
    layers.value = Value::Array(
        values
            .iter()
            .map(|value| match value {
                Scalar::Unsigned(layer) => Scalar::Signed(*layer as i64),
                other => other.clone(),
            })
            .collect(),
    );
    assert_eq!(inspect_qwen(&directory, &model).unwrap(), unsigned);
    let layers = directory
        .metadata
        .iter_mut()
        .find(|item| item.name == "dflash.target_layers")
        .unwrap();
    layers.value = Value::Array(vec![Scalar::Signed(-1)]);
    assert!(matches!(
        inspect_qwen(&directory, &model),
        Err(Error::Metadata { .. })
    ));
}

/// The released Qwen3.8 27B drafts (not catalog entries; headers of the exact
/// qualification artifacts) against the Qwen3.8 27B target.
const QWEN38_TARGET: &str = "qwen3.8-27b__target-gguf_q4.json";
const QWEN38_DSPARK: &str = "qwen3.8-27b__draft-dspark.json";
const QWEN38_DFLASH2: &str = "qwen3.8-27b__draft-dflash2.json";

fn qwen38(draft: &str) -> Result<DraftDefinition, Error> {
    let model = Qwen35Family
        .inspect(&headers::directory(QWEN38_TARGET), None, identity())
        .unwrap();
    inspect(&headers::directory(draft), &model, &|layer| {
        Qwen35Family.layer_entry(&model, layer)
    })
}

#[test]
fn qwen38_dspark_drafts_seven_from_the_anchor_with_yarn() {
    let draft = qwen38(QWEN38_DSPARK).unwrap();
    assert_eq!(draft.method.variant(), DraftVariant::DSpark);
    assert_eq!(draft.layout, BlockLayout::AnchorFirst);
    assert_eq!((draft.block_size, draft.max_proposals()), (7, 7));
    assert_eq!(draft.mask_token, TokenId(248077));
    assert_eq!(draft.taps, block_taps(&[5, 17, 29, 41, 53]));
    let Operator::Attention(attention) = &draft.blocks[0].sublayers[0].op else {
        unreachable!()
    };
    assert_eq!((attention.heads, attention.kv_heads, attention.width), (40, 8, 128));
    // YaRN folds into the pairs: the scaled low frequencies differ from the
    // plain table's.
    let Rotary::Table { pairs, .. } = &attention.rotary else {
        panic!("rotary table");
    };
    assert_eq!(pairs.len(), 64);
    assert_ne!(pairs[63].frequency, 1e7f64.powf(-126.0 / 128.0));
}

#[test]
fn qwen38_dflash2_convolves_every_sublayer_and_selects_a_path() {
    let draft = qwen38(QWEN38_DFLASH2).unwrap();
    assert_eq!(draft.method.variant(), DraftVariant::DFlash2);
    assert_eq!(draft.layout, BlockLayout::MaskSlots);
    assert_eq!((draft.block_size, draft.max_proposals()), (8, 7));
    assert_eq!(draft.mask_token, TokenId(248070));
    assert_eq!(draft.taps, block_taps(&[6, 20, 34, 48, 62]));
    assert_eq!(domains(&draft), vec![HistoryDomain::Window { tokens: 2049 }; 5]);
    let DraftMethod::DFlash2 {
        kernel,
        group,
        convolutions,
        selector,
    } = &draft.method
    else {
        unreachable!()
    };
    assert_eq!((*kernel, *group), (2, 16));
    assert_eq!(convolutions.len(), 5);
    for layer in convolutions {
        for convolution in [&layer.attention, &layer.feed_forward] {
            assert_eq!(convolution.base.shape, [2, 2, 5120]);
            assert_eq!(convolution.projection.shape, [1280, 5120]);
        }
    }
    assert_eq!((selector.rank, selector.top_k), (256, 16));
    assert_eq!(selector.hidden.shape, [256, 5120]);
    assert_eq!(selector.predecessor.shape, [248320, 256]);
    assert_eq!(selector.successor.shape, [248320, 256]);
    // One text axis over the whole head is the plain table.
    let Operator::Attention(attention) = &draft.blocks[0].sublayers[0].op else {
        unreachable!()
    };
    let Rotary::Table { pairs, divisors: None } = &attention.rotary else {
        panic!("plain rotary table");
    };
    assert_eq!(pairs.len(), 64);
    assert_eq!(pairs[1].frequency, 1e7f64.powf(-2.0 / 128.0));
}

#[test]
fn dflash2_roles_are_all_or_nothing() {
    let model = Qwen35Family
        .inspect(&headers::directory(QWEN38_TARGET), None, identity())
        .unwrap();
    let bind = |directory: &Directory| {
        inspect(directory, &model, &|layer| Qwen35Family.layer_entry(&model, layer))
    };
    let mut missing = headers::directory(QWEN38_DFLASH2);
    headers::remove_tensor(&mut missing, "blk.4.ffn_conv_proj.weight");
    assert_eq!(
        bind(&missing).unwrap_err(),
        Error::MissingWeight("blk.4.ffn_conv_proj.weight".into())
    );
    // Selector metadata without the selector roles is not a DFlash draft.
    let mut stray = headers::directory(QWEN38_DSPARK);
    headers::set(
        &mut stray,
        "dflash.selector_rank",
        Value::Scalar(Scalar::Unsigned(256)),
    );
    assert!(matches!(bind(&stray), Err(Error::Geometry(_))));
    // Sections that rotate a pair by an image axis are refused.
    let mut sectioned = headers::directory(QWEN38_DFLASH2);
    headers::set(
        &mut sectioned,
        "dflash.rope.dimension_sections",
        Value::Array(vec![Scalar::Unsigned(32), Scalar::Unsigned(32), Scalar::Unsigned(0), Scalar::Unsigned(0)]),
    );
    assert!(matches!(bind(&sectioned), Err(Error::Geometry(_))));
}
