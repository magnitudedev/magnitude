use magnitude_artifacts::{
    gguf::{Directory, Scalar, Value},
    ArtifactIdentity, PackageIdentity,
};
use magnitude_family_common::headers::{self, add_tensor, set as set_metadata};
use magnitude_family_contracts::{
    ActivationFunction, AttentionGate, FeedForwardUp, HeadNorm, HistoryDomain, ImportTransform,
    InputNorm, KeyValue, ModelDefinition, ModelFamily, Operator, OutputForm, Rotary,
};
use magnitude_artifacts::UnsupportedReason;
use magnitude_family_contracts::FamilyError;
use magnitude_family_llama::{admit, inspect_components, Error, Feature, LlamaFamily};

const VARIANTS: [&str; 2] = [
    "minicpm5-2b__target-gguf_q4.json",
    "minicpm5-2b__target-gguf_q8.json",
];

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    }
}

fn inspect(directory: &Directory) -> Result<ModelDefinition, Error> {
    inspect_components(directory, None, identity())
}

fn minicpm5() -> Directory {
    headers::directory(VARIANTS[0])
}

#[test]
fn every_minicpm5_variant_binds_the_admitted_feature_set() {
    let definitions = VARIANTS.map(|file| {
        inspect(&headers::directory(file)).unwrap_or_else(|error| panic!("{file}: {error}"))
    });
    // Variants differ only in encodings, which the definition does not carry.
    assert_eq!(definitions[0], definitions[1]);
    let model = &definitions[0];
    assert_eq!(model.family.0, "llama");
    assert!(model.deferred_forms().is_empty());
    assert!(model.head.is_none() && model.vision.is_none());
    assert_eq!(model.inputs.coordinate_axes, 1);
    let decoder = &model.decoder;
    assert_eq!(
        (decoder.hidden, decoder.vocabulary, decoder.context_limit),
        (2048, 130560, 131072)
    );
    assert_eq!(decoder.entry.embedding.name, "token_embd.weight");
    assert_eq!(decoder.exit.output.name, "output.weight");
    assert_eq!(decoder.blocks.len(), 42);
    for block in &decoder.blocks {
        let [mixer, feed_forward] = block.sublayers.as_slice() else {
            panic!("a llama block is [attention, feed-forward]");
        };
        assert!(
            matches!(mixer.input, InputNorm::Rms(ref norm) if (norm.epsilon - 1e-6).abs() < 1e-12)
        );
        assert!(matches!(mixer.output, OutputForm::Residual));
        assert!(matches!(feed_forward.output, OutputForm::Residual));
        let Operator::Attention(attention) = &mixer.op else {
            panic!("mixer is attention");
        };
        assert_eq!(
            (attention.heads, attention.kv_heads, attention.width),
            (16, 2, 128)
        );
        assert_eq!(attention.gate, AttentionGate::None);
        assert_eq!(attention.query_norm, HeadNorm::None);
        assert_eq!(attention.scale, 1.0 / 128f64.sqrt());
        let KeyValue::Owned {
            key,
            key_norm,
            domain,
            ..
        } = &attention.key_value
        else {
            panic!("every layer owns its history");
        };
        assert_eq!(*key_norm, HeadNorm::None);
        assert_eq!(*domain, HistoryDomain::Token);
        assert_eq!(key.shape, [256, 2048]);
        assert_eq!(attention.query.shape, [2048, 2048]);
        // Query and key rows move from adjacent pairs to half-split pairs;
        // values and outputs are consumed as stored.
        for descriptor in [&attention.query, key] {
            let [ImportTransform::PermuteRows { order }] = descriptor.transforms.as_slice() else {
                panic!("{} carries one row permutation", descriptor.name);
            };
            assert_eq!(order.len(), 128);
            assert_eq!(&order[..3], &[0, 2, 4]);
            assert_eq!(&order[64..67], &[1, 3, 5]);
            assert_eq!(
                descriptor.transformed_shape(&descriptor.shape).unwrap(),
                descriptor.shape
            );
        }
        assert!(attention.output.transforms.is_empty());
        let Rotary::Table { pairs, .. } = &attention.rotary else {
            panic!("full-width rotary table");
        };
        assert_eq!(pairs.len(), 64);
        assert_eq!(pairs[0].frequency, 1.0);
        assert!((pairs[1].frequency - 5e6f64.powf(-2.0 / 128.0)).abs() < 1e-15);
        assert!(pairs.iter().all(|pair| pair.amplitude == 1.0));
        let Operator::DenseFfn(ffn) = &feed_forward.op else {
            panic!("feed-forward is dense");
        };
        assert_eq!(ffn.intermediate, 6144);
        assert!(matches!(
            ffn.up,
            FeedForwardUp::Gated {
                activation: ActivationFunction::Silu,
                ..
            }
        ));
    }
    assert!(LlamaFamily.recognizes(&minicpm5()));
    assert!(LlamaFamily.media_placeholder(model).is_none());
}

#[test]
fn claims_no_other_catalog_component() {
    for (model, role, file) in headers::index() {
        let expected = model == "minicpm5-2b" && role.starts_with("target-");
        assert_eq!(
            LlamaFamily.recognizes(&headers::directory(&file)),
            expected,
            "{file}"
        );
    }
}

#[test]
fn features_outside_the_admitted_set_are_each_refused_by_name() {
    let mut scaled = minicpm5();
    set_metadata(
        &mut scaled,
        "llama.rope.scaling.type",
        Value::Scalar(Scalar::String("yarn".into())),
    );
    assert_eq!(
        admit(&scaled),
        Err(Error::Unadmitted(vec![Feature::RopeScaling("yarn".into())]))
    );

    let mut per_layer = minicpm5();
    set_metadata(
        &mut per_layer,
        "llama.attention.head_count_kv",
        Value::Array(vec![Scalar::Unsigned(2); 42]),
    );
    assert_eq!(
        admit(&per_layer),
        Err(Error::Unadmitted(vec![Feature::PerLayerMetadata(
            "llama.attention.head_count_kv".into()
        )]))
    );

    let mut foreign = minicpm5();
    set_metadata(
        &mut foreign,
        "minicpm.scale_emb",
        Value::Scalar(Scalar::Float(12.0)),
    );
    assert_eq!(
        admit(&foreign),
        Err(Error::Unadmitted(vec![Feature::UnknownMetadata(
            "minicpm.scale_emb".into()
        )]))
    );

    for role in [
        "blk.0.attn_q.bias",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "rope_freqs.weight",
    ] {
        let mut extra = minicpm5();
        add_tensor(&mut extra, role, &[128]);
        assert_eq!(
            admit(&extra),
            Err(Error::Unadmitted(vec![Feature::Tensor(role.into())]))
        );
    }

    let mut tied = minicpm5();
    tied.tensors.retain(|tensor| tensor.name != "output.weight");
    assert_eq!(admit(&tied), Err(Error::Unadmitted(vec![Feature::TiedOutput])));

    // Every refused feature is named at once, and the family error carries
    // each as a reason.
    let mut several = minicpm5();
    add_tensor(&mut several, "rope_freqs.weight", &[64]);
    several.tensors.retain(|tensor| tensor.name != "output.weight");
    set_metadata(
        &mut several,
        "llama.rope.scaling.type",
        Value::Scalar(Scalar::String("llama3".into())),
    );
    let error = FamilyError::from(inspect(&several).unwrap_err());
    assert_eq!(
        error.reasons,
        [
            UnsupportedReason::RopeScaling {
                kind: "llama3".into()
            },
            UnsupportedReason::UnrecognizedTensor {
                name: "rope_freqs.weight".into()
            },
            UnsupportedReason::TiedOutputHead,
        ]
    );
    // The family still claims the architecture, so the refusal names the
    // features instead of disowning the file.
    assert!(LlamaFamily.recognizes(&several));

    let mut other = minicpm5();
    set_metadata(
        &mut other,
        "general.architecture",
        Value::Scalar(Scalar::String("mistral3".into())),
    );
    assert_eq!(
        admit(&other),
        Err(Error::Architecture(Some("mistral3".into())))
    );
}

#[test]
fn recognized_packages_are_bound_strictly() {
    let mut missing = minicpm5();
    missing
        .tensors
        .retain(|tensor| tensor.name != "blk.41.ffn_up.weight");
    assert_eq!(
        inspect(&missing),
        Err(Error::MissingWeight("blk.41.ffn_up.weight".into()))
    );

    let mut misshapen = minicpm5();
    misshapen
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "blk.3.attn_k.weight")
        .unwrap()
        .shape = vec![512, 2048];
    assert_eq!(
        inspect(&misshapen),
        Err(Error::WeightShape {
            name: "blk.3.attn_k.weight".into(),
            expected: vec![256, 2048],
            received: vec![512, 2048],
        })
    );

    let mut beyond = minicpm5();
    add_tensor(&mut beyond, "blk.42.attn_norm.weight", &[2048]);
    assert_eq!(
        inspect(&beyond),
        Err(Error::UnboundWeights(vec!["blk.42.attn_norm.weight".into()]))
    );

    let mut unsized_vocabulary = minicpm5();
    unsized_vocabulary
        .metadata
        .retain(|item| item.name != "llama.vocab_size");
    assert!(matches!(
        inspect(&unsized_vocabulary),
        Err(Error::Metadata { key, .. }) if key == "llama.vocab_size"
    ));

    let mut textual = minicpm5();
    set_metadata(
        &mut textual,
        "llama.attention.layer_norm_rms_epsilon",
        Value::Scalar(Scalar::String("1e-6".into())),
    );
    assert!(matches!(inspect(&textual), Err(Error::Metadata { .. })));

    let mut partial = minicpm5();
    set_metadata(
        &mut partial,
        "llama.rope.dimension_count",
        Value::Scalar(Scalar::Unsigned(64)),
    );
    assert!(matches!(inspect(&partial), Err(Error::Geometry(_))));

    let target = minicpm5();
    assert_eq!(
        inspect_components(&target, Some(&target), identity()),
        Err(Error::Projector)
    );
}
