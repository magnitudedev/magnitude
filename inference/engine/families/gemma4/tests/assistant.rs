//! Interpretation of Google's Gemma 4 QAT assistant drafters against their
//! catalog targets (`inference/catalog/headers/`). Each
//! assistant directory carries the metadata and tensor directory published in
//! `mtp-gemma-4-<size>-it.gguf` beside its target, without tokenizer keys.

use magnitude_artifacts::{
    ArtifactIdentity, PackageIdentity,
    gguf::{Array, ByteOrder, Directory, Encoding, Metadata, Scalar, TensorDescriptor, Value},
};
use magnitude_family_common::headers::{self, directory as header};
use magnitude_family_contracts::{
    EmbeddingScale, ExitNorm, HeadChain, HeadComponent, HeadHistory, HeadInput, HistoryDomain,
    KeyValue, ModelDefinition, Operator, OutputForm, ProposalReadoutPolicy, SublayerIndex,
    VocabularyProjection,
};
use magnitude_family_gemma4::{ASSISTANT_ARCHITECTURE, inspect_assistant, inspect_components};

const VOCABULARY: u64 = 262_144;
const WINDOW_HEAD: u64 = 256;
const FULL_HEAD: u64 = 512;

/// The geometry a published assistant declares.
struct Published {
    model: &'static str,
    width: u64,
    heads: u64,
    /// Key/value heads of each of its four layers (three window, one full).
    kv_heads: [u64; 4],
    feed_forward: u64,
    window: u64,
    target_width: u64,
}

const PUBLISHED: [Published; 5] = [
    Published {
        model: "e2b",
        width: 256,
        heads: 4,
        kv_heads: [1; 4],
        feed_forward: 2048,
        window: 512,
        target_width: 1536,
    },
    Published {
        model: "e4b",
        width: 256,
        heads: 4,
        kv_heads: [2; 4],
        feed_forward: 2048,
        window: 512,
        target_width: 2560,
    },
    Published {
        model: "12b",
        width: 1024,
        heads: 16,
        kv_heads: [8, 8, 8, 1],
        feed_forward: 8192,
        window: 1024,
        target_width: 3840,
    },
    Published {
        model: "26b-a4b",
        width: 1024,
        heads: 16,
        kv_heads: [8, 8, 8, 2],
        feed_forward: 8192,
        window: 1024,
        target_width: 2816,
    },
    Published {
        model: "31b",
        width: 1024,
        heads: 32,
        kv_heads: [16, 16, 16, 4],
        feed_forward: 8192,
        window: 1024,
        target_width: 5376,
    },
];

fn published(model: &str) -> &'static Published {
    PUBLISHED.iter().find(|p| p.model == model).unwrap()
}

fn target(model: &str) -> ModelDefinition {
    inspect_components(
        &header(&format!("gemma-4-{model}-it-qat__target-gguf_q4.json")),
        None,
        PackageIdentity {
            target: ArtifactIdentity([7; 32]),
            projector: None,
        },
    )
    .unwrap()
}

fn assistant(p: &Published) -> Directory {
    let unsigned = |value| Value::Scalar(Scalar::Unsigned(value));
    let float = |value| Value::Scalar(Scalar::Float(value));
    let metadata = [
        ("block_count", unsigned(4)),
        ("context_length", unsigned(131_072)),
        ("embedding_length", unsigned(p.width)),
        ("feed_forward_length", unsigned(p.feed_forward)),
        ("attention.head_count", unsigned(p.heads)),
        (
            "attention.head_count_kv",
            Value::Array(p.kv_heads.to_vec().into()),
        ),
        ("rope.freq_base", float(1_000_000.0)),
        ("rope.freq_base_swa", float(10_000.0)),
        ("attention.layer_norm_rms_epsilon", float(1e-6)),
        ("attention.key_length", unsigned(FULL_HEAD)),
        ("attention.value_length", unsigned(FULL_HEAD)),
        ("attention.sliding_window", unsigned(p.window)),
        ("attention.shared_kv_layers", unsigned(4)),
        ("embedding_length_per_layer_input", unsigned(0)),
        (
            "attention.sliding_window_pattern",
            Value::Array(vec![true, true, true, false].into()),
        ),
        ("attention.key_length_swa", unsigned(WINDOW_HEAD)),
        ("attention.value_length_swa", unsigned(WINDOW_HEAD)),
        ("rope.dimension_count", unsigned(FULL_HEAD)),
        ("rope.dimension_count_swa", unsigned(WINDOW_HEAD)),
        ("embedding_length_out", unsigned(p.target_width)),
        ("nextn_predict_layers", unsigned(4)),
    ];
    let w = p.width;
    let mut tensors = vec![
        ("rope_freqs.weight".to_owned(), vec![FULL_HEAD / 2]),
        ("token_embd.weight".to_owned(), vec![VOCABULARY, w]),
    ];
    for layer in 0..4 {
        let head = if layer == 3 { FULL_HEAD } else { WINDOW_HEAD };
        let rows = p.heads * head;
        tensors.extend(
            [
                ("attn_norm.weight", vec![w]),
                ("layer_output_scale.weight", vec![1]),
                ("ffn_down.weight", vec![w, p.feed_forward]),
                ("ffn_gate.weight", vec![p.feed_forward, w]),
                ("ffn_up.weight", vec![p.feed_forward, w]),
                ("post_attention_norm.weight", vec![w]),
                ("post_ffw_norm.weight", vec![w]),
                ("ffn_norm.weight", vec![w]),
                ("attn_output.weight", vec![w, rows]),
                ("attn_q_norm.weight", vec![head]),
                ("attn_q.weight", vec![rows, w]),
            ]
            .into_iter()
            .map(|(name, shape)| (format!("blk.{layer}.{name}"), shape)),
        );
    }
    tensors.extend([
        ("output_norm.weight".to_owned(), vec![w]),
        (
            "nextn.post_projection.weight".to_owned(),
            vec![p.target_width, w],
        ),
        (
            "nextn.pre_projection.weight".to_owned(),
            vec![w, 2 * p.target_width],
        ),
    ]);
    Directory {
        version: 3,
        byte_order: ByteOrder::Little,
        alignment: 32,
        data_offset: 0,
        metadata: std::iter::once(Metadata {
            name: "general.architecture".into(),
            value: Value::Scalar(Scalar::String(ASSISTANT_ARCHITECTURE.into())),
        })
        .chain(metadata.into_iter().map(|(key, value)| Metadata {
            name: format!("{ASSISTANT_ARCHITECTURE}.{key}"),
            value,
        }))
        .collect(),
        tensors: tensors
            .into_iter()
            .map(|(name, shape)| TensorDescriptor {
                name,
                nbytes: shape.iter().product::<u64>() * 2,
                shape,
                encoding: Encoding::F16,
                offset: 0,
            })
            .collect(),
    }
}

fn key(name: &str) -> String {
    format!("{ASSISTANT_ARCHITECTURE}.{name}")
}

#[test]
fn trained_cluster_readout_preserves_canonical_tokens_and_rejects_invalid_ordering() {
    let published = published("e2b");
    let target = target(published.model);
    let mut directory = assistant(published);
    directory.metadata.extend([
        Metadata {
            name: key("output.cluster_count"),
            value: Value::Scalar(Scalar::Unsigned(2048)),
        },
        Metadata {
            name: key("output.cluster_top_k"),
            value: Value::Scalar(Scalar::Unsigned(32)),
        },
        Metadata {
            name: key("output.cluster_token_order"),
            value: Value::Array((0..VOCABULARY).rev().collect::<Vec<u64>>().into()),
        },
    ]);
    directory.tensors.push(TensorDescriptor {
        name: "output_centroids.weight".into(),
        shape: vec![2048, published.width],
        encoding: Encoding::BF16,
        nbytes: 2048 * published.width * 2,
        offset: 0,
    });
    let head = inspect_assistant(&directory, &target).unwrap();
    let ProposalReadoutPolicy::Clustered {
        token_order,
        selected_clusters,
        ..
    } = &head.blocks[0].readout.policy
    else {
        panic!("trained cluster policy must be retained");
    };
    assert_eq!(*selected_clusters, 32);
    assert_eq!(
        (token_order[0].0, token_order.last().unwrap().0),
        (VOCABULARY as u32 - 1, 0)
    );
    let ordering = directory
        .metadata
        .iter_mut()
        .find(|value| value.name == key("output.cluster_token_order"))
        .unwrap();
    let Value::Array(Array::Unsigned(values)) = &mut ordering.value else {
        unreachable!()
    };
    values[1] = values[0];
    assert!(
        inspect_assistant(&directory, &target)
            .unwrap_err()
            .to_string()
            .contains("repeats a token")
    );
}

#[test]
fn every_published_assistant_is_a_draft_head_of_its_qat_target() {
    for p in &PUBLISHED {
        let target = target(p.model);
        let head = inspect_assistant(&assistant(p), &target)
            .unwrap_or_else(|error| panic!("{}: {error}", p.model));
        assert_eq!(
            (head.history, head.component, head.depth()),
            (HeadHistory::Target, HeadComponent::Draft, 1),
            "{}",
            p.model
        );
        let block = &head.blocks[0];
        assert_eq!(block.width(&target.decoder), p.width);
        let HeadInput::Projected {
            pre,
            embedding_scale,
        } = &block.input
        else {
            panic!("an assistant projects its joined input");
        };
        assert_eq!(pre.shape, [p.width, 2 * p.target_width]);
        // The target's embedding enters scaled as the target scales it.
        assert_eq!(*embedding_scale, EmbeddingScale::SqrtHidden);
        assert_eq!(*embedding_scale, target.decoder.entry.scale);
        // Its own full-vocabulary readout, with no soft cap of its own.
        let VocabularyProjection::Own { table } = &block.readout.projection else {
            panic!("an assistant reads out through its own table");
        };
        assert_eq!(table.shape, [VOCABULARY, p.width]);
        assert!(matches!(block.output_norm, ExitNorm::Rms(_)));
        let HeadChain::Projected { post } = &block.chain else {
            panic!("an assistant chains through its post projection");
        };
        assert_eq!(post.shape, [p.target_width, p.width]);
        // Window layers read the target's second-to-last layer, the full
        // layer its last; none stores keys or values.
        let last = target.decoder.blocks.len() as u32 - 1;
        let sources = block
            .blocks
            .iter()
            .map(|block| {
                let Operator::Attention(attention) = &block.sublayers[0].op else {
                    panic!("attention sublayer");
                };
                let KeyValue::Shared { source } = &attention.key_value else {
                    panic!("an assistant layer owns no history");
                };
                *source
            })
            .collect::<Vec<_>>();
        assert_eq!(
            sources,
            [last - 1, last - 1, last - 1, last].map(|block| SublayerIndex { block, sublayer: 0 }),
            "{}",
            p.model
        );
        assert_eq!(
            target.decoder.history_domain(sources[0]),
            Some(HistoryDomain::Window { tokens: p.window })
        );
        assert_eq!(
            target.decoder.history_domain(sources[3]),
            Some(HistoryDomain::Token)
        );
        assert!(matches!(
            block.blocks[0].sublayers[1].output,
            OutputForm::ScaledPostNorm { .. }
        ));
        let definition = ModelDefinition {
            head: Some(head),
            ..target
        };
        definition.validate().unwrap();
        assert!(definition.deferred_forms().is_empty());
    }
}

#[test]
fn an_assistant_of_another_target_is_rejected() {
    let e4b = target("e4b");
    let error = |draft: &Directory, target: &ModelDefinition| {
        inspect_assistant(draft, target).unwrap_err().to_string()
    };
    // Another size's assistant: the target width differs.
    assert_eq!(
        error(&assistant(published("e2b")), &e4b),
        "Gemma assistant drafts for a target of width 1536, not 2560"
    );
    // The 12B and 26B assistants share their width; 26B's full layer reads
    // two key/value heads, the 12B target's last layer one.
    let mut wrong_heads = assistant(published("26b-a4b"));
    headers::set(
        &mut wrong_heads,
        &key("embedding_length_out"),
        Value::Scalar(Scalar::Unsigned(3840)),
    );
    headers::reshape_tensor(
        &mut wrong_heads,
        "nextn.pre_projection.weight",
        &[1024, 7680],
    );
    headers::reshape_tensor(
        &mut wrong_heads,
        "nextn.post_projection.weight",
        &[3840, 1024],
    );
    assert_eq!(
        error(&wrong_heads, &target("12b")),
        "shared attention history disagrees with its source layer"
    );
    // Another window than the layer it reads keeps.
    let mut window = assistant(published("e4b"));
    headers::set(
        &mut window,
        &key("attention.sliding_window"),
        Value::Scalar(Scalar::Unsigned(1024)),
    );
    assert!(error(&window, &e4b).contains("reads target layer"));
    // An assistant storing its own keys is not one that reads its target's.
    let mut keys = assistant(published("e4b"));
    headers::add_tensor(&mut keys, "blk.0.attn_k.weight", &[512, 256]);
    assert_eq!(
        error(&keys, &e4b),
        "artifact contains unbound weight roles [\"blk.0.attn_k.weight\"]"
    );
    let mut partial = assistant(published("e4b"));
    headers::set(
        &mut partial,
        &key("nextn_predict_layers"),
        Value::Scalar(Scalar::Unsigned(1)),
    );
    assert_eq!(
        error(&partial, &e4b),
        "every Gemma assistant layer is a prediction layer reading its target's history"
    );
    let mut unknown = assistant(published("e4b"));
    headers::set(
        &mut unknown,
        &key("final_logit_softcapping"),
        Value::Scalar(Scalar::Float(30.0)),
    );
    assert!(error(&unknown, &e4b).contains("unknown metadata"));
    let mut other = assistant(published("e4b"));
    other.metadata[0].value = Value::Scalar(Scalar::String("dflash".into()));
    assert!(error(&other, &e4b).contains("unsupported Gemma assistant GGUF architecture"));
    let mut foreign = e4b.clone();
    foreign.family.0 = "qwen35".into();
    assert!(error(&assistant(published("e4b")), &foreign).contains("drafts for a Gemma 4 target"));
}

/// The real published files, read through the engine's own header reader.
/// They live on the test hosts, so the test runs only when
/// `GEMMA4_ASSISTANT_GGUF` lists them as colon-separated `model=path` pairs
/// (`model` as in the catalog header names, such as `26b-a4b`).
#[test]
#[ignore = "needs GEMMA4_ASSISTANT_GGUF on a host with the published assistants"]
fn downloaded_assistants_inspect_against_their_targets() {
    let pairs = std::env::var("GEMMA4_ASSISTANT_GGUF").expect("GEMMA4_ASSISTANT_GGUF lists files");
    for pair in pairs.split(':') {
        let (model, path) = pair.split_once('=').expect("model=path");
        let headers = magnitude_artifacts::PackageHeaders::open(path, None)
            .unwrap_or_else(|error| panic!("{path}: {error}"));
        let head = inspect_assistant(headers.target(), &target(model))
            .unwrap_or_else(|error| panic!("{path}: {error}"));
        assert_eq!(
            head.blocks[0].width(&target(model).decoder),
            published(model).width
        );
    }
}
