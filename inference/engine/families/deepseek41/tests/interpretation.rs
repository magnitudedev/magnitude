//! Slice A (layer 0 of DeepSeek V4.1 Flash with its embedding and head, the
//! converter's `--part text --layers 0`) and slice B (`--layers 0,1`, layer
//! 1 an Engram layer) interpreted from their real headers
//! (`engine/fixtures/assessment/headers`), each bound form checked against
//! the header's own metadata and tensor directory.

use magnitude_artifacts::{
    gguf::{self, Array, Directory, Scalar, Value},
    ArtifactIdentity, PackageIdentity,
};
use magnitude_family_common::{architecture_keys, Metadata};
use magnitude_family_contracts::{
    Attention, BlockElement, BlockRounding, BlockScale, ExitNorm, ExitSite, ExpertSelection, FeedForwardUp, GluLimit, HeadNorm,
    HistoryReads, HistoryRows, HistorySource, HyperConnectionMixing, HyperConnectionWeights,
    InputNorm, ProjectedRows, RowProducer, RowRounding, TokenDomain,
    ImportTransform, ModelDefinition, ModelFamily, NGramGate, NGramHash, NGramInput, NGramMedia,
    Operator, OutputForm, RowRange, ScaleCodes, OutputProjection, OutputRotary,
    QueryProjection, ResidualForm, Rotary, RouteNormalization, RoutedFfn, ScoreFunction,
    SharedExpertGate, ValueNorm, ValueSource,
};
use magnitude_family_deepseek41::{rotary, DeepSeek41Family, FAMILY_ID, ROTARY_LAYOUT};
use std::io::Read;

const SLICE_A: &[u8] =
    include_bytes!("../../../fixtures/assessment/headers/DeepSeek-V4.1-Flash-1017-slice-A.gguf.gz");
const SLICE_B: &[u8] =
    include_bytes!("../../../fixtures/assessment/headers/DeepSeek-V4.1-Flash-1017-slice-B.gguf.gz");

fn slice() -> Directory {
    header(SLICE_A)
}

/// A gzipped header fixture's directory.
fn header(compressed: &[u8]) -> Directory {
    let mut header = Vec::new();
    flate2::read::GzDecoder::new(compressed)
        .read_to_end(&mut header)
        .unwrap();
    gguf::read_header(&header).unwrap()
}

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    }
}

fn inspect(
    directory: &Directory,
) -> Result<ModelDefinition, magnitude_family_contracts::FamilyError> {
    DeepSeek41Family.inspect(directory, None, identity())
}

/// `deepseek41.<key>`, as the header states it.
fn value<'a>(directory: &'a Directory, key: &str) -> &'a Value {
    directory
        .value(&format!("{FAMILY_ID}.{key}"))
        .unwrap_or_else(|| panic!("the slice states {key}"))
}

fn integer(directory: &Directory, key: &str) -> u64 {
    value(directory, key).unsigned().unwrap()
}

fn number(directory: &Directory, key: &str) -> f64 {
    match value(directory, key) {
        Value::Scalar(Scalar::Float(value)) => *value,
        other => panic!("{key} is {other:?}"),
    }
}

fn numbers(directory: &Directory, key: &str) -> Vec<f64> {
    match value(directory, key) {
        Value::Array(Array::Float(values)) => values.clone(),
        other => panic!("{key} is {other:?}"),
    }
}

/// The stored shape (outermost first) of `name`.
fn shape(directory: &Directory, name: &str) -> Vec<u64> {
    directory.tensor(name).unwrap().shape.clone()
}

#[test]
fn slice_a_binds_every_layer_zero_form_from_its_header() {
    let directory = slice();
    let definition = inspect(&directory).unwrap();
    assert_eq!(definition.deferred_forms(), []);
    let decoder = &definition.decoder;
    let (hidden, heads, width) = (
        integer(&directory, "embedding_length"),
        integer(&directory, "attention.head_count"),
        integer(&directory, "attention.key_length"),
    );
    let epsilon = number(&directory, "attention.layer_norm_rms_epsilon");
    let streams = integer(&directory, "hyper_connection.count");
    assert_eq!(decoder.hidden, hidden);
    assert_eq!(
        decoder.vocabulary,
        shape(&directory, "token_embd.weight")[0]
    );
    assert_eq!(decoder.context_limit, integer(&directory, "context_length"));
    assert_eq!(
        decoder.residual,
        ResidualForm::HyperConnections(magnitude_family_contracts::HyperConnections {
            streams,
            mixing: HyperConnectionMixing::Sinkhorn {
                iterations: integer(&directory, "hyper_connection.sinkhorn_iterations"),
                epsilon: number(&directory, "hyper_connection.epsilon"),
                norm_epsilon: epsilon,
            },
        })
    );
    assert_eq!(
        decoder.blocks.len() as u64,
        integer(&directory, "block_count")
    );
    let ExitNorm::HyperConnectionHead(exit) = &decoder.exit.norm else {
        panic!("the exit collapses the streams");
    };
    assert_eq!(exit.site, ExitSite::PreviousSite);
    assert_eq!(
        exit.norm.as_ref().unwrap().weight.name,
        "output_norm.weight"
    );
    assert_eq!(decoder.exit.output.name, "output.weight");

    let [attention, feed_forward] = &decoder.blocks[0].sublayers[..] else {
        panic!("a block is attention then its feed-forward");
    };
    for (sublayer, site, norm) in [
        (attention, "blk.0.hc_attn", "blk.0.attn_norm.weight"),
        (feed_forward, "blk.0.hc_ffn", "blk.0.ffn_norm.weight"),
    ] {
        let InputNorm::HyperConnectionMix(input) = &sublayer.input else {
            panic!("{site} mixes the streams");
        };
        let HyperConnectionWeights::Sinkhorn(weights) = &input.site else {
            panic!("{site} is a Sinkhorn site");
        };
        let mixes = (2 + streams) * streams;
        assert_eq!(weights.function.shape, [mixes, streams * hidden]);
        assert_eq!(
            weights.function.shape,
            shape(&directory, &format!("{site}_fn.weight"))
        );
        assert_eq!(weights.base.shape, [mixes]);
        assert_eq!(weights.scale.shape, [3]);
        let input_norm = input.norm.as_ref().unwrap();
        assert_eq!(
            (input_norm.weight.name.as_str(), input_norm.epsilon),
            (norm, epsilon)
        );
        assert_eq!(sublayer.output, OutputForm::Scatter);
    }

    let Operator::Attention(attention) = &attention.op else {
        panic!("the mixer is attention");
    };
    check_attention(&directory, attention, heads, width, epsilon);
    let Operator::RoutedFfn(routed) = &feed_forward.op else {
        panic!("the feed-forward is routed");
    };
    check_routed(&directory, routed, hidden);
}

fn check_attention(
    directory: &Directory,
    attention: &Attention,
    heads: u64,
    width: u64,
    epsilon: f64,
) {
    let rank = integer(directory, "attention.q_lora_rank");
    let (groups, output_rank) = (
        integer(directory, "attention.output_group_count"),
        integer(directory, "attention.output_lora_rank"),
    );
    assert_eq!(
        (attention.heads, attention.kv_heads, attention.width),
        (heads, 1, width)
    );
    let QueryProjection::LowRank { down, norm, up } = &attention.query else {
        panic!("the query is low-rank");
    };
    assert_eq!(down.shape[0], rank);
    assert_eq!(
        (norm.weight.shape.as_slice(), norm.epsilon),
        (&[rank][..], epsilon)
    );
    assert_eq!(up.shape, [heads * width, rank]);
    // The FP8 projections bind with their grids.
    assert!(!up.transforms.is_empty(), "{up:?}");
    assert_eq!(attention.query_norm, HeadNorm::None);
    let [HistorySource {
        rows:
            HistoryRows::Own(RowProducer::Projected(ProjectedRows {
                key,
                value,
                key_norm,
                value_norm,
                domain,
                rounding,
            })),
        reads: HistoryReads::Visible,
    }] = &attention.history[..]
    else {
        panic!("layer 0 owns its history");
    };
    // The window's rows: E4M3 per 32 values under a UE8M0 scale.
    assert_eq!(
        *rounding,
        RowRounding::Block(BlockRounding {
            element: BlockElement::E4M3,
            block: 32,
            scale: BlockScale::Ue8m0,
            amax_floor: 1e-4,
        })
    );
    assert_eq!(key.shape, shape(directory, "blk.0.attn_kv.weight"));
    assert_eq!(*value, ValueSource::StoredKey);
    let HeadNorm::Rms(key_norm) = key_norm else {
        panic!("the key is normalized");
    };
    assert_eq!(key_norm.weight.name, "blk.0.attn_kv_a_norm.weight");
    assert_eq!(*value_norm, ValueNorm::None);
    assert_eq!(
        *domain,
        TokenDomain::Window {
            tokens: integer(directory, "attention.sliding_window")
        }
    );
    let Rotary::Table {
        pairs,
        layout,
        divisors,
        output,
    } = &attention.rotary
    else {
        panic!("a rotary table");
    };
    let rotated = integer(directory, "rope.dimension_count");
    let base = number(directory, "rope.freq_base");
    assert_eq!(
        (*layout, divisors, *output),
        (ROTARY_LAYOUT, &None, OutputRotary::Inverse)
    );
    assert_eq!(pairs.len() as u64, rotated / 2);
    for (pair, rotation) in pairs.iter().enumerate() {
        assert_eq!(
            rotation.frequency,
            base.powf(-2.0 * pair as f64 / rotated as f64)
        );
        assert_eq!(rotation.amplitude, 1.0);
    }
    assert_eq!(attention.scale, (width as f64).powf(-0.5));
    assert_eq!(attention.sinks.as_ref().unwrap().shape, [heads]);
    let OutputProjection::GroupedLowRank {
        groups: g,
        rank: r,
        a,
        b,
    } = &attention.output
    else {
        panic!("the output is grouped low-rank");
    };
    assert_eq!((*g, *r), (groups, output_rank));
    assert_eq!(a.shape, [groups * output_rank, heads * width / groups]);
    assert_eq!(a.shape, shape(directory, "blk.0.attn_output_a.weight"));
    assert_eq!(b.shape, shape(directory, "blk.0.attn_output_b.weight"));
}

fn check_routed(directory: &Directory, routed: &RoutedFfn, hidden: u64) {
    let (experts, selected, intermediate) = (
        integer(directory, "expert_count"),
        integer(directory, "expert_used_count"),
        integer(directory, "expert_feed_forward_length"),
    );
    assert_eq!(
        (routed.experts, routed.selected, routed.intermediate),
        (experts, selected, intermediate)
    );
    let router = &routed.router;
    assert_eq!(router.weight.shape, [experts, hidden]);
    assert_eq!(router.score, ScoreFunction::SqrtSoftplus);
    let ExpertSelection::TopK { bias: Some(bias) } = &router.selection else {
        panic!("selection ranks on a bias");
    };
    assert_eq!(bias.name, "blk.0.exp_probs_b.bias");
    assert_eq!(
        router.media_bias.as_ref().unwrap().name,
        "blk.0.exp_probs_b_vl.bias"
    );
    assert_eq!(
        router.normalization,
        RouteNormalization::SumPlusEpsilon(1e-20)
    );
    assert_eq!(router.scale, number(directory, "expert_weights_scale"));
    let limit = |key| GluLimit::new(numbers(directory, key)[0]).unwrap();
    let FeedForwardUp::Gated {
        limit: routed_limit,
        gate,
        ..
    } = &routed.expert_up
    else {
        panic!("the experts are gated");
    };
    assert_eq!(*routed_limit, Some(limit("swiglu_clamp_exp")));
    assert_eq!(gate.shape, [experts, intermediate, hidden]);
    assert_eq!(routed.expert_down.shape, [experts, hidden, intermediate]);
    let shared = routed.shared.as_ref().unwrap();
    assert_eq!(
        shared.intermediate,
        intermediate * integer(directory, "expert_shared_count")
    );
    assert_eq!(shared.up.limit(), Some(limit("swiglu_clamp_shexp")));
    assert_eq!(shared.gate, SharedExpertGate::None);
}

/// V4.1's compressed layers' rotary (θ `compress_rope_freq_base`, YaRN ×16
/// over 65536) is YaRN at unit amplitude: its frequencies are YaRN's
/// interpolated inverse frequencies, computed in F32 at the slice's
/// constants, and its amplitude is 1 (no mscale).
#[test]
fn the_compressed_layers_rotary_is_yarn_at_unit_amplitude() {
    const INVERSE_FREQUENCIES: [f32; 32] = [
        1.000000000e+00,
        6.876560450e-01,
        4.728707969e-01,
        3.251724541e-01,
        2.236067951e-01,
        1.537645608e-01,
        1.057371199e-01,
        7.271077484e-02,
        5.000000075e-02,
        3.438280150e-02,
        2.364354022e-02,
        1.625862345e-02,
        1.118034031e-02,
        7.688228041e-03,
        5.286856554e-03,
        3.635538509e-03,
        2.265624935e-03,
        1.396801323e-03,
        8.496897062e-04,
        5.080819828e-04,
        2.969777852e-04,
        1.681799768e-04,
        9.086784121e-05,
        4.544423064e-05,
        1.953125320e-05,
        5.372312444e-06,
        3.694303359e-06,
        2.540409696e-06,
        1.746928206e-06,
        1.201285613e-06,
        8.260713003e-07,
        5.680529398e-07,
    ];
    let directory = slice();
    let keys = architecture_keys(&directory, FAMILY_ID)
        .map(|(key, _)| key)
        .collect::<Vec<_>>();
    let m = Metadata::new(&directory, FAMILY_ID, &keys).unwrap();
    let Rotary::Table {
        pairs,
        layout,
        output,
        ..
    } = rotary(&m, 2, 512).unwrap()
    else {
        panic!("a rotary table");
    };
    assert_eq!((layout, output), (ROTARY_LAYOUT, OutputRotary::Inverse));
    assert_eq!(pairs.len(), INVERSE_FREQUENCIES.len());
    for (pair, (rotation, expected)) in pairs.iter().zip(INVERSE_FREQUENCIES).enumerate() {
        // The expected values are computed in F32: within two of its units
        // in the last place.
        let ours = rotation.frequency as f32;
        assert!(
            (ours.to_bits() as i64 - expected.to_bits() as i64).abs() <= 2,
            "pair {pair}: {ours:e} against the expected {expected:e}"
        );
        assert_eq!(rotation.amplitude, 1.0, "pair {pair}");
    }
}

/// The published model (the catalog's target, every shard) binds its
/// layers from 20 by their header's source lists: 20 pools at ratio 1
/// (Identity) and produces the candidate pool, 21 reuses its picks, 24
/// indexes its rows and keys (Reindex) restricted to its pool, and 25 reads
/// 20's rows through 24's picks.
#[test]
fn the_full_model_binds_its_reindex_layers_and_candidate_pool() {
    use magnitude_family_contracts::{Candidates, IndexerKeys, Pool, Selection, SublayerIndex};
    let directory = magnitude_family_common::headers::target("deepseek-v4.1-flash:gguf:mxfp4");
    let definition = inspect(&directory).unwrap();
    let layer = |block: u32| SublayerIndex { block, sublayer: 0 };
    let second = |block: u32| match &definition.decoder.blocks[block as usize].sublayers[0].op {
        Operator::Attention(attention) => attention.history[1].clone(),
        _ => panic!("layer {block}'s first sublayer attends"),
    };
    let own = |source: &HistorySource| match &source.reads {
        HistoryReads::Selected(Selection::Own(indexer)) => indexer.as_ref().clone(),
        reads => panic!("an own indexer, not {reads:?}"),
    };
    let producer = second(20);
    let HistoryRows::Own(RowProducer::Pooled(rows)) = &producer.rows else {
        panic!("layer 20 pools its own rows");
    };
    assert_eq!((rows.rate, &rows.pool), (1, &Pool::Identity));
    let produced = own(&producer);
    assert!(matches!(produced.keys, IndexerKeys::Latent { .. }));
    assert_eq!(
        produced.candidates,
        Candidates::Produce {
            blocks: integer(&directory, "attention.indexer.candidate_top_k_blocks"),
            block_units: integer(&directory, "attention.indexer.candidate_block_size"),
        }
    );
    let shared = HistoryRows::Shared { source: layer(20), part: 1 };
    let reuse = |picks: u32| HistorySource {
        rows: shared.clone(),
        reads: HistoryReads::Selected(Selection::Reuse { source: layer(picks), part: 1 }),
    };
    assert_eq!(second(21), reuse(20));
    let reindex = second(24);
    assert_eq!(reindex.rows, shared);
    let restricted = own(&reindex);
    assert_eq!(restricted.keys, IndexerKeys::Shared { source: layer(20), part: 1 });
    assert_eq!(restricted.candidates, Candidates::Restrict { source: layer(20), part: 1 });
    assert_eq!(second(25), reuse(24));
}

/// A compressed layer reads the latest kv and index source at or before
/// it: one with none is invalid.
#[test]
fn a_compressed_layer_without_a_source_is_invalid() {
    let mut directory = slice();
    let ratios = format!("{FAMILY_ID}.attention.compress_ratios");
    set(
        &mut directory,
        &ratios,
        Value::Array(Array::Unsigned(vec![2, 0, 0, 0])),
    );
    let error = inspect(&directory).unwrap_err();
    assert!(
        error.to_string().contains("compressed layer 0 follows no kv and index source"),
        "{error}"
    );
}

/// `deepseek41.<key>`'s nonnegative integers, of whichever integer type the
/// header stores them in.
fn counts(directory: &Directory, key: &str) -> Vec<u64> {
    match value(directory, key) {
        Value::Array(Array::Unsigned(values)) => values.clone(),
        Value::Array(Array::Signed(values)) => values
            .iter()
            .map(|value| u64::try_from(*value).unwrap_or_else(|_| panic!("{key} holds {value}")))
            .collect(),
        _ => panic!("{key} is no integer array"),
    }
}

/// Slice B's Engram layer 1 is the trailing n-gram input of block 0, every
/// constant its header's: the compressed token map and pad, layer 1's
/// multipliers, sizes and offsets,
/// the gridded table, the fused projection's two row ranges and the
/// product gate.
#[test]
fn slice_b_binds_its_engram_layer_from_its_header() {
    let directory = header(SLICE_B);
    let definition = inspect(&directory).unwrap();
    assert_eq!(definition.deferred_forms(), []);
    let decoder = &definition.decoder;
    let (streams, hidden) = (integer(&directory, "hyper_connection.count"), decoder.hidden);
    assert_eq!(decoder.blocks.len(), 2);
    assert_eq!(decoder.blocks[1].sublayers.len(), 2);
    let [_, _, sublayer] = &decoder.blocks[0].sublayers[..] else {
        panic!("block 0 ends with the n-gram input");
    };
    assert_eq!(
        (&sublayer.input, &sublayer.output),
        (&InputNorm::None, &OutputForm::Residual)
    );
    let Operator::NGramInput(input) = &sublayer.op else {
        panic!("an n-gram input");
    };
    let NGramInput {
        hash,
        width,
        table,
        key,
        value: projected,
        gate,
        epsilon,
        mixing,
    } = &**input;
    assert_eq!(counts(&directory, "ple.layers"), [1]);
    let map = counts(&directory, "engram.token_map");
    assert_eq!(map.len(), 129_280);
    assert_eq!(map.len() as u64, decoder.vocabulary);
    let distinct = map.iter().collect::<std::collections::HashSet<_>>().len();
    assert_eq!(distinct, 99_092);
    let pad = integer(&directory, "engram.pad_token_id");
    let ngram = integer(&directory, "ple.ngram_size");
    let NGramHash {
        context,
        heads_per_order,
        multipliers,
        sizes,
        offsets,
        token_map,
        fill,
        boundary,
        media,
    } = hash;
    assert_eq!(*context, ngram - 1);
    assert_eq!(*heads_per_order, integer(&directory, "ple.heads_per_ngram"));
    assert_eq!(*multipliers, counts(&directory, "ple.layer_multipliers"));
    assert_eq!(multipliers.len(), 4);
    assert_eq!(*sizes, counts(&directory, "ple.head_vocab_sizes"));
    assert_eq!(*offsets, counts(&directory, "ple.head_offsets"));
    assert_eq!((sizes.len(), offsets[0]), (24, 0));
    assert_eq!(
        token_map.as_ref().unwrap().iter().map(|id| u64::from(*id)).collect::<Vec<_>>(),
        map
    );
    assert_eq!(hash.hashed_vocabulary(decoder.vocabulary), distinct as u64);
    assert_eq!((*fill, *boundary, *media), (map[pad as usize], None, NGramMedia::Blocking));
    assert_eq!(*width, integer(&directory, "embedding_length_per_layer_input"));
    let rows = sizes.iter().sum::<u64>();
    assert_eq!(rows, 384_006_168);
    assert_eq!(table.name, "per_layer_token_embd.weight");
    assert_eq!(table.shape, [rows, 256]);
    assert_eq!(shape(&directory, "per_layer_token_embd.scale"), [rows, 8]);
    assert_eq!(
        table.transforms,
        [ImportTransform::ScaleBlocks {
            tensor: "per_layer_token_embd.scale".into(),
            tile: [1, 32],
            codes: ScaleCodes::Ue8m0,
        }]
    );
    let wide = streams * hidden;
    let fused = "blk.1.engram_kv.weight";
    assert_eq!(shape(&directory, fused), [wide + hidden, 24 * 256]);
    assert_eq!(shape(&directory, "blk.1.engram_kv.scale"), [(wide + hidden) / 32, 24 * 256 / 32]);
    for (weight, start, count) in [(key, 0, wide), (projected, wide, hidden)] {
        assert_eq!(weight.name, fused);
        assert_eq!(weight.shape, [count, 24 * 256]);
        assert_eq!(
            weight.transforms,
            [
                ImportTransform::ScaleBlocks {
                    tensor: "blk.1.engram_kv.scale".into(),
                    tile: [32, 32],
                    codes: ScaleCodes::Ue8m0,
                },
                ImportTransform::Rows(RowRange { start, rows: count }),
            ]
        );
    }
    let NGramGate::Product(gate) = gate else {
        panic!("the stored product gate");
    };
    assert_eq!(shape(&directory, "blk.1.engram_gate.weight"), [streams, hidden]);
    assert_eq!(
        (gate.name.as_str(), &gate.shape[..], &gate.transforms[..]),
        ("blk.1.engram_gate.weight", &[wide][..], &[ImportTransform::Flatten][..])
    );
    assert_eq!(*epsilon, number(&directory, "attention.layer_norm_rms_epsilon"));
    assert_eq!(*mixing, None);
}

/// The host row rule over slice B's bound constants equals the reference's
/// `ngram_rows` (`families/deepseek41.py`) on the R1 fixtures' rendered
/// tokens, row for row: the fixture holds the reference's rows' digest.
#[test]
fn slice_b_rows_equal_the_reference_rows_of_the_r1_fixtures() {
    use sha2::{Digest, Sha256};
    let definition = inspect(&header(SLICE_B)).unwrap();
    let Operator::NGramInput(input) = &definition.decoder.blocks[0].sublayers[2].op else {
        panic!("an n-gram input");
    };
    let fixtures: serde_json::Value = serde_json::from_str(include_str!("fixtures/engram_rows.json")).unwrap();
    let fixtures = fixtures["fixtures"].as_array().unwrap();
    assert_eq!(fixtures.len(), 2);
    for fixture in fixtures {
        let tokens = fixture["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|token| token.as_u64().unwrap())
            .collect::<Vec<_>>();
        let rows = input.hash.rows(&[], &tokens);
        assert_eq!(rows.len() as u64, fixture["rows"].as_u64().unwrap());
        let digest = Sha256::digest(rows.iter().flat_map(|row| row.to_le_bytes()).collect::<Vec<_>>());
        let hex = digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        assert_eq!(hex, fixture["sha256"].as_str().unwrap(), "{}", fixture["id"]);
        // A chunk boundary inside the prompt reads the tokens before it.
        let split = tokens.len() / 2;
        let mut chunked = input.hash.rows(&[], &tokens[..split]);
        chunked.extend(input.hash.rows(&tokens[..split], &tokens[split..]));
        assert_eq!(chunked, rows);
    }
}

fn set(directory: &mut Directory, name: &str, value: Value) {
    directory.metadata.retain(|entry| entry.name != name);
    directory.metadata.push(gguf::Metadata {
        name: name.into(),
        value,
    });
}
