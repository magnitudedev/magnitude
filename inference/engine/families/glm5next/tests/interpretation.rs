//! GLM-5.3-Flash's published FP8 file (`magnitudedev/GLM-5.3-Flash-GGUF` at
//! `06087538ebffc3fa8993a316f652166ca3297a41`, `FP8/`, all seven shards),
//! interpreted from its header alone: the pinned artifact `glm-5.3-flash-fp8`
//! of the catalog header export (`MAGNITUDE_CATALOG_HEADERS`).

use magnitude_artifacts::{
    gguf::{self, Array, ByteOrder, Directory, Elements, Encoding, Scalar, TensorDescriptor, Value},
    ArtifactIdentity, PackageIdentity,
};
use magnitude_family_common::headers;
use magnitude_family_contracts::{
    DecayGate, ExitNorm, ExitSite, ExpertSelection, FamilyError, FeedForwardUp, GateFunction,
    GateProjection, GluLimit, HeadComponent, HeadHistory, HistoryReads,
    HyperConnectionMixing, ImportTransform, IndexerInput, IndexerKeys, InputNorm, ModelDefinition,
    ModelFamily, Operator, OutputForm, OutputProjection, Pool, PoolNorm, QueryKeys, RecurrentDecay,
    ResidualForm, Rotary, RouteNormalization, RowPart, ScaleCodes, ScaledActivations, ScoreFunction,
    Selection, SinkhornInput,
};
use magnitude_family_glm5next::{Glm5NextFamily, FAMILY_ID};
use serde_json::Value as Json;

/// A header array of `values`, each read as `scalar`.
fn array<T>(values: Vec<T>, scalar: fn(T) -> Scalar) -> Value {
    Value::Array(Array::of(values.into_iter().map(scalar)).unwrap())
}
use std::collections::{BTreeMap, BTreeSet};

const ARTIFACT: &str = "glm-5.3-flash-fp8";

/// The published file's trunk: 45 blocks, latent attention at every fourth
/// from block 3, KDA elsewhere; the first 3 feed-forwards dense.
const TRUNK: usize = 45;

fn published() -> Directory {
    headers::artifact(ARTIFACT)
}

fn identity() -> PackageIdentity {
    PackageIdentity {
        target: ArtifactIdentity([7; 32]),
        projector: None,
    }
}

fn inspect(directory: &Directory) -> Result<ModelDefinition, FamilyError> {
    Glm5NextFamily.inspect(directory, None, identity())
}

/// An F32 metadata value as the header stores it, widened.
fn f32_value(value: f32) -> f64 {
    f64::from(value)
}

/// Every weight descriptor of `json` (an object with a name, a shape and
/// transforms), by name, with every other stored tensor its imports read
/// (a scale grid, a part and its grid).
fn descriptors(json: &Json, out: &mut BTreeMap<String, BTreeSet<String>>) {
    match json {
        Json::Object(object) => {
            if let (Some(Json::String(name)), Some(_), Some(transforms)) =
                (object.get("name"), object.get("shape"), object.get("transforms"))
            {
                let mut reads = BTreeSet::new();
                references(transforms, &mut reads);
                out.entry(name.clone()).or_default().extend(reads);
                return;
            }
            object.values().for_each(|value| descriptors(value, out));
        }
        Json::Array(values) => values.iter().for_each(|value| descriptors(value, out)),
        _ => {}
    }
}

/// The stored tensors a descriptor's transforms read.
fn references(json: &Json, out: &mut BTreeSet<String>) {
    match json {
        Json::Object(object) => {
            for (key, value) in object {
                match (key.as_str(), value) {
                    ("tensor" | "grid", Json::String(tensor)) => {
                        out.insert(tensor.clone());
                    }
                    _ => references(value, out),
                }
            }
        }
        Json::Array(values) => values.iter().for_each(|value| references(value, out)),
        _ => {}
    }
}

/// Every stored tensor, the MTP head's indexer included, is claimed by exactly
/// the definition: each a weight it names or a tensor one of them reads, and
/// every F8_E4M3 weight reads its own F32 `.scale` grid.
#[test]
fn the_published_fp8_file_binds_every_tensor_to_its_form() {
    let directory = published();
    let definition = inspect(&directory).unwrap();
    let mut bound = BTreeMap::new();
    descriptors(&serde_json::to_value(&definition).unwrap(), &mut bound);
    let claimed = bound
        .iter()
        .flat_map(|(name, reads)| std::iter::once(name).chain(reads))
        .cloned()
        .collect::<BTreeSet<_>>();
    let stored = directory
        .tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(stored.len(), 1727);
    assert_eq!(claimed, stored);
    let mut fp8 = 0;
    for tensor in &directory.tensors {
        if tensor.encoding != Encoding::F8E4M3 {
            continue;
        }
        fp8 += 1;
        let grid = tensor.name.replace(".weight", ".scale");
        let stored_grid = directory.tensor(&grid).unwrap_or_else(|| panic!("{grid} is stored"));
        assert_eq!(stored_grid.encoding, Encoding::F32, "{grid}");
        assert!(bound[&tensor.name].contains(&grid), "{} reads {grid}", tensor.name);
    }
    // The trunk's and the head's. Per latent block (11 and the head's): q_a,
    // q_b, kv_a, output; per routed block (42 and the head's): three stacks
    // and three shared; per dense block (3): three.
    assert_eq!(fp8, 12 * 4 + 43 * 6 + 3 * 3);

    let decoder = &definition.decoder;
    assert_eq!(
        (decoder.hidden, decoder.vocabulary, decoder.context_limit),
        (4096, 154_880, 1_048_576)
    );
    assert_eq!(definition.inputs.coordinate_axes, 1);
    let ResidualForm::HyperConnections(streams) = &decoder.residual else {
        panic!("{FAMILY_ID} mixes hyper-connection streams")
    };
    assert_eq!(streams.streams, 4);
    assert_eq!(
        streams.mixing,
        HyperConnectionMixing::Sinkhorn {
            iterations: 20,
            epsilon: f32_value(1e-6),
            norm_epsilon: f32_value(1e-5),
            input: SinkhornInput::OwnSite,
        }
    );
    let ExitNorm::HyperConnectionHead(exit) = &decoder.exit.norm else {
        panic!("the exit collapses the streams")
    };
    assert_eq!(exit.site, ExitSite::Mean);
    assert_eq!(exit.norm.as_ref().unwrap().weight.name, "output_norm.weight");
    assert_eq!(decoder.exit.output.name, "output.weight");

    assert_eq!(decoder.blocks.len(), TRUNK);
    let limit = GluLimit::new(10.0).unwrap();
    for (index, block) in decoder.blocks.iter().enumerate() {
        let [mixer, feed_forward] = &block.sublayers[..] else {
            panic!("block {index} is a mixer and a feed-forward")
        };
        for sublayer in [mixer, feed_forward] {
            assert!(matches!(sublayer.input, InputNorm::HyperConnectionMix(_)), "block {index}");
            assert_eq!(sublayer.output, OutputForm::Scatter, "block {index}");
        }
        match (&mixer.op, index % 4 == 3) {
            (Operator::GatedDelta(_), false) | (Operator::Attention(_), true) => {}
            (op, _) => panic!("block {index}'s mixer is {op:?}"),
        }
        match (&feed_forward.op, index < 3) {
            (Operator::DenseFfn(dense), true) => {
                assert_eq!(dense.intermediate, 12_288);
                assert_eq!(dense.up.limit(), Some(limit));
            }
            (Operator::RoutedFfn(_), false) => {}
            (op, _) => panic!("block {index}'s feed-forward is {op:?}"),
        }
    }

    // KDA (block 0).
    let Operator::GatedDelta(kda) = &decoder.blocks[0].sublayers[0].op else {
        unreachable!()
    };
    assert_eq!(
        (kda.key_heads, kda.value_heads, kda.width, kda.convolution_width),
        (64, 64, 128, 4)
    );
    assert_eq!(kda.query_key_value.shape, [3 * 8192, 4096]);
    assert_eq!(kda.convolution.shape, [3 * 8192, 4]);
    assert_eq!(kda.key_norm_epsilon, 1e-6);
    let RecurrentDecay::PerChannel { projection, bias, decay, gate } = &kda.decay else {
        panic!("KDA decays per key channel")
    };
    assert_eq!(*gate, DecayGate::BoundedSigmoid { lower: -5.0 });
    assert_eq!((projection.down.shape.as_slice(), projection.up.shape.as_slice()), (&[128, 4096][..], &[8192, 128][..]));
    assert_eq!((bias.name.as_str(), decay.name.as_str()), ("blk.0.ssm_dt.bias", "blk.0.ssm_a"));
    assert_eq!(kda.output_gate.function, GateFunction::Sigmoid);
    assert!(matches!(kda.output_gate.projection, GateProjection::LowRank(_)));

    // Latent attention and its indexer (block 3).
    let Operator::Attention(attention) = &decoder.blocks[3].sublayers[0].op else {
        unreachable!()
    };
    assert_eq!((attention.heads, attention.kv_heads, attention.width), (64, 1, 512));
    let QueryKeys::Absorbed { weight, width } = &attention.query_keys else {
        panic!("each query head is absorbed")
    };
    assert_eq!((weight.shape.as_slice(), *width), (&[64 * 512, 256][..], 256));
    assert_eq!(attention.rotary, Rotary::None);
    assert_eq!(attention.scale, 1.0 / 16.0);
    let OutputProjection::GroupedLowRank { groups, rank, a, b } = &attention.output else {
        panic!("each head's output is projected by its own value block")
    };
    assert_eq!((*groups, *rank), (64, 256));
    assert_eq!((a.shape.as_slice(), b.shape.as_slice()), (&[64 * 256, 512][..], &[4096, 64 * 256][..]));
    let [source] = &attention.history[..] else {
        panic!("one history source")
    };
    let HistoryReads::Selected(Selection::Own(indexer)) = &source.reads else {
        panic!("the layer reads what its own indexer picks")
    };
    assert_eq!((indexer.heads, indexer.width, indexer.top), (32, 128, 512));
    assert_eq!(indexer.query.input, IndexerInput::QueryRank);
    assert_eq!(indexer.scale, 1.0 / 64.0);
    let IndexerKeys::Pooled(keys) = &indexer.keys else {
        panic!("the keys are pooled per 4 tokens")
    };
    assert_eq!(keys.rate, 4);
    let Pool::SoftmaxGated { gate, position: Some(position) } = &keys.pool else {
        panic!("a softmax gate with a per-place term")
    };
    assert_eq!(
        (gate.name.as_str(), position.name.as_str(), position.shape.as_slice()),
        ("blk.3.indexer_compressor_gate.weight", "blk.3.indexer_compressor_ape.weight", &[4, 128][..])
    );
    let PoolNorm::Token { norm, bias } = &keys.norm else {
        panic!("each token's key is LayerNorm-ed with a bias")
    };
    assert_eq!(norm.epsilon, f32_value(1e-6));
    assert_eq!(bias.name, "blk.3.indexer.k_norm.bias");

    // The routed feed-forward (block 3).
    let Operator::RoutedFfn(routed) = &decoder.blocks[3].sublayers[1].op else {
        unreachable!()
    };
    assert_eq!((routed.experts, routed.selected, routed.intermediate), (288, 8, 2048));
    assert_eq!(routed.router.score, ScoreFunction::Sigmoid);
    assert!(matches!(&routed.router.selection, ExpertSelection::TopK { bias: Some(bias) } if bias.name == "blk.3.exp_probs_b.bias"));
    assert_eq!(routed.router.normalization, RouteNormalization::SumPlusEpsilon(1e-20));
    assert_eq!(routed.router.scale, f32_value(2.5));
    assert_eq!(routed.expert_up.gate().unwrap().shape, [288, 2048, 4096]);
    assert_eq!(routed.expert_down.shape, [288, 4096, 2048]);
    assert_eq!(routed.expert_up.limit(), Some(limit));
    let shared = routed.shared.as_ref().unwrap();
    assert_eq!(shared.intermediate, 2048);
    assert!(matches!(&shared.up, FeedForwardUp::Gated { limit: Some(shared), .. } if *shared == limit));

    // The MTP head (block 45 in nextn layout) reads the rows its own indexer
    // picks, with the indexer's weights of its own block.
    let head = definition.head.as_ref().expect("the file stores an MTP head");
    assert_eq!((head.history, head.component), (HeadHistory::Own, HeadComponent::Target));
    let [head] = &head.blocks[..] else { panic!("one MTP block") };
    let [block] = &head.blocks[..] else { panic!("one head layer") };
    let Operator::Attention(attention) = &block.sublayers[0].op else { panic!("the head attends") };
    let [source] = &attention.history[..] else { panic!("one history source") };
    let HistoryReads::Selected(Selection::Own(indexer)) = &source.reads else {
        panic!("the head reads what its own indexer picks")
    };
    assert_eq!((indexer.heads, indexer.width, indexer.top), (32, 128, 512));
    assert_eq!(indexer.query.projection.name, "blk.45.indexer.attn_q_b.weight");
    let IndexerKeys::Pooled(keys) = &indexer.keys else { panic!("the head's keys are pooled") };
    assert_eq!(keys.projection.name, "blk.45.indexer.attn_k.weight");
}

/// `directory` with `name` stored as `encoding`.
fn encode(directory: &mut Directory, name: &str, encoding: Encoding) {
    directory
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("{name} is stored"))
        .encoding = encoding;
}

fn refusal(directory: &Directory) -> String {
    inspect(directory).unwrap_err().to_string()
}

/// A header missing a tensor, adding one, or storing a grid of another
/// shape or type is refused, naming the tensor.
#[test]
fn a_header_missing_adding_or_misgridding_a_tensor_is_refused_naming_it() {
    let published = published();
    for missing in ["blk.3.indexer.k_norm.bias", "blk.0.ssm_conv1d_v.weight", "blk.45.nextn.eh_proj.weight", "blk.44.hc_ffn_base.weight"] {
        let mut directory = published.clone();
        headers::remove_tensor(&mut directory, missing);
        let error = refusal(&directory);
        assert!(error.contains(missing), "{missing}: {error}");
    }
    let mut added = published.clone();
    headers::add_tensor(&mut added, "blk.3.indexer.q_norm.weight", &[128]);
    let error = refusal(&added);
    assert!(error.contains("blk.3.indexer.q_norm.weight"), "{error}");
    // A grid that doesn't divide the stack's matrices into tiles.
    let mut reshaped = published.clone();
    headers::reshape_tensor(&mut reshaped, "blk.3.ffn_gate_exps.scale", &[288, 15, 32]);
    let error = refusal(&reshaped);
    assert!(error.contains("blk.3.ffn_gate_exps.scale"), "{error}");
    // A grid of a type no grid is stored as.
    let mut retyped = published.clone();
    encode(&mut retyped, "blk.0.ffn_gate.scale", Encoding::F16);
    let error = refusal(&retyped);
    assert!(error.contains("blk.0.ffn_gate.scale"), "{error}");
    // A weight of another shape.
    let mut misshapen = published.clone();
    headers::reshape_tensor(&mut misshapen, "blk.3.attn_k_b.weight", &[64, 256, 512]);
    let error = refusal(&misshapen);
    assert!(error.contains("blk.3.attn_k_b.weight"), "{error}");
}

/// A header naming a key the family doesn't read, omitting one it does, or
/// stating a mixer or indexer it doesn't interpret is refused, naming the
/// key.
#[test]
fn a_header_with_unread_missing_or_uninterpreted_metadata_is_refused_naming_it() {
    let published = published();
    let key = |name: &str| format!("{FAMILY_ID}.{name}");
    let mut unread = published.clone();
    headers::set(&mut unread, &key("rope.freq_base"), Value::Scalar(Scalar::Float(1e4)));
    assert!(refusal(&unread).contains(&key("rope.freq_base")));
    let mut missing = published.clone();
    missing.metadata.retain(|entry| entry.name != key("attention.indexer.kpool"));
    assert!(refusal(&missing).contains(&key("attention.indexer.kpool")));
    let Some(Value::Array(kv)) = published.value(&key("attention.head_count_kv")) else {
        panic!("head_count_kv is an I32 array")
    };
    let Ok(Elements::Signed(kv)) = kv.elements() else {
        panic!("head_count_kv is an I32 array")
    };
    let kv = kv.clone();
    let mut two = kv.clone();
    two[3] = 2;
    let mut doubled = published.clone();
    headers::set(&mut doubled, &key("attention.head_count_kv"), array(two, Scalar::Signed));
    assert!(refusal(&doubled).contains(&key("attention.head_count_kv")));
    // The MTP block attends.
    let mut recurrent = kv;
    recurrent[45] = 0;
    let mut drafted = published.clone();
    headers::set(&mut drafted, &key("attention.head_count_kv"), array(recurrent, Scalar::Signed));
    assert!(refusal(&drafted).contains(&key("attention.head_count_kv")));
    let mut shared = published.clone();
    let mut types = vec![true; TRUNK];
    types[7] = false;
    headers::set(&mut shared, &key("attention.indexer.types"), array(types, Scalar::Bool));
    assert!(refusal(&shared).contains(&key("attention.indexer.types")));
}

/// How a synthetic weight is stored: the NVFP4 file's forms
/// (`glm-nvfp4-converter`'s map of NVIDIA's `GLM-5.3-Flash-FP4-Local`).
#[derive(Clone, Copy)]
enum Stored {
    Bf16,
    F32,
    /// F8_E4M3 beside an I8 UE8M0 grid, one per 1×32.
    Mxfp8,
    /// F8_E4M3 beside an F32 grid, one per 128×128 (FP8_PB_WO).
    Fp8Tiles,
    /// Stacked NVFP4 with an F32 `.scale` per expert, and an F32
    /// `.input_scale` per expert when calibrated for FP4 activations.
    Nvfp4 { activations: bool },
}

/// A synthetic header in the NVFP4 file's forms: block 0 KDA with a dense
/// feed-forward, block 1 latent attention with routed experts, block 2 the
/// MTP layer, at small widths.
fn nvfp4_header() -> Directory {
    const H: u64 = 256;
    let mut tensors = Vec::new();
    let mut put = |name: &str, shape: &[u64], stored: Stored| {
        let tensor = |name: String, shape: Vec<u64>, encoding| TensorDescriptor {
            name,
            shape,
            encoding,
            offset: 0,
            nbytes: 0,
        };
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        // A grid's extents: the stack's, then the matrix's over `tile`.
        let grid = |tile: [u64; 2]| {
            let (leading, matrix) = shape.split_at(shape.len() - 2);
            [leading, &[matrix[0] / tile[0], matrix[1] / tile[1]]].concat()
        };
        let (encoding, companions) = match stored {
            Stored::Bf16 => (Encoding::BF16, vec![]),
            Stored::F32 => (Encoding::F32, vec![]),
            Stored::Mxfp8 => (Encoding::F8E4M3, vec![(format!("{stem}.scale"), grid([1, 32]), Encoding::I8)]),
            Stored::Fp8Tiles => (Encoding::F8E4M3, vec![(format!("{stem}.scale"), grid([128, 128]), Encoding::F32)]),
            Stored::Nvfp4 { activations } => (
                Encoding::Nvfp4,
                std::iter::once(format!("{stem}.scale"))
                    .chain(activations.then(|| format!("{stem}.input_scale")))
                    .map(|name| (name, vec![shape[0]], Encoding::F32))
                    .collect(),
            ),
        };
        tensors.push(tensor(name.to_owned(), shape.to_vec(), encoding));
        tensors.extend(companions.into_iter().map(|(name, shape, encoding)| tensor(name, shape, encoding)));
    };
    use Stored::*;
    put("token_embd.weight", &[512, H], Bf16);
    put("output_norm.weight", &[H], Bf16);
    put("output.weight", &[512, H], Mxfp8);
    // Latent attention: MLA in FP8_PB_WO, the MTP layer's in MXFP8 with its
    // v_b MXFP8 and k_b BF16; the trunk's indexer in BF16, the MTP's
    // projections in MXFP8.
    let latent = |put: &mut dyn FnMut(&str, &[u64], Stored), p: &str, linear: Stored, v_b: Stored, index: Stored| {
        put(&format!("{p}attn_q_a.weight"), &[128, H], linear);
        put(&format!("{p}attn_q_a_norm.weight"), &[128], Bf16);
        put(&format!("{p}attn_q_b.weight"), &[2 * 128, 128], linear);
        put(&format!("{p}attn_kv_a_mqa.weight"), &[128, H], linear);
        put(&format!("{p}attn_kv_a_norm.weight"), &[128], Bf16);
        put(&format!("{p}attn_k_b.weight"), &[2, 128, 128], Bf16);
        put(&format!("{p}attn_v_b.weight"), &[2, 128, 128], v_b);
        put(&format!("{p}attn_output.weight"), &[H, 2 * 128], linear);
        put(&format!("{p}indexer.attn_q_b.weight"), &[2 * 128, 128], index);
        put(&format!("{p}indexer.attn_k.weight"), &[128, H], index);
        put(&format!("{p}indexer.k_norm.weight"), &[128], Bf16);
        put(&format!("{p}indexer.k_norm.bias"), &[128], Bf16);
        put(&format!("{p}indexer.proj.weight"), &[2, H], index);
        put(&format!("{p}indexer_compressor_gate.weight"), &[128, H], Bf16);
        put(&format!("{p}indexer_compressor_ape.weight"), &[4, 128], Bf16);
    };
    let routed = |put: &mut dyn FnMut(&str, &[u64], Stored), p: &str, experts: Stored, shared: Stored| {
        put(&format!("{p}ffn_gate_inp.weight"), &[4, H], Bf16);
        put(&format!("{p}exp_probs_b.bias"), &[4], F32);
        put(&format!("{p}ffn_gate_exps.weight"), &[4, 128, H], experts);
        put(&format!("{p}ffn_up_exps.weight"), &[4, 128, H], experts);
        put(&format!("{p}ffn_down_exps.weight"), &[4, H, 128], experts);
        put(&format!("{p}ffn_gate_shexp.weight"), &[128, H], shared);
        put(&format!("{p}ffn_up_shexp.weight"), &[128, H], shared);
        put(&format!("{p}ffn_down_shexp.weight"), &[H, 128], shared);
    };
    for block in 0..3 {
        let p = format!("blk.{block}.");
        put(&format!("{p}attn_norm.weight"), &[H], Bf16);
        put(&format!("{p}ffn_norm.weight"), &[H], Bf16);
        if block < 2 {
            for site in ["attn", "ffn"] {
                put(&format!("{p}hc_{site}_fn.weight"), &[24, 4 * H], Bf16);
                put(&format!("{p}hc_{site}_base.weight"), &[24], F32);
                put(&format!("{p}hc_{site}_scale.weight"), &[3], F32);
            }
        }
        match block {
            0 => {
                // KDA in MXFP8, the dense feed-forward too.
                for part in ["q", "k", "v"] {
                    put(&format!("{p}attn_{part}.weight"), &[H, H], Mxfp8);
                    put(&format!("{p}ssm_conv1d_{part}.weight"), &[1, H, 1, 4], Bf16);
                }
                put(&format!("{p}ssm_f_a.weight"), &[128, H], Bf16);
                put(&format!("{p}ssm_f_b.weight"), &[H, 128], Bf16);
                put(&format!("{p}ssm_g_a.weight"), &[128, H], Bf16);
                put(&format!("{p}ssm_g_b.weight"), &[H, 128], Bf16);
                put(&format!("{p}ssm_beta.weight"), &[2, H], Bf16);
                put(&format!("{p}ssm_a"), &[2], F32);
                put(&format!("{p}ssm_dt.bias"), &[H], F32);
                put(&format!("{p}ssm_norm.weight"), &[128], Bf16);
                put(&format!("{p}attn_output.weight"), &[H, H], Mxfp8);
                put(&format!("{p}ffn_gate.weight"), &[H, H], Mxfp8);
                put(&format!("{p}ffn_up.weight"), &[H, H], Mxfp8);
                put(&format!("{p}ffn_down.weight"), &[H, H], Mxfp8);
            }
            1 => {
                latent(&mut put, &p, Fp8Tiles, Bf16, Bf16);
                routed(&mut put, &p, Nvfp4 { activations: true }, Fp8Tiles);
            }
            _ => {
                put(&format!("{p}nextn.eh_proj.weight"), &[H, 2 * H], Bf16);
                for norm in ["enorm", "hnorm", "shared_head_norm"] {
                    put(&format!("{p}nextn.{norm}.weight"), &[H], Bf16);
                }
                latent(&mut put, &p, Mxfp8, Mxfp8, Mxfp8);
                routed(&mut put, &p, Nvfp4 { activations: false }, Mxfp8);
            }
        }
    }
    let unsigned = |value| Value::Scalar(Scalar::Unsigned(value));
    let float = |value| Value::Scalar(Scalar::Float(value));
    let key = |name: &str, value| gguf::Metadata { name: format!("{FAMILY_ID}.{name}"), value };
    let metadata = vec![
        gguf::Metadata { name: "general.architecture".into(), value: Value::Scalar(Scalar::String(FAMILY_ID.into())) },
        key("block_count", unsigned(3)),
        key("context_length", unsigned(4096)),
        key("embedding_length", unsigned(H)),
        key("vocab_size", unsigned(512)),
        key("feed_forward_length", unsigned(H)),
        key("leading_dense_block_count", unsigned(1)),
        key("nextn_predict_layers", unsigned(1)),
        key("attention.head_count", unsigned(2)),
        key("attention.head_count_kv", array(vec![0, 1, 1], Scalar::Signed)),
        key("attention.layer_norm_rms_epsilon", float(1e-5)),
        key("attention.layer_norm_epsilon", float(1e-6)),
        key("attention.q_lora_rank", unsigned(128)),
        key("attention.kv_lora_rank", unsigned(128)),
        key("attention.key_length", unsigned(128)),
        key("attention.value_length", unsigned(128)),
        key("attention.key_length_mla", unsigned(128)),
        key("attention.value_length_mla", unsigned(128)),
        key("attention.indexer.head_count", unsigned(2)),
        key("attention.indexer.key_length", unsigned(128)),
        key("attention.indexer.top_k", unsigned(8)),
        key("attention.indexer.kpool", unsigned(4)),
        key("attention.indexer.kpool_select_tail", Value::Scalar(Scalar::Bool(true))),
        key("attention.indexer.types", array(vec![true, true], Scalar::Bool)),
        key("rope.dimension_count", unsigned(0)),
        key("ssm.conv_kernel", unsigned(4)),
        key("kda.head_dim", unsigned(128)),
        key("kda.gate_lower_bound", float(-5.0)),
        key("hyper_connection.count", unsigned(4)),
        key("hyper_connection.sinkhorn_iterations", unsigned(20)),
        key("hyper_connection.epsilon", float(1e-6)),
        key("expert_count", unsigned(4)),
        key("expert_used_count", unsigned(2)),
        key("expert_group_count", unsigned(1)),
        key("expert_group_used_count", unsigned(1)),
        key("expert_gating_func", unsigned(2)),
        key("expert_feed_forward_length", unsigned(128)),
        key("expert_shared_count", unsigned(1)),
        key("expert_weights_scale", float(2.5)),
        key("expert_weights_norm", Value::Scalar(Scalar::Bool(true))),
        key("swiglu_clamp_exp", array(vec![10.0; 3], Scalar::Float)),
        key("swiglu_clamp_shexp", array(vec![10.0; 3], Scalar::Float)),
    ];
    Directory {
        version: 3,
        byte_order: ByteOrder::Little,
        alignment: 32,
        data_offset: 0,
        metadata,
        tensors,
    }
}

/// A file without MTP blocks, as upstream writes one (a layer slice, or a
/// conversion without its MTP layer), states no `nextn_predict_layers`: its
/// blocks are all trunk and it has no head.
#[test]
fn a_file_without_nextn_predict_layers_has_no_head() {
    let mut directory = nvfp4_header();
    directory.tensors.retain(|tensor| !tensor.name.starts_with("blk.2."));
    let key = |name: &str| format!("{FAMILY_ID}.{name}");
    directory.metadata.retain(|entry| entry.name != key("nextn_predict_layers"));
    headers::set(&mut directory, &key("block_count"), Value::Scalar(Scalar::Unsigned(2)));
    headers::set(&mut directory, &key("attention.head_count_kv"), array(vec![0, 1], Scalar::Signed));
    for clamp in ["swiglu_clamp_exp", "swiglu_clamp_shexp"] {
        headers::set(&mut directory, &key(clamp), array(vec![10.0; 2], Scalar::Float));
    }
    let definition = inspect(&directory).unwrap();
    assert!(definition.head.is_none());
    assert_eq!(definition.decoder.blocks.len(), 2);
}

/// The NVFP4 file's forms bind in the same family, every tensor but the
/// MTP head's, claimed: W4A4 NVFP4 experts with their per-expert
/// scales and activation scales, and MXFP8 under I8 UE8M0 grids beside F32
/// 128×128 grids in one file. The MTP layer's forms (W4A16 experts, its
/// stacked v_b under a UE8M0 grid, k_b in BF16) are bound with the head.
#[test]
fn the_nvfp4_files_forms_bind_every_tensor() {
    let directory = nvfp4_header();
    let definition = inspect(&directory).unwrap();
    let json = serde_json::to_value(&definition).unwrap();
    let mut bound = BTreeMap::new();
    descriptors(&json, &mut bound);
    let mut transforms = BTreeMap::new();
    collect_transforms(&json, &mut transforms);
    // A W4A4 weight's `.input_scale` is acknowledged by its NVFP4
    // activations, not read.
    let acknowledged = transforms.iter().filter_map(|(name, transforms)| {
        transforms
            .iter()
            .any(|transform| {
                matches!(
                    transform,
                    ImportTransform::ScaleByTensor { activations: ScaledActivations::Nvfp4, .. }
                )
            })
            .then(|| name.replace(".weight", ".input_scale"))
    });
    let claimed = bound
        .iter()
        .flat_map(|(name, reads)| std::iter::once(name).chain(reads))
        .cloned()
        .chain(acknowledged)
        .collect::<BTreeSet<_>>();
    let stored = directory
        .tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    assert_eq!(claimed, stored);

    let grid = |tensor: &str, tile: [u64; 2], codes: ScaleCodes| ImportTransform::ScaleBlocks {
        tensor: tensor.into(),
        tile,
        codes,
    };
    let nvfp4 = |tensor: &str, activations: ScaledActivations| ImportTransform::ScaleByTensor {
        tensor: tensor.into(),
        activations,
    };
    let of = |name: &str| transforms.get(name).unwrap_or_else(|| panic!("{name} is a weight")).clone();
    // KDA's q, k and v in MXFP8, each part under its own UE8M0 grid.
    assert_eq!(
        of("blk.0.attn_q.weight"),
        [
            grid("blk.0.attn_q.scale", [1, 32], ScaleCodes::Ue8m0),
            ImportTransform::AppendRows {
                parts: ["k", "v"]
                    .map(|part| RowPart {
                        tensor: format!("blk.0.attn_{part}.weight"),
                        rows: 256,
                        grid: Some(format!("blk.0.attn_{part}.scale")),
                    })
                    .to_vec(),
            },
        ]
    );
    assert_eq!(of("blk.0.ffn_gate.weight"), [grid("blk.0.ffn_gate.scale", [1, 32], ScaleCodes::Ue8m0)]);
    assert_eq!(of("output.weight"), [grid("output.scale", [1, 32], ScaleCodes::Ue8m0)]);
    // FP8_PB_WO latent attention and shared expert under F32 tiles.
    assert_eq!(of("blk.1.attn_q_a.weight"), [grid("blk.1.attn_q_a.scale", [128, 128], ScaleCodes::F32)]);
    assert_eq!(of("blk.1.ffn_down_shexp.weight"), [grid("blk.1.ffn_down_shexp.scale", [128, 128], ScaleCodes::F32)]);
    // W4A4 trunk experts.
    for part in ["gate", "up", "down"] {
        let trunk = format!("blk.1.ffn_{part}_exps");
        assert_eq!(of(&format!("{trunk}.weight")), [nvfp4(&format!("{trunk}.scale"), ScaledActivations::Nvfp4)]);
    }
    assert!(definition.head.is_some());

    // A grid stored as F16, no grid's type, is refused.
    let mut misgridded = nvfp4_header();
    encode(&mut misgridded, "blk.1.attn_q_a.scale", Encoding::F16);
    assert!(refusal(&misgridded).contains("blk.1.attn_q_a.scale"));
    // A W4A4 expert stack's activation scale of another extent is refused.
    let mut short = nvfp4_header();
    headers::reshape_tensor(&mut short, "blk.1.ffn_up_exps.input_scale", &[3]);
    assert!(refusal(&short).contains("blk.1.ffn_up_exps.input_scale"));
}

/// Every weight descriptor's transforms in `json`, by its name.
fn collect_transforms(json: &Json, out: &mut BTreeMap<String, Vec<ImportTransform>>) {
    match json {
        Json::Object(object) => {
            if let (Some(Json::String(name)), Some(_), Some(transforms)) =
                (object.get("name"), object.get("shape"), object.get("transforms"))
            {
                out.insert(name.clone(), serde_json::from_value(transforms.clone()).unwrap());
                return;
            }
            object.values().for_each(|value| collect_transforms(value, out));
        }
        Json::Array(values) => values.iter().for_each(|value| collect_transforms(value, out)),
        _ => {}
    }
}

/// The published projector (`mmproj-BF16.gguf`, the same revision) is the
/// `glm5v` tower and its gated merger: every stored vision tensor bound, the
/// downsample convolution read as a linear over interleaved cells, both
/// feed-forwards clamped at the metadata's limit, and an image expanding
/// between GLM's markers.
#[test]
fn the_published_projector_is_the_glm5v_tower_and_merger() {
    use magnitude_family_contracts::{
        ActivationDType, CellReduction, VisionActivation, VisionNorm, VisionOperator,
        VisionResize, VisionStem, VisionUp,
    };
    let projector = headers::projector("glm-5.3-flash:gguf:fp8");
    let definition = Glm5NextFamily
        .inspect(&published(), Some(&projector), identity())
        .unwrap();
    let vision = definition.vision.as_ref().expect("a vision component");
    vision.validate().unwrap();
    assert_eq!(vision.activation_dtype, ActivationDType::F16);
    assert_eq!((vision.hidden, vision.output_hidden, vision.blocks.len()), (1024, 4096, 24));
    let p = &vision.preprocessing;
    assert_eq!(
        p.resize,
        VisionResize::PaddedCanvas { min_pixels: 12_544, max_pixels: 6_272_000 }
    );
    assert_eq!((p.channels, p.patch, p.merge), (3, 14, 2));
    assert_eq!(vision.max_cells().unwrap(), 8000);
    assert_eq!(vision.patch_row_width().unwrap(), 1176);
    let VisionStem::Patch { frames, bias, positions, norm, registers } = &vision.stem else {
        panic!("a patch convolution stem");
    };
    assert_eq!(frames.len(), 2);
    assert!(bias.is_some() && positions.is_none() && norm.is_none() && registers.is_none());
    for block in &vision.blocks {
        let VisionOperator::Attention(attention) = &block.operator else {
            panic!("the tower attends");
        };
        assert_eq!((attention.heads, attention.width), (16, 64));
        assert!(attention.query_norm.is_some() && attention.key_norm.is_some());
        assert_eq!(attention.rotary_base, Some(10_000.0));
        let VisionUp::Gated { limit, .. } = &block.feedforward.up else {
            panic!("the tower's feed-forward is gated");
        };
        assert_eq!(*limit, Some(10.0));
        assert_eq!(block.feedforward.activation, VisionActivation::Silu);
        assert!(matches!(block.feedforward_norm, VisionNorm::Rms { .. }));
    }
    let merger = &vision.merger;
    assert_eq!(merger.reduction, CellReduction::Interleave);
    assert!(matches!(merger.norm, Some(VisionNorm::Rms { .. })));
    let [downsample, projection] = merger.stages.as_slice() else {
        panic!("a downsample and a projection stage");
    };
    assert_eq!(downsample.linear.weight.shape, [4096, 4096]);
    assert_eq!(downsample.linear.weight.transforms, [ImportTransform::MergeTrailingAxes]);
    assert!(downsample.linear.bias.is_some() && downsample.norm.is_none() && downsample.activation.is_none());
    // PyTorch's LayerNorm default, which the released merger keeps: not the
    // RMS norms' epsilon the file states.
    assert!(matches!(projection.norm, Some(VisionNorm::Layer { epsilon: 1e-5, .. })));
    assert_eq!(projection.activation, Some(VisionActivation::GeluErf));
    let feed_forward = merger.feed_forward.as_ref().expect("the merger's gated feed-forward");
    assert_eq!(feed_forward.intermediate(), 10_240);
    assert!(matches!(feed_forward.up, VisionUp::Gated { limit: Some(limit), .. } if limit == 10.0));
    assert_eq!(
        Glm5NextFamily.media_placeholder(&definition),
        Some("<|begin_of_image|><|image|><|end_of_image|>")
    );
}
