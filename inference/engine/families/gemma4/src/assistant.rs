//! Gemma 4 assistant drafters (GGUF architecture `gemma4-assistant`): a
//! separate draft component that is a draft head of its Gemma 4 target.
//!
//! The assistant has its own width and Gemma-form blocks, but no keys or
//! values: every attention layer reads the target's history, a window layer
//! the target's second-to-last layer and a full layer its last. Its input is
//! the target's token embedding, scaled as the target scales it, beside the
//! target's normalized output row, projected into its width
//! (`nextn.pre_projection`). Its normalized output reads out through its own
//! full-vocabulary token table, and `nextn.post_projection` returns it to the
//! target's width as the hidden row of the next chained step. It has no soft
//! cap.

use crate::{
    Error, LayerKind, Shape, attention, dense, full_geometry, invalid, layer_kinds, product,
    window_geometry,
};
use magnitude_artifacts::gguf::{Directory, Value};
use magnitude_family_common::{Metadata, Tensors};
use magnitude_family_contracts::{
    Block, ExitNorm, Head, HeadBlock, HeadChain, HeadComponent, HeadHistory, HeadInput,
    HeadReadout, InputNorm, MediaRowAttention, ModelDefinition, OutputForm, ProposalReadoutPolicy,
    Sublayer, SublayerIndex, TokenId, VocabularyProjection,
};

/// The GGUF architecture of an assistant drafter.
pub const ASSISTANT_ARCHITECTURE: &str = "gemma4-assistant";

/// Every `gemma4-assistant.` metadata key this family interprets.
const METADATA_KEYS: &[&str] = &[
    "block_count",
    "context_length",
    "embedding_length",
    "embedding_length_out",
    "embedding_length_per_layer_input",
    "feed_forward_length",
    "nextn_predict_layers",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.key_length_swa",
    "attention.value_length_swa",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "attention.sliding_window_pattern",
    "attention.shared_kv_layers",
    "rope.freq_base",
    "rope.freq_base_swa",
    "rope.dimension_count",
    "rope.dimension_count_swa",
    "output.cluster_count",
    "output.cluster_top_k",
    "output.cluster_token_order",
];

/// The target layer an assistant layer of `kind` reads, counted back from
/// the target's last layer.
fn source_offset(kind: LayerKind) -> u64 {
    match kind {
        LayerKind::Window => 2,
        LayerKind::Full => 1,
    }
}

/// Interpret an assistant drafter against its Gemma 4 `target`: the draft
/// head it is. A draft that is not an assistant of this target is an error.
pub fn inspect_assistant(draft: &Directory, target: &ModelDefinition) -> Result<Head, Error> {
    match draft.value("general.architecture").and_then(Value::string) {
        Some(ASSISTANT_ARCHITECTURE) => {}
        architecture => {
            return Err(invalid(format!(
                "unsupported Gemma assistant GGUF architecture {architecture:?}"
            )));
        }
    }
    if target.family.0 != crate::ARCHITECTURE {
        return Err(invalid(format!(
            "a Gemma assistant drafts for a Gemma 4 target, not {:?}",
            target.family.0
        )));
    }
    if target.head.is_some() || target.draft.is_some() {
        return Err(invalid(
            "a Gemma assistant drafts for a target without a drafter",
        ));
    }
    let decoder = &target.decoder;
    let m = Metadata::new(draft, ASSISTANT_ARCHITECTURE, METADATA_KEYS)?;
    let layers = m.integer("block_count")?;
    if layers > draft.tensors.len() as u64 {
        return Err(invalid(
            "Gemma assistant layer count exceeds its stored weights",
        ));
    }
    if m.integer("nextn_predict_layers")? != layers
        || m.count("attention.shared_kv_layers")? != layers
        || m.count("embedding_length_per_layer_input")? != 0
    {
        return Err(invalid(
            "every Gemma assistant layer is a prediction layer reading its target's history",
        ));
    }
    let target_width = m.integer("embedding_length_out")?;
    if target_width != decoder.hidden {
        return Err(invalid(format!(
            "Gemma assistant drafts for a target of width {target_width}, not {}",
            decoder.hidden
        )));
    }
    // The assistant's declared context is the positions it was trained on;
    // it stores no rows, so the target's context bounds a request.
    m.integer("context_length")?;
    let shape = Shape {
        hidden: m.integer("embedding_length")?,
        heads: m.integer("attention.head_count")?,
        epsilon: m.positive_number("attention.layer_norm_rms_epsilon")?,
    };
    let width = shape.hidden;
    let kv_heads = m.per_layer("attention.head_count_kv", layers)?;
    let feed_forward = m.per_layer("feed_forward_length", layers)?;
    let kinds = layer_kinds(&m, layers)?;
    // A draft row is one text row at its anchor's position; how media rows
    // attend one another does not bear on it.
    let window = window_geometry(&m, MediaRowAttention::Causal)?;

    let mut b = Tensors::new(draft);
    let table = b.bind("token_embd.weight", &[decoder.vocabulary, width])?;
    let policy = match m.optional_integer("output.cluster_count")? {
        None => {
            if m.optional_integer("output.cluster_top_k")?.is_some()
                || draft
                    .value("gemma4-assistant.output.cluster_token_order")
                    .is_some()
            {
                return Err(invalid("partial assistant cluster readout policy"));
            }
            ProposalReadoutPolicy::Full
        }
        Some(clusters) => ProposalReadoutPolicy::Clustered {
            centroids: b.bind("output_centroids.weight", &[clusters, width])?,
            selected_clusters: m.integer("output.cluster_top_k")?,
            token_order: m
                .counts("output.cluster_token_order")?
                .into_iter()
                .map(|value| {
                    u32::try_from(value)
                        .map(TokenId)
                        .map_err(|_| invalid("assistant cluster token exceeds canonical ID range"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        },
    };
    let readout = HeadReadout {
        projection: VocabularyProjection::Own { table },
        policy,
    };
    let output_norm = b.bind("output_norm.weight", &[width])?;
    let full = full_geometry(&m, &mut b)?;
    let joined = product(&[2, decoder.hidden])?;
    let pre = b.bind("nextn.pre_projection.weight", &[width, joined])?;
    let post = b.bind("nextn.post_projection.weight", &[decoder.hidden, width])?;

    let target_layers = decoder.blocks.len() as u64;
    let mut blocks = Vec::with_capacity(layers as usize);
    for layer in 0..layers {
        let index = layer as usize;
        let p = format!("blk.{layer}.");
        let kind = kinds[index];
        let geometry = match kind {
            LayerKind::Window => &window,
            LayerKind::Full => &full,
        };
        let source = target_layers
            .checked_sub(source_offset(kind))
            .ok_or_else(|| invalid("Gemma assistant target has too few layers"))?;
        // The history the assistant was trained to read is the one its
        // source layer keeps: a window layer the same window, a full layer
        // every row.
        let source_domain = u32::try_from(source)
            .ok()
            .and_then(|block| decoder.history_domain(SublayerIndex { block, sublayer: 0 }));
        if source_domain != Some(geometry.domain) {
            return Err(invalid(format!(
                "Gemma assistant layer {layer} reads target layer {source}, whose history is {source_domain:?}, as {:?}",
                geometry.domain
            )));
        }
        let attention = attention(
            &mut b,
            &shape,
            &p,
            geometry,
            kv_heads[index],
            Some(source as usize),
        )?;
        blocks.push(Block {
            sublayers: vec![
                attention,
                Sublayer {
                    input: InputNorm::Rms(
                        shape.rms(b.bind(&format!("{p}ffn_norm.weight"), &[width])?),
                    ),
                    op: dense(&mut b, &p, width, feed_forward[index])?,
                    output: OutputForm::ScaledPostNorm {
                        norm: shape.rms(b.bind(&format!("{p}post_ffw_norm.weight"), &[width])?),
                        layer_scale: b.bind(&format!("{p}layer_output_scale.weight"), &[1])?,
                    },
                },
            ],
        });
    }
    b.finish()?;
    let head = Head {
        blocks: vec![HeadBlock {
            residual: magnitude_family_contracts::ResidualForm::Single,
            input: HeadInput::Projected {
                pre,
                embedding_scale: decoder.entry.scale,
            },
            blocks,
            output_norm: ExitNorm::Rms(shape.rms(output_norm)),
            readout,
            chain: HeadChain::Projected { post },
        }],
        history: HeadHistory::Target,
        component: HeadComponent::Draft,
    };
    head.validate(decoder, target.inputs.coordinate_axes)
        .map_err(|error| invalid(error.to_string()))?;
    Ok(head)
}
