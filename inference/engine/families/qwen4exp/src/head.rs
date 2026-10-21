//! Prediction blocks sharing the target's token tables: packaged
//! separately, or stored after the target's last block in its own file.

use super::{Error, FAMILY_ID, hyper_site, recognize};
use magnitude_artifacts::gguf::{Directory, Scalar, Value};
use magnitude_family_common::Tensors;
use magnitude_family_contracts::{
    Block, ExitNorm, Head, HeadBlock, HeadChain, HeadComponent, HeadHistory, HeadInput,
    HeadReadout, HyperConnectionMixing, InputNorm, ModelDefinition, Operator, OutputForm,
    ResidualForm, Sublayer, VocabularyProjection,
};
use magnitude_family_qwen::{
    GEOMETRY_KEYS, Geometry, MIXER_KEYS, Metadata, ROUTED_KEYS, Weight, product,
};

/// The keys a prediction package interprets beside the shared Qwen
/// geometry, mixer and routed keys. The package repeats its target's
/// attention layout (`full_attention_interval`, the indexer's `top_k`),
/// which its own unselected attention does not read.
const HEAD_KEYS: &[&str] = &[
    "block_count",
    "context_length",
    "nextn_predict_layers",
    "nextn_shared_target_tensors",
    "attention.compress_ratios",
    "attention.indexer.head_count",
    "attention.indexer.key_length",
    "attention.indexer.top_k",
    "hyper_connection.count",
    "hyper_connection.low_rank",
];

/// Bind a separate prediction package to its Qwen4 `target`: the draft head
/// it is. A package that is not a head of this target is a mismatch.
pub fn inspect_head(draft: &Directory, target: &ModelDefinition) -> Result<Head, Error> {
    recognize(draft)?;
    if target.family.0 != FAMILY_ID || target.head.is_some() || target.draft.is_some() {
        return Err(Error::mismatch(
            "Qwen4 prediction head requires a Qwen4 target without a drafter",
        ));
    }
    let m = Metadata::new(
        draft,
        FAMILY_ID,
        &[HEAD_KEYS, GEOMETRY_KEYS, MIXER_KEYS, ROUTED_KEYS].concat(),
    )?;
    if m.integer("nextn_predict_layers")? != 1
        || m.integer("block_count")? != target.decoder.blocks.len() as u64 + 1
        || !matches!(
            draft.value("qwen4exp.nextn_shared_target_tensors"),
            Some(Value::Scalar(Scalar::Bool(true)))
        )
    {
        return Err(Error::mismatch(
            "prediction package must contain one head sharing its target's token tables",
        ));
    }
    let prefix = format!("blk.{}.", target.decoder.blocks.len());
    let mut tensors = Tensors::new(draft);
    let head = prediction_head(&m, target, HeadComponent::Draft, &mut |name, shape| {
        Ok(tensors.bind_scaled(&format!("{prefix}{name}"), shape)?)
    })?;
    tensors.finish()?;
    Ok(head)
}

/// The prediction head after `target`'s last block, read from `m` with
/// `weight` binding its block's roles: a separate package's
/// (`HeadComponent::Draft`), whose disagreement with the target is a
/// mismatch, or the target file's own (`HeadComponent::Target`), whose
/// disagreement is a header that contradicts itself.
pub(crate) fn prediction_head(
    m: &Metadata,
    target: &ModelDefinition,
    component: HeadComponent,
    weight: &mut Weight,
) -> Result<Head, Error> {
    let refuse = |message: &str| match component {
        HeadComponent::Draft => Error::mismatch(message),
        HeadComponent::Target => Error::invalid(message),
    };
    let decoder = &target.decoder;
    let ResidualForm::HyperConnections(form) = &decoder.residual else {
        return Err(refuse("prediction target has no residual streams"));
    };
    let HyperConnectionMixing::LowRankGated { rank } = form.mixing else {
        return Err(refuse(
            "prediction target does not use low-rank gated mixing",
        ));
    };
    let geometry = Geometry::read(m, true)?;
    if geometry.hidden != decoder.hidden
        || m.integer("hyper_connection.count")? != form.streams
        || m.integer("hyper_connection.low_rank")? != rank
        || m.integer("context_length")? < decoder.context_limit
    {
        return Err(refuse("prediction head geometry differs from its target"));
    }
    let hidden = decoder.hidden;
    let wide = product(&[hidden, form.streams])?;
    let input = HeadInput::Combine {
        embedding_norm: geometry.rms(weight("nextn.enorm.weight", &[hidden])?),
        hidden_norm: geometry.rms(weight("nextn.hnorm.weight", &[wide])?),
        combine: weight("nextn.eh_proj.weight", &[hidden, product(&[2, hidden])?])?,
    };
    let attention = geometry.attention(weight)?;
    let target_attention = decoder
        .sublayers()
        .find_map(|(_, sublayer)| match &sublayer.op {
            Operator::Attention(attention) => Some(attention),
            _ => None,
        })
        .ok_or_else(|| refuse("prediction target has no attention"))?;
    if attention.heads != target_attention.heads
        || attention.kv_heads != target_attention.kv_heads
        || attention.width != target_attention.width
        || attention.rotary != target_attention.rotary
        || attention.scale != target_attention.scale
    {
        return Err(refuse(
            "prediction attention geometry differs from its target",
        ));
    }
    let feed_forward = geometry.feed_forward(weight, geometry.intermediate)?;
    let target_feed_forward = decoder
        .sublayers()
        .find_map(|(_, sublayer)| match &sublayer.op {
            Operator::RoutedFfn(ffn) => Some(ffn),
            _ => None,
        })
        .ok_or_else(|| refuse("prediction target has no routed feed-forward"))?;
    let Operator::RoutedFfn(ffn) = &feed_forward else {
        return Err(refuse("prediction head must use routed feed-forward"));
    };
    if ffn.experts != target_feed_forward.experts
        || ffn.selected != target_feed_forward.selected
        || ffn.intermediate != target_feed_forward.intermediate
    {
        return Err(refuse("prediction expert geometry differs from its target"));
    }

    let block = Block {
        sublayers: vec![
            Sublayer {
                input: InputNorm::HyperConnectionMix(hyper_site(
                    &geometry,
                    weight,
                    "hc_attn",
                    form.streams,
                    rank,
                    true,
                )?),
                op: Operator::Attention(Box::new(attention)),
                output: OutputForm::Scatter,
            },
            Sublayer {
                input: InputNorm::HyperConnectionMix(hyper_site(
                    &geometry,
                    weight,
                    "hc_ffn",
                    form.streams,
                    rank,
                    true,
                )?),
                op: feed_forward,
                output: OutputForm::Scatter,
            },
        ],
    };
    let output_norm = ExitNorm::HyperConnectionHead(hyper_site(
        &geometry,
        weight,
        "nextn.hc_head",
        form.streams,
        rank,
        false,
    )?);
    // Published packages retain the indexer weights even though the prediction
    // block declares compression ratio zero and uses ordinary causal attention.
    let ratios = m.integers("attention.compress_ratios")?;
    if ratios.len() != decoder.blocks.len() + 1 || ratios.last() != Some(&0) {
        return Err(refuse("prediction head must declare unselected attention"));
    }
    let indexer_width = m.integer("attention.indexer.key_length")?;
    let indexer_heads = m.integer("attention.indexer.head_count")?;
    for (name, shape) in [
        ("indexer.k_norm.weight", vec![indexer_width]),
        ("indexer.k_proj.weight", vec![indexer_width, hidden]),
        ("indexer.q_norm.weight", vec![indexer_width]),
        (
            "indexer.q_proj.weight",
            vec![product(&[indexer_heads, indexer_width])?, hidden],
        ),
    ] {
        weight(name, &shape)?;
    }
    let head = Head {
        blocks: vec![HeadBlock {
            residual: decoder.residual.clone(),
            input,
            blocks: vec![block],
            output_norm,
            readout: HeadReadout::full(VocabularyProjection::Target),
            chain: HeadChain::Residual,
        }],
        history: HeadHistory::Own,
        component,
    };
    head.validate(decoder, target.inputs.coordinate_axes)
        .map_err(|error| refuse(&error.to_string()))?;
    Ok(head)
}
