//! Separately packaged prediction blocks sharing the target's token tables.

use super::{Error, FAMILY_ID, hyper_site, invalid, recognize};
use magnitude_artifacts::gguf::{Directory, Scalar, Value};
use magnitude_family_common::Tensors;
use magnitude_family_contracts::{
    Block, ExitNorm, Head, HeadBlock, HeadChain, HeadComponent, HeadHistory, HeadInput,
    HeadReadout, HyperConnectionMixing, InputNorm, ModelDefinition, Operator, OutputForm,
    ResidualForm, Sublayer, VocabularyProjection,
};
use magnitude_family_qwen::{Geometry, Metadata, product};

pub fn inspect_head(draft: &Directory, target: &ModelDefinition) -> Result<Head, Error> {
    recognize(draft)?;
    if target.family.0 != FAMILY_ID || target.head.is_some() || target.draft.is_some() {
        return Err(invalid(
            "Qwen4 prediction head requires a Qwen4 target without a drafter",
        ));
    }
    let decoder = &target.decoder;
    let m = Metadata::new(draft, FAMILY_ID);
    if m.integer("nextn_predict_layers")? != 1
        || m.integer("block_count")? != decoder.blocks.len() as u64 + 1
        || !matches!(
            draft.value("qwen4exp.nextn_shared_target_tensors"),
            Some(Value::Scalar(Scalar::Bool(true)))
        )
    {
        return Err(invalid(
            "prediction package must contain one head sharing its target's token tables",
        ));
    }
    let ResidualForm::HyperConnections(form) = &decoder.residual else {
        return Err(invalid("prediction target has no residual streams"));
    };
    let HyperConnectionMixing::LowRankGated { rank } = form.mixing else {
        return Err(invalid(
            "prediction target does not use low-rank gated mixing",
        ));
    };
    let geometry = Geometry::read(&m, true)?;
    if geometry.hidden != decoder.hidden
        || m.integer("hyper_connection.count")? != form.streams
        || m.integer("hyper_connection.low_rank")? != rank
        || m.integer("context_length")? < decoder.context_limit
    {
        return Err(invalid("prediction head geometry differs from its target"));
    }
    let hidden = decoder.hidden;
    let wide = product(&[hidden, form.streams])?;
    let prefix = format!("blk.{}.", decoder.blocks.len());
    let mut tensors = Tensors::new(draft);
    let mut weight =
        |name: &str, shape: &[u64]| Ok(tensors.bind(&format!("{prefix}{name}"), shape)?);
    let input = HeadInput::Combine {
        embedding_norm: geometry.rms(weight("nextn.enorm.weight", &[hidden])?),
        hidden_norm: geometry.rms(weight("nextn.hnorm.weight", &[wide])?),
        combine: weight("nextn.eh_proj.weight", &[hidden, product(&[2, hidden])?])?,
    };
    let attention = geometry.attention(&mut weight)?;
    let target_attention = decoder
        .sublayers()
        .find_map(|(_, sublayer)| match &sublayer.op {
            Operator::Attention(attention) => Some(attention),
            _ => None,
        })
        .ok_or_else(|| invalid("prediction target has no attention"))?;
    if attention.heads != target_attention.heads
        || attention.kv_heads != target_attention.kv_heads
        || attention.width != target_attention.width
        || attention.rotary != target_attention.rotary
        || attention.scale != target_attention.scale
    {
        return Err(invalid(
            "prediction attention geometry differs from its target",
        ));
    }
    let feed_forward = geometry.feed_forward(&mut weight, geometry.intermediate)?;
    let target_feed_forward = decoder
        .sublayers()
        .find_map(|(_, sublayer)| match &sublayer.op {
            Operator::RoutedFfn(ffn) => Some(ffn),
            _ => None,
        })
        .ok_or_else(|| invalid("prediction target has no routed feed-forward"))?;
    let Operator::RoutedFfn(ffn) = &feed_forward else {
        return Err(invalid("prediction head must use routed feed-forward"));
    };
    if ffn.experts != target_feed_forward.experts
        || ffn.selected != target_feed_forward.selected
        || ffn.intermediate != target_feed_forward.intermediate
    {
        return Err(invalid(
            "prediction expert geometry differs from its target",
        ));
    }

    let block = Block {
        sublayers: vec![
            Sublayer {
                input: InputNorm::HyperConnectionMix(hyper_site(
                    &geometry,
                    &mut weight,
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
                    &mut weight,
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
        &mut weight,
        "nextn.hc_head",
        form.streams,
        rank,
        false,
    )?);
    // Published packages retain the indexer weights even though the prediction
    // block declares compression ratio zero and uses ordinary causal attention.
    let ratios = m.integers("attention.compress_ratios")?;
    if ratios.len() != decoder.blocks.len() + 1 || ratios.last() != Some(&0) {
        return Err(invalid("prediction head must declare unselected attention"));
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
    tensors.finish()?;
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
        component: HeadComponent::Draft,
    };
    head.validate(decoder, target.inputs.coordinate_axes)
        .map_err(|error| invalid(error.to_string()))?;
    Ok(head)
}
