//! The separate draft (DFlash, DSpark, DFlash2, LiLiCorr; plan §3.8) forms this
//! executor runs, and the roles of a draft's weights.
//!
//! The draft's fusion (its projection of the concatenated target taps and the
//! norm the context K/V projections read) lives with the target: every target
//! step publishes its fused taps as the target's features. Every other draft
//! weight is the drafter component's, imported when drafting first runs.

use super::{FeedForward, Mixer, PairedBlock, paired_block, role, sublayer_weights};
use crate::error::PlanError;
use magnitude_family_contracts::{
    AffinePart, AttentionGate, Block, CorrectorSite, CorrectorWeight, DraftDefinition,
    DraftEmbedding, DraftMethod, HeadHistory, KeyValue, ModelDefinition, OutputForm,
    SublayerIndex, TapPoint, ValueSource, WeightDescriptor, WeightKind, WeightRole, WeightScope,
};

/// The draft weights a target step reads: the fusion projection and the
/// fusion norm.
pub(crate) fn fusion_weights(draft: &DraftDefinition) -> [(WeightRole, &WeightDescriptor); 2] {
    [
        (
            role(WeightScope::Draft, WeightKind::DraftFusion),
            &draft.fusion,
        ),
        (
            role(WeightScope::Draft, WeightKind::DraftFusionNorm),
            &draft.fusion_norm.weight,
        ),
    ]
}

/// Every drafter weight of a draft (all but its fusion), in model order.
pub(crate) fn draft_weights(
    draft: &DraftDefinition,
) -> Result<Vec<(WeightRole, &WeightDescriptor)>, PlanError> {
    let mut weights = Vec::new();
    if let DraftEmbedding::Own(table) = &draft.embedding {
        weights.push((role(WeightScope::Draft, WeightKind::Embedding), table));
    }
    if let Some(output) = &draft.output {
        weights.push((role(WeightScope::Draft, WeightKind::Output), output));
    }
    for (block, value) in draft.blocks.iter().enumerate() {
        let block = u32::try_from(block).map_err(|_| PlanError::Arithmetic("draft block index"))?;
        for (sublayer, value) in value.sublayers.iter().enumerate() {
            let index = SublayerIndex {
                block,
                sublayer: u32::try_from(sublayer)
                    .map_err(|_| PlanError::Arithmetic("draft sublayer index"))?,
            };
            sublayer_weights(
                value,
                WeightScope::DraftSublayer(index),
                &mut weights,
                |_| unreachable!("admitted draft layers have no parallel branches"),
            );
        }
    }
    weights.push((
        role(WeightScope::Draft, WeightKind::OutputNorm),
        &draft.output_norm.weight,
    ));
    // DSpark's confidence head is never evaluated (the service threshold is
    // zero), so no entry reads its weights and they are not imported.
    if let DraftMethod::DSpark { markov, .. } = &draft.method {
        weights.push((
            role(WeightScope::Draft, WeightKind::MarkovEmbedding),
            &markov.embedding,
        ));
        weights.push((
            role(WeightScope::Draft, WeightKind::MarkovProjection),
            &markov.projection,
        ));
    }
    if let Some(convolutions) = &draft.convolutions {
        for (block, layer) in convolutions.layers.iter().enumerate() {
            let block =
                u32::try_from(block).map_err(|_| PlanError::Arithmetic("draft block index"))?;
            for (sublayer, convolution) in [(0, &layer.attention), (1, &layer.feed_forward)] {
                let scope = WeightScope::DraftSublayer(SublayerIndex { block, sublayer });
                weights.push((role(scope, WeightKind::ConvolutionBase), &convolution.base));
                weights.push((
                    role(scope, WeightKind::ConvolutionProjection),
                    &convolution.projection,
                ));
            }
        }
    }
    if let DraftMethod::LiLiCorr { corrector } = &draft.method {
        weights.extend(corrector.weights());
    }
    if let DraftMethod::DFlash2 { selector } = &draft.method {
        weights.push((
            role(WeightScope::Draft, WeightKind::SelectorHidden),
            &selector.hidden,
        ));
        weights.push((
            role(WeightScope::Draft, WeightKind::SelectorPredecessor),
            &selector.predecessor,
        ));
        weights.push((
            role(WeightScope::Draft, WeightKind::SelectorSuccessor),
            &selector.successor,
        ));
    }
    Ok(weights)
}

/// The blocks of a definition's drafter that keep their own history: its
/// draft head's, or its separate draft's layers (an executed definition
/// carries at most one drafter). A head reading target history keeps none.
pub(crate) fn drafter_blocks(definition: &ModelDefinition) -> Vec<&Block> {
    match (&definition.head, &definition.draft) {
        (Some(head), _) => match head.history {
            HeadHistory::Own => head.blocks.iter().flat_map(|block| &block.blocks).collect(),
            HeadHistory::Target => Vec::new(),
        },
        (None, Some(draft)) => draft.blocks.iter().collect(),
        (None, None) => Vec::new(),
    }
}

/// The draft layer `block`'s attention and dense feed-forward.
pub(crate) fn draft_block(
    draft: &DraftDefinition,
    block: usize,
) -> Result<PairedBlock<'_>, PlanError> {
    let block = draft
        .blocks
        .get(block)
        .ok_or(PlanError::Topology("draft block is absent"))?;
    paired_block(block)
}

/// Admit a definition's draft: pre-normed attention and dense feed-forward
/// layers with residual outputs and one normalization epsilon, and taps at
/// target block entries or at the exit.
pub(crate) fn admit(definition: &ModelDefinition) -> Result<(), PlanError> {
    let Some(draft) = &definition.draft else {
        return Ok(());
    };
    // A target block graph taps the residual entering it, entering its
    // feed-forward, or (the last block, for the exit) leaving it.
    for tap in &draft.taps {
        if let TapPoint::Sublayer(SublayerIndex { block, sublayer }) = tap {
            let paired = definition
                .decoder
                .blocks
                .get(*block as usize)
                .map(paired_block)
                .transpose()?
                .ok_or(PlanError::Topology("draft tap beyond the target"))?;
            match sublayer {
                0 => {}
                1 if paired.feed_forward.is_some() => {}
                _ => return Err(PlanError::Unsupported("draft tap inside a target sublayer")),
            }
        }
    }
    let epsilon = draft.output_norm.epsilon;
    if draft.fusion_norm.epsilon != epsilon {
        return Err(PlanError::Unsupported(
            "normalization epsilons differing within a draft",
        ));
    }
    for index in 0..draft.blocks.len() {
        let paired = draft_block(draft, index)?;
        let (Mixer::Attention(attention), Some(feed_forward)) = (paired.mixer, paired.feed_forward)
        else {
            return Err(PlanError::Unsupported(
                "draft layer other than attention and feed-forward",
            ));
        };
        if !matches!(feed_forward.op, FeedForward::Dense(_))
            || [paired.mixer_output, feed_forward.output]
                .iter()
                .any(|output| **output != OutputForm::Residual)
        {
            return Err(PlanError::Unsupported("draft layer form"));
        }
        // A convolved backbone runs its layers unfused: plain query, key and
        // value projections of the convolved rows.
        if draft.convolutions.is_some()
            && (!matches!(attention.gate, AttentionGate::None)
                || !matches!(
                    attention.key_value,
                    KeyValue::Owned {
                        value: ValueSource::Projected(_),
                        ..
                    }
                ))
        {
            return Err(PlanError::Unsupported(
                "a convolved draft layer other than ungated attention with projected values",
            ));
        }
        if [
            paired.epsilon(),
            super::attention::head_norm_epsilon(attention, epsilon)?,
        ]
        .into_iter()
        .chain(feed_forward.epsilons())
        .any(|value| value != epsilon)
        {
            return Err(PlanError::Unsupported(
                "normalization epsilons differing within a draft",
            ));
        }
    }
    if let DraftMethod::LiLiCorr { corrector } = &draft.method {
        // The corrector's norms read the draft's one epsilon.
        if corrector
            .layers
            .iter()
            .flat_map(|layer| [&layer.attention_norm, &layer.mlp_norm])
            .chain([&corrector.output_norm, &corrector.anchor_norm])
            .any(|norm| norm.epsilon != epsilon)
        {
            return Err(PlanError::Unsupported(
                "normalization epsilons differing within a draft",
            ));
        }
        // Its nodes attend one block through its corrector attention.
        if corrector.width % corrector.heads != 0 {
            return Err(PlanError::Topology("corrector heads do not divide its width"));
        }
    }
    Ok(())
}

/// Whether the corrector entries read `weight` as F32 (`corrector.seismic`):
/// biases, the feature norm and projection, the slot and rank embeddings and
/// the attention bias. Its other matrices are projections and its norms
/// norm weights, resident like any draft's.
pub(crate) fn corrector_reads_f32(weight: CorrectorWeight) -> bool {
    match weight {
        CorrectorWeight::Affine { site, part } => {
            part == AffinePart::Bias || site == CorrectorSite::FeatureUp
        }
        CorrectorWeight::FeatureNorm(_)
        | CorrectorWeight::SlotEmbedding
        | CorrectorWeight::RankEmbedding
        | CorrectorWeight::AttentionBias => true,
        CorrectorWeight::OutputNorm
        | CorrectorWeight::AnchorNorm
        | CorrectorWeight::AttentionNorm
        | CorrectorWeight::MlpNorm => false,
    }
}
