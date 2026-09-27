//! The separate draft (DFlash, DSpark; plan §3.8) forms this executor runs,
//! and the roles of a draft's weights.
//!
//! The draft's fusion (its projection of the concatenated target taps and the
//! norm the context K/V projections read) lives with the target: every target
//! step publishes its fused taps as the target's features. Every other draft
//! weight is the drafter component's, imported when drafting first runs.

use super::{paired_block, role, sublayer_weights, FeedForward, Mixer, PairedBlock};
use crate::error::PlanError;
use magnitude_family_contracts::{
    Block, DraftDefinition, DraftEmbedding, DraftMethod, ModelDefinition, OutputForm,
    SublayerIndex, TapPoint, WeightDescriptor, WeightKind, WeightRole, WeightScope,
};

/// The draft weights a target step reads: the fusion projection and the
/// fusion norm.
pub(crate) fn fusion_weights(draft: &DraftDefinition) -> [(WeightRole, &WeightDescriptor); 2] {
    [
        (role(WeightScope::Draft, WeightKind::DraftFusion), &draft.fusion),
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
    for (block, value) in draft.blocks.iter().enumerate() {
        let block = u32::try_from(block).map_err(|_| PlanError::Arithmetic("draft block index"))?;
        for (sublayer, value) in value.sublayers.iter().enumerate() {
            let index = SublayerIndex {
                block,
                sublayer: u32::try_from(sublayer)
                    .map_err(|_| PlanError::Arithmetic("draft sublayer index"))?,
            };
            sublayer_weights(value, WeightScope::DraftSublayer(index), &mut weights, |_| {
                unreachable!("admitted draft layers have no parallel branches")
            });
        }
    }
    weights.push((
        role(WeightScope::Draft, WeightKind::OutputNorm),
        &draft.output_norm.weight,
    ));
    if let DraftMethod::DSpark { markov, confidence } = &draft.method {
        weights.push((role(WeightScope::Draft, WeightKind::MarkovEmbedding), &markov.embedding));
        weights.push((
            role(WeightScope::Draft, WeightKind::MarkovProjection),
            &markov.projection,
        ));
        weights.push((
            role(WeightScope::Draft, WeightKind::ConfidenceWeight),
            &confidence.weight,
        ));
        weights.push((role(WeightScope::Draft, WeightKind::ConfidenceBias), &confidence.bias));
    }
    Ok(weights)
}

/// The blocks of a definition's drafter: its embedded head's, or its separate
/// draft's layers (an executed definition carries at most one drafter).
pub(crate) fn drafter_blocks(definition: &ModelDefinition) -> Vec<&Block> {
    match (&definition.head, &definition.draft) {
        (Some(head), _) => head.blocks.iter().map(|block| &block.block).collect(),
        (None, Some(draft)) => draft.blocks.iter().collect(),
        (None, None) => Vec::new(),
    }
}

/// The draft layer `block`'s attention and dense feed-forward.
pub(crate) fn draft_block(draft: &DraftDefinition, block: usize) -> Result<PairedBlock<'_>, PlanError> {
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
            return Err(PlanError::Unsupported("draft layer other than attention and feed-forward"));
        };
        if !matches!(feed_forward.op, FeedForward::Dense(_))
            || [paired.mixer_output, feed_forward.output]
                .iter()
                .any(|output| **output != OutputForm::Residual)
        {
            return Err(PlanError::Unsupported("draft layer form"));
        }
        if [paired.epsilon(), super::attention::head_norm_epsilon(attention, epsilon)?]
            .into_iter()
            .chain(feed_forward.epsilons())
            .any(|value| value != epsilon)
        {
            return Err(PlanError::Unsupported(
                "normalization epsilons differing within a draft",
            ));
        }
    }
    Ok(())
}
