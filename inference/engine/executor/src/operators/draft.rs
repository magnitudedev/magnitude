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
    DraftEmbedding, DraftFusion, DraftMask, DraftMethod, HeadHistory, HeadNorm,
    HyperConnectionMixing, MarkovHead, ModelDefinition, OutputForm, ProjectedRows, ResidualForm,
    SublayerIndex, TapPoint, ValueSource, WeightDescriptor, WeightKind, WeightRole, WeightScope,
};

/// The draft weights a target step reads: the fusion projection (or softmax
/// mix weights) and the fusion norm(s).
pub(crate) fn fusion_weights(draft: &DraftDefinition) -> Vec<(WeightRole, &WeightDescriptor)> {
    match &draft.fusion {
        DraftFusion::Linear { projection, norm } => vec![
            (role(WeightScope::Draft, WeightKind::DraftFusion), projection),
            (
                role(WeightScope::Draft, WeightKind::DraftFusionNorm),
                &norm.weight,
            ),
        ],
        DraftFusion::SoftmaxMix { weights, norms } => {
            let mut out = Vec::with_capacity(1 + norms.len());
            out.push((role(WeightScope::Draft, WeightKind::DraftFusion), weights));
            // Per-layer fusion norms live in `DraftSublayer(block, 0)` (and the
            // extra `f_rnn` norm, when present, in `WeightScope::Draft`).
            let layer_count = draft.blocks.len();
            for (index, norm) in norms.iter().enumerate() {
                if index < layer_count {
                    let scope = WeightScope::DraftSublayer(SublayerIndex {
                        block: u32::try_from(index).expect("draft block index fits u32"),
                        sublayer: 0,
                    });
                    out.push((role(scope, WeightKind::DraftFusionNorm), &norm.weight));
                } else {
                    out.push((
                        role(WeightScope::Draft, WeightKind::DraftFusionNorm),
                        &norm.weight,
                    ));
                }
            }
            out
        }
    }
}

/// Every drafter weight of a draft (all but its fusion), in model order.
pub(crate) fn draft_weights(
    draft: &DraftDefinition,
) -> Result<Vec<(WeightRole, &WeightDescriptor)>, PlanError> {
    let mut weights = Vec::new();
    if let DraftEmbedding::Own(table) = &draft.embedding {
        weights.push((role(WeightScope::Draft, WeightKind::Embedding), table));
    }
    if let DraftMask::Learned(mask) = &draft.mask {
        weights.push((role(WeightScope::Draft, WeightKind::DraftMaskEmbedding), mask));
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
    if let Some(context_projections) = &draft.context_projections {
        for (block, ctx) in context_projections.iter().enumerate() {
            let scope = WeightScope::DraftSublayer(SublayerIndex {
                block: u32::try_from(block)
                    .map_err(|_| PlanError::Arithmetic("draft block index"))?,
                sublayer: 0,
            });
            weights.push((role(scope, WeightKind::DraftContextKey), &ctx.key));
            weights.push((role(scope, WeightKind::DraftContextValue), &ctx.value));
            match &ctx.key_norm {
                HeadNorm::Rms(norm) => {
                    weights.push((role(scope, WeightKind::DraftContextKeyNorm), &norm.weight));
                }
                HeadNorm::L2Clip {
                    scale: Some(scale), ..
                } => {
                    weights.push((role(scope, WeightKind::DraftContextKeyNorm), scale));
                }
                HeadNorm::L2Clip { scale: None, .. } | HeadNorm::None => {}
            }
        }
    }
    weights.push((
        role(WeightScope::Draft, WeightKind::OutputNorm),
        &draft.output_norm.weight,
    ));
    // DSpark's confidence head is never evaluated (the service threshold is
    // zero), so no entry reads its weights and they are not imported.
    if let DraftMethod::DSpark { markov, .. } = &draft.method {
        match markov {
            MarkovHead::Bilinear {
                embedding,
                projection,
                ..
            } => {
                weights.push((role(WeightScope::Draft, WeightKind::MarkovEmbedding), embedding));
                weights.push((
                    role(WeightScope::Draft, WeightKind::MarkovProjection),
                    projection,
                ));
            }
            MarkovHead::RnnHidden(rnn) => {
                weights.push((
                    role(WeightScope::Draft, WeightKind::MarkovEmbedding),
                    &rnn.embedding,
                ));
                weights.push((
                    role(WeightScope::Draft, WeightKind::MarkovRnnWeight),
                    &rnn.rnn_proj.weight,
                ));
                weights.push((
                    role(WeightScope::Draft, WeightKind::MarkovRnnBias),
                    &rnn.rnn_proj.bias,
                ));
                weights.push((
                    role(WeightScope::Draft, WeightKind::MarkovHiddenOutWeight),
                    &rnn.hidden_out.weight,
                ));
                weights.push((
                    role(WeightScope::Draft, WeightKind::MarkovHiddenOutBias),
                    &rnn.hidden_out.bias,
                ));
            }
        }
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
    let expected_output = match &draft.residual {
        ResidualForm::Single | ResidualForm::BlockAttention { .. } => OutputForm::Residual,
        ResidualForm::HyperConnections(form) => match form.mixing {
            HyperConnectionMixing::Sinkhorn { .. } => OutputForm::Scatter,
            HyperConnectionMixing::LowRankGated { .. } => {
                return Err(PlanError::Unsupported(
                    "draft layers admit Sinkhorn hyper-connections only",
                ));
            }
        },
    };
    if !matches!(draft.residual, ResidualForm::Single) && draft.convolutions.is_some() {
        return Err(PlanError::Unsupported(
            "convolved draft layers require a single residual stream",
        ));
    }
    let epsilon = draft.output_norm.epsilon;
    if draft.fusion.norms().iter().any(|norm| norm.epsilon != epsilon) {
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
        if !matches!(
            feed_forward.op,
            FeedForward::Dense(_) | FeedForward::Routed(_) | FeedForward::GeneralRouted(_)
        ) || [paired.mixer_output, feed_forward.output]
            .iter()
            .any(|output| **output != expected_output)
        {
            return Err(PlanError::Unsupported("draft layer form"));
        }
        if draft.convolutions.is_some() && !matches!(feed_forward.op, FeedForward::Dense(_)) {
            return Err(PlanError::Unsupported(
                "a convolved draft layer requires a dense feed-forward",
            ));
        }
        // A convolved backbone runs its layers unfused: plain query, key and
        // value projections of the convolved rows.
        if draft.convolutions.is_some()
            && (!matches!(attention.gate, AttentionGate::None)
                || !matches!(
                    super::attention::fresh_rows(attention),
                    super::attention::FreshRows::Own(ProjectedRows {
                        value: ValueSource::Projected(_),
                        ..
                    })
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
    let has_softmax_mix = matches!(draft.fusion, DraftFusion::SoftmaxMix { .. });
    let has_learned_mask = matches!(draft.mask, DraftMask::Learned(_));
    let has_context_proj = draft.context_projections.is_some();
    let has_rnn_markov = matches!(
        draft.method,
        DraftMethod::DSpark {
            markov: MarkovHead::RnnHidden(_),
            ..
        }
    );
    if has_softmax_mix || has_learned_mask || has_context_proj || has_rnn_markov {
        if !(has_softmax_mix && has_learned_mask && has_rnn_markov) {
            return Err(PlanError::Unsupported(
                "RNN DSpark requires SoftmaxMix fusion, Learned mask, and RnnHidden Markov head",
            ));
        }
        if let DraftFusion::SoftmaxMix { norms, .. } = &draft.fusion {
            if norms.len() != draft.blocks.len() + 1 {
                return Err(PlanError::Topology(
                    "RNN DSpark SoftmaxMix fusion requires one norm per draft block plus f_rnn",
                ));
            }
        }
        if let Some(ctx) = &draft.context_projections {
            if ctx.len() != draft.blocks.len() {
                return Err(PlanError::Topology(
                    "draft context projections count must match draft block count",
                ));
            }
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

/// Fuse `taps` target residuals (`[taps, M, hidden]` F32) via
/// `DraftFusion::SoftmaxMix`: for each output `o`, compute
/// `w = softmax(weights[o, :])`, `y = sum_t w[t] * taps[t]`, and apply
/// `norms[o]` RMSNorm (`[hidden]`) in F32.
#[allow(dead_code)]
pub(crate) fn softmax_mix_fuse(
    taps: &[&[f32]],
    weights: &[f32],
    norms: &[(&[f32], f32)],
    rows: usize,
    hidden: usize,
) -> Result<Vec<Vec<f32>>, PlanError> {
    let n_taps = taps.len();
    let n_out = norms.len();
    if weights.len() != n_out * n_taps {
        return Err(PlanError::Topology("softmax-mix weights shape mismatch"));
    }
    let mut outputs = Vec::with_capacity(n_out);
    for (o, &(gamma, eps)) in norms.iter().enumerate() {
        let row_w = &weights[o * n_taps..(o + 1) * n_taps];
        let max_w = row_w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row_w.iter().map(|&v| (v - max_w).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        let coeffs: Vec<f32> = exps.iter().map(|&e| e / sum_e).collect();

        let mut fused = vec![0.0f32; rows * hidden];
        for r in 0..rows {
            let base = r * hidden;
            for (t, &c) in coeffs.iter().enumerate() {
                let src = &taps[t][base..base + hidden];
                for i in 0..hidden {
                    fused[base + i] = c.mul_add(src[i], fused[base + i]);
                }
            }
            let sq: f32 = fused[base..base + hidden].iter().map(|&v| v * v).sum();
            let inv = 1.0 / (sq / hidden as f32 + eps).sqrt();
            for i in 0..hidden {
                fused[base + i] *= inv * gamma[i];
            }
        }
        outputs.push(fused);
    }
    Ok(outputs)
}

/// One recurrent slot step of `MarkovHead::RnnHidden` (matching SGLang's
/// `Inkling2MTPHead._rnn_step` / `_rnn_gates`):
///   `proj = rnn_proj([h_prev ; markov_w1[prev_token] ; hidden_i ; f_rnn])`
///   `gate = sigmoid(proj[0..R])`
///   `h_next = gate * h_prev + (1 - gate) * tanh(proj[R..2R])`
///   `latent = tanh(proj[2R..3R])`
/// Returns `(h_next [rank], latent [rank])`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn dot_f32_simd(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;
    let mut ac = a.chunks_exact(8);
    let mut bc = b.chunks_exact(8);
    for (x, y) in ac.by_ref().zip(bc.by_ref()) {
        s0 = x[0].mul_add(y[0], s0);
        s1 = x[1].mul_add(y[1], s1);
        s2 = x[2].mul_add(y[2], s2);
        s3 = x[3].mul_add(y[3], s3);
        s4 = x[4].mul_add(y[4], s4);
        s5 = x[5].mul_add(y[5], s5);
        s6 = x[6].mul_add(y[6], s6);
        s7 = x[7].mul_add(y[7], s7);
    }
    let mut sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
    for (&x, &y) in ac.remainder().iter().zip(bc.remainder().iter()) {
        sum = x.mul_add(y, sum);
    }
    sum
}

/// Precompute `rnn_proj_b + W_frnn @ f_rnn` once per slot per draft cycle,
/// eliminating 40% of the RNN gate matvec work on steps `1..steps`.
pub(crate) fn precompute_rnn_frnn_bias(
    f_rnn: &[f32],
    rank: usize,
    hidden: usize,
    rnn_proj_w: &[f32],
    rnn_proj_b: &[f32],
) -> Vec<f32> {
    let in_dim = 2 * rank + 2 * hidden;
    let frnn_off = 2 * rank + hidden;
    let mut base = vec![0.0f32; 3 * rank];
    for o in 0..3 * rank {
        let row = &rnn_proj_w[o * in_dim..(o + 1) * in_dim];
        base[o] = rnn_proj_b[o] + dot_f32_simd(&row[frnn_off..], f_rnn);
    }
    base
}

pub(crate) fn step_rnn_markov_gates_precomputed(
    h_prev: &[f32],
    prev_emb: &[f32],
    hidden_i: &[f32],
    base_bias: &[f32],
    rnn_proj_w: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let rank = h_prev.len();
    let hidden = hidden_i.len();
    let in_dim = 2 * rank + 2 * hidden;
    let mut gates = vec![0.0f32; 3 * rank];
    for o in 0..3 * rank {
        let row = &rnn_proj_w[o * in_dim..(o + 1) * in_dim];
        let d_h = dot_f32_simd(&row[0..rank], h_prev);
        let d_e = dot_f32_simd(&row[rank..2 * rank], prev_emb);
        let d_x = dot_f32_simd(&row[2 * rank..2 * rank + hidden], hidden_i);
        gates[o] = base_bias[o] + d_h + d_e + d_x;
    }

    let mut h_next = vec![0.0f32; rank];
    let mut latent = vec![0.0f32; rank];
    for r in 0..rank {
        let gate = 1.0 / (1.0 + (-gates[r]).exp());
        let cand = gates[rank + r].tanh();
        h_next[r] = gate * h_prev[r] + (1.0 - gate) * cand;
        latent[r] = gates[2 * rank + r].tanh();
    }
    (h_next, latent)
}

/// One recurrent slot step of `MarkovHead::RnnHidden` when
/// `step_bias = rnn_proj_b + W_frnn @ f_rnn + W_x @ hidden_i` has already been
/// computed (e.g. on GPU via `vision_linear`), using the contiguous
/// `[3 * rank, 2 * rank]` recurrent/embedding slice `w_he = [W_h | W_e]`.
pub(crate) fn step_rnn_markov_gates_he(
    h_prev: &[f32],
    prev_emb: &[f32],
    step_bias: &[f32],
    w_he: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let rank = h_prev.len();
    let he_dim = 2 * rank;
    let mut gates = vec![0.0f32; 3 * rank];
    for o in 0..3 * rank {
        let row = &w_he[o * he_dim..(o + 1) * he_dim];
        let d_h = dot_f32_simd(&row[0..rank], h_prev);
        let d_e = dot_f32_simd(&row[rank..he_dim], prev_emb);
        gates[o] = step_bias[o] + d_h + d_e;
    }

    let mut h_next = vec![0.0f32; rank];
    let mut latent = vec![0.0f32; rank];
    for r in 0..rank {
        let gate = 1.0 / (1.0 + (-gates[r]).exp());
        let cand = gates[rank + r].tanh();
        h_next[r] = gate * h_prev[r] + (1.0 - gate) * cand;
        latent[r] = gates[2 * rank + r].tanh();
    }
    (h_next, latent)
}

#[allow(dead_code)]
pub(crate) fn step_rnn_markov_gates(
    h_prev: &[f32],
    prev_emb: &[f32],
    hidden_i: &[f32],
    f_rnn: &[f32],
    rnn_proj_w: &[f32],
    rnn_proj_b: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let base = precompute_rnn_frnn_bias(f_rnn, h_prev.len(), hidden_i.len(), rnn_proj_w, rnn_proj_b);
    step_rnn_markov_gates_precomputed(h_prev, prev_emb, hidden_i, &base, rnn_proj_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dspark_softmax_mix_and_folded_rnn_markov_parity() {
        let (rows, hidden, rank, vocab) = (2, 8, 4, 12);
        let tap0: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let tap1: Vec<f32> = (0..rows * hidden).map(|i| -(i as f32 + 1.0) * 0.05).collect();
        let weights = vec![0.0f32, 0.0, 1.0, -1.0]; // 2 outputs over 2 taps
        let gamma = vec![1.0f32; hidden];
        let fused = softmax_mix_fuse(
            &[&tap0, &tap1],
            &weights,
            &[(&gamma, 1e-6), (&gamma, 1e-6)],
            rows,
            hidden,
        )
        .unwrap();
        assert_eq!(fused.len(), 2);
        assert!(fused.iter().flatten().all(|v| v.is_finite()));

        let in_dim = 2 * rank + 2 * hidden;
        let rnn_w: Vec<f32> = (0..3 * rank * in_dim)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.02)
            .collect();
        let rnn_b: Vec<f32> = (0..3 * rank).map(|i| (i as f32) * 0.01).collect();
        let h_prev = vec![0.1f32; rank];
        let prev_emb = vec![0.2f32; rank];
        let hidden_i = &fused[0][..hidden];
        let f_rnn = &fused[1][..hidden];

        let (h_next, latent) = step_rnn_markov_gates(
            &h_prev, &prev_emb, hidden_i, f_rnn, &rnn_w, &rnn_b,
        );
        assert_eq!(h_next.len(), rank);
        assert_eq!(latent.len(), rank);
        assert!(h_next.iter().all(|v| v.is_finite()));
        assert!(latent.iter().all(|v| v.is_finite() && v.abs() <= 1.0));

        // Verify split `step_rnn_markov_gates_he` matches full `step_rnn_markov_gates`.
        let mut w_he = Vec::with_capacity(3 * rank * 2 * rank);
        let mut step_bias = vec![0.0f32; 3 * rank];
        for o in 0..3 * rank {
            let row = &rnn_w[o * in_dim..(o + 1) * in_dim];
            w_he.extend_from_slice(&row[0..2 * rank]);
            let d_x = dot_f32_simd(&row[2 * rank..2 * rank + hidden], hidden_i);
            let d_f = dot_f32_simd(&row[2 * rank + hidden..], f_rnn);
            step_bias[o] = rnn_b[o] + d_x + d_f;
        }
        let (h_next_he, latent_he) =
            step_rnn_markov_gates_he(&h_prev, &prev_emb, &step_bias, &w_he);
        for (a, b) in h_next.iter().zip(&h_next_he) {
            assert!((a - b).abs() < 1e-6, "h_next mismatch: {a} vs {b}");
        }
        for (a, b) in latent.iter().zip(&latent_he) {
            assert!((a - b).abs() < 1e-6, "latent mismatch: {a} vs {b}");
        }
        let _ = vocab;
    }
}
