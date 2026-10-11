//! A separate draft component (DFlash, DSpark; plan §3.8).
//!
//! The draft is a small decoder conditioned on the target's residual stream.
//! For every token the target commits, the residuals entering the tapped
//! target layers are concatenated, fused (`fusion`, then `fusion_norm`) into
//! one context feature, and each draft layer's key and value projections of
//! that feature (without the layer's input norm) enter the draft's history.
//! A draft pass then runs one block `[anchor, mask, …]` through the draft
//! layers over that history and the block (all of it, or its rows up to the
//! reading row, per layer's [`BlockAttention`]), and reads the block's slots
//! out through the target's vocabulary projection.

use crate::decoder::{self, Context};
use crate::{
    checked_product, expect_shape, Block, Decoder, DefinitionError, KeyValue, Operator, RmsNorm,
    SublayerIndex, TokenId, WeightDescriptor,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// A target residual the draft reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TapPoint {
    /// The residual entering this target sublayer.
    Sublayer(SublayerIndex),
    /// The final residual, before the decoder's output norm.
    Exit,
}

/// The table that embeds the block's tokens (raw rows: no scale, no norm).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DraftEmbedding {
    /// The target decoder's token embedding.
    Target,
    /// The draft's own `[vocabulary, hidden]` table.
    Own(WeightDescriptor),
}

/// Which block slots predict which positions. The block's first slot holds
/// the anchor (the last committed token, at position `n`); the others hold
/// the mask token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockLayout {
    /// Slot `i ≥ 1` predicts position `n + i` (DFlash): a block of `B` rows
    /// drafts at most `B − 1` tokens.
    MaskSlots,
    /// Slot `i ≥ 0` predicts position `n + 1 + i` (DSpark,
    /// `sample_from_anchor`): a block of `B` rows drafts at most `B` tokens.
    AnchorFirst,
}

/// How one draft layer's block rows attend the block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockAttention {
    /// Each row reads the block's rows up to and including itself.
    Causal,
    /// Each row reads the whole block.
    Bidirectional,
}

/// DSpark's rank-`R` Markov bias: slot `i`'s logits gain
/// `projection · embedding[previous_i]`, where `previous_0` is the anchor
/// and `previous_{i+1}` is slot `i`'s selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarkovHead {
    pub rank: u64,
    /// `[vocabulary, rank]`.
    pub embedding: WeightDescriptor,
    /// `[vocabulary, rank]`.
    pub projection: WeightDescriptor,
}

/// DSpark's acceptance estimate of slot `i`:
/// `sigmoid(weight · [hidden_i ; markov.embedding[previous_i]] + bias)`,
/// where `hidden_i` is the slot's output-normed row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceHead {
    /// `[hidden + rank]`.
    pub weight: WeightDescriptor,
    /// `[1]`.
    pub bias: WeightDescriptor,
}

/// One grouped dynamic causal convolution of DFlash2, applied around a draft
/// sublayer over the rows of one block. The sublayer's normed input rows
/// `x` project to per-row coefficients `dynamic = projection · x`, viewed
/// `[2, kernel, groups]`; half `h` of the convolution of rows `y` is
/// `out[r, c] = Σ_{o < kernel, o ≤ r} (base[h, o, c] + dynamic[r, h, o, c / group]) · y[r − o, c]`
/// with `r` counted from the block's first row. Half 0 convolves the normed
/// input the operator reads; half 1 convolves the operator's output before
/// the residual add, with the same row's second-half coefficients.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DynamicConvolution {
    /// `[2, kernel, hidden]`.
    pub base: WeightDescriptor,
    /// `[2 · kernel · groups, hidden]`.
    pub projection: WeightDescriptor,
}

/// A DFlash2 draft layer's convolutions: around its attention and around its
/// feed-forward.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerConvolutions {
    pub attention: DynamicConvolution,
    pub feed_forward: DynamicConvolution,
}

/// DFlash2's ordered candidate path. Each proposing row keeps its `top_k`
/// highest vocabulary logits as candidates; row `i` then selects, in order,
/// the candidate maximizing
/// `logit + (predecessor[previous_i] ∘ (hidden · h_i)) · successor[candidate]`,
/// where `h_i` is the row's output-normed hidden row, `previous_0` is the
/// anchor and `previous_{i+1}` is row `i`'s selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateSelector {
    pub rank: u64,
    pub top_k: u64,
    /// `[rank, hidden]`.
    pub hidden: WeightDescriptor,
    /// `[vocabulary, rank]`.
    pub predecessor: WeightDescriptor,
    /// `[vocabulary, rank]`.
    pub successor: WeightDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DraftMethod {
    /// Every slot's proposal is its own argmax.
    DFlash,
    /// Slots chain through the Markov bias and carry a confidence.
    DSpark {
        markov: MarkovHead,
        confidence: ConfidenceHead,
    },
    /// Every draft sublayer is wrapped in a grouped dynamic causal
    /// convolution over its block, and proposals follow an ordered
    /// candidate path.
    DFlash2 {
        /// Convolution taps (`kernel`) and channels per coefficient group.
        kernel: u64,
        group: u64,
        /// One per draft layer.
        convolutions: Vec<LayerConvolutions>,
        selector: CandidateSelector,
    },
}

/// The separate-draft method a draft's definition implements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DraftVariant {
    DFlash,
    DSpark,
    DFlash2,
}

impl fmt::Display for DraftVariant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DFlash => "DFlash",
            Self::DSpark => "DSpark",
            Self::DFlash2 => "DFlash2",
        })
    }
}

impl DraftMethod {
    pub fn variant(&self) -> DraftVariant {
        match self {
            Self::DFlash => DraftVariant::DFlash,
            Self::DSpark { .. } => DraftVariant::DSpark,
            Self::DFlash2 { .. } => DraftVariant::DFlash2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DraftDefinition {
    pub method: DraftMethod,
    /// Target residuals in fusion column order: tap `t` owns columns
    /// `[t·hidden, (t + 1)·hidden)` of `fusion`.
    pub taps: Vec<TapPoint>,
    /// `[hidden, taps·hidden]`.
    pub fusion: WeightDescriptor,
    pub fusion_norm: RmsNorm,
    pub embedding: DraftEmbedding,
    /// Draft layers over the target's width. Their attention reads the
    /// injected context and the fresh block.
    pub blocks: Vec<Block>,
    /// One per draft layer: how its block rows attend the block.
    pub block_attention: Vec<BlockAttention>,
    pub output_norm: RmsNorm,
    /// Rows of the trained draft block, anchor included: the widest block a
    /// pass runs (`block_rows`).
    pub block_size: u64,
    pub mask_token: TokenId,
    pub layout: BlockLayout,
}

impl DraftDefinition {
    /// The most tokens one block drafts.
    pub fn max_proposals(&self) -> u64 {
        match self.layout {
            BlockLayout::MaskSlots => self.block_size - 1,
            BlockLayout::AnchorFirst => self.block_size,
        }
    }

    /// Rows of the block a pass drafting `proposals` tokens runs: through its
    /// last proposing row, as serving engines size the block to the verify
    /// window. A causal layer's rows do not read later rows, so only the
    /// bidirectional layers see fewer mask rows than in training.
    pub fn block_rows(&self, proposals: u64) -> u64 {
        (self.proposal_row(proposals.max(1) - 1) + 1).clamp(2, self.block_size)
    }

    /// The block row whose output drafts proposal `proposal` (the token at
    /// position `n + 1 + proposal`).
    pub fn proposal_row(&self, proposal: u64) -> u64 {
        match self.layout {
            BlockLayout::MaskSlots => proposal + 1,
            BlockLayout::AnchorFirst => proposal,
        }
    }

    pub fn validate(&self, decoder: &Decoder, coordinate_axes: u8) -> Result<(), DefinitionError> {
        let hidden = decoder.hidden;
        if self.taps.is_empty() || self.blocks.is_empty() || self.block_size < 2 {
            return Err(DefinitionError::new(
                "draft has no taps, blocks or proposals",
            ));
        }
        if self.block_attention.len() != self.blocks.len() {
            return Err(DefinitionError::new(
                "draft block attention differs from its layer count",
            ));
        }
        if u64::from(self.mask_token.0) >= decoder.vocabulary {
            return Err(DefinitionError::new(
                "draft mask token is outside the vocabulary",
            ));
        }
        for (index, tap) in self.taps.iter().enumerate() {
            if self.taps[..index].contains(tap) {
                return Err(DefinitionError::new("draft taps repeat a target residual"));
            }
            if let TapPoint::Sublayer(sublayer) = tap {
                let exists = decoder
                    .blocks
                    .get(sublayer.block as usize)
                    .is_some_and(|block| (sublayer.sublayer as usize) < block.sublayers.len());
                if !exists {
                    return Err(DefinitionError::new("draft tap names no target sublayer"));
                }
            }
        }
        let fused = checked_product(&[self.taps.len() as u64, hidden])?;
        expect_shape(&self.fusion, &[hidden, fused], "draft fusion projection")?;
        decoder::validate_rms(&self.fusion_norm, hidden, "draft fusion norm")?;
        if let DraftEmbedding::Own(table) = &self.embedding {
            expect_shape(
                table,
                &[decoder.vocabulary, hidden],
                "draft token embedding",
            )?;
        }
        decoder::validate_blocks(
            &self.blocks,
            &Context {
                hidden,
                coordinate_axes,
                per_layer: None,
            },
        )?;
        // Every draft attention layer injects its own context history.
        for block in &self.blocks {
            for sublayer in &block.sublayers {
                if let Operator::Attention(attention) = &sublayer.op {
                    if !matches!(attention.key_value, KeyValue::Owned { .. }) {
                        return Err(DefinitionError::new(
                            "draft attention layers own their key and value history",
                        ));
                    }
                }
            }
        }
        decoder::validate_rms(&self.output_norm, hidden, "draft output norm")?;
        match &self.method {
            DraftMethod::DFlash => {}
            DraftMethod::DSpark { markov, confidence } => {
                if markov.rank == 0 {
                    return Err(DefinitionError::new("draft Markov head has no rank"));
                }
                let table = [decoder.vocabulary, markov.rank];
                expect_shape(&markov.embedding, &table, "draft Markov embedding")?;
                expect_shape(&markov.projection, &table, "draft Markov projection")?;
                let joined = hidden
                    .checked_add(markov.rank)
                    .ok_or_else(|| DefinitionError::new("draft confidence width overflows"))?;
                expect_shape(&confidence.weight, &[joined], "draft confidence weight")?;
                expect_shape(&confidence.bias, &[1], "draft confidence bias")?;
            }
            DraftMethod::DFlash2 {
                kernel,
                group,
                convolutions,
                selector,
            } => {
                if self.layout != BlockLayout::MaskSlots {
                    return Err(DefinitionError::new(
                        "a DFlash2 draft proposes from its mask slots",
                    ));
                }
                if *kernel == 0 || *group == 0 || !hidden.is_multiple_of(*group) {
                    return Err(DefinitionError::new(
                        "draft convolution groups do not divide the hidden width",
                    ));
                }
                if convolutions.len() != self.blocks.len() {
                    return Err(DefinitionError::new(
                        "a DFlash2 draft convolves every draft layer",
                    ));
                }
                let coefficients = checked_product(&[2, *kernel, hidden / *group])?;
                for layer in convolutions {
                    for convolution in [&layer.attention, &layer.feed_forward] {
                        expect_shape(
                            &convolution.base,
                            &[2, *kernel, hidden],
                            "draft convolution base",
                        )?;
                        expect_shape(
                            &convolution.projection,
                            &[coefficients, hidden],
                            "draft convolution projection",
                        )?;
                    }
                }
                if selector.rank == 0 || selector.top_k == 0 || selector.top_k > decoder.vocabulary
                {
                    return Err(DefinitionError::new("draft candidate selector geometry"));
                }
                let codebook = [decoder.vocabulary, selector.rank];
                expect_shape(
                    &selector.hidden,
                    &[selector.rank, hidden],
                    "draft selector projection",
                )?;
                expect_shape(
                    &selector.predecessor,
                    &codebook,
                    "draft predecessor codebook",
                )?;
                expect_shape(&selector.successor, &codebook, "draft successor codebook")?;
            }
        }
        Ok(())
    }
}
