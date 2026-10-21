//! A separate draft component (DFlash, DSpark, DFlash2, LiLiCorr; plan
//! §3.8).
//!
//! The draft is a small decoder conditioned on the target's residual stream.
//! For every token the target commits, the residuals entering the tapped
//! target layers are concatenated, fused (`fusion`, then `fusion_norm`) into
//! one context feature, and each draft layer's key and value projections of
//! that feature (without the layer's input norm) enter the draft's history.
//! A draft pass then runs one block `[anchor, mask, …]` through the draft
//! layers over that history and the block (all of it, or its rows up to the
//! reading row, per layer's [`BlockAttention`]), and reads the block's slots
//! out through an owned or target-shared vocabulary projection. Its
//! [`DraftMethod`] turns the slots' logits into proposals.

use crate::decoder::{self, Context};
use crate::{
    Block, Decoder, DefinitionError, KeyValue, Operator, ResidualForm, RmsNorm, SublayerIndex,
    TokenId, WeightDescriptor, WeightKind, WeightRole, WeightScope, checked_product,
    expect_shape,
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

/// One grouped dynamic causal convolution of a convolved draft backbone
/// (DFlash2's), applied around a draft sublayer over the rows of one block.
/// The sublayer's normed input rows
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

/// A convolved draft layer's convolutions: around its attention and around
/// its feed-forward.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerConvolutions {
    pub attention: DynamicConvolution,
    pub feed_forward: DynamicConvolution,
}

/// A convolved draft backbone (DFlash2's): every draft sublayer is wrapped in
/// a grouped dynamic causal convolution over its block. It is a property of
/// the draft's layers, independent of how proposals are read out.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BlockConvolutions {
    /// Convolution taps.
    pub kernel: u64,
    /// Channels per coefficient group.
    pub group: u64,
    /// One per draft layer.
    pub layers: Vec<LayerConvolutions>,
}

/// A linear map with a bias: `y = weight · x + bias`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Affine {
    /// `[outputs, inputs]`.
    pub weight: WeightDescriptor,
    /// `[outputs]`.
    pub bias: WeightDescriptor,
}

impl Affine {
    fn validate(&self, outputs: u64, inputs: u64, role: &str) -> Result<(), DefinitionError> {
        expect_shape(&self.weight, &[outputs, inputs], role)?;
        expect_shape(&self.bias, &[outputs], role)
    }
}

/// A centered (layer) norm with a stored multiplier and bias.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CenteredNorm {
    pub weight: WeightDescriptor,
    pub bias: WeightDescriptor,
    pub epsilon: f64,
}

/// Per-candidate features a corrector reads, in order: the candidate's log
/// probability under the slot's full softmax, its probability, its log
/// probability less the slot's best, its rank over `top_k − 1` (0 when
/// `top_k` is 1), and whether it is the slot's best (1 or 0).
pub const CORRECTOR_FEATURES: u64 = 5;

/// One pre-norm transformer layer of a candidate corrector over a block's
/// candidate nodes: RMS-normed nodes project to heads' queries, keys and
/// values (`attention_input` rows: every query, then every key, then every
/// value, head-major), attend every node of the block with the per-head
/// additive `CandidateCorrector::attention_bias`, and project back into the
/// residual; then an RMS-normed `down · silu(up · x)` joins the residual.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CorrectorLayer {
    pub attention_norm: RmsNorm,
    /// `[3·width, width]`.
    pub attention_input: Affine,
    /// `[width, width]`.
    pub attention_output: Affine,
    pub mlp_norm: RmsNorm,
    /// `[mlp, width]`.
    pub up: Affine,
    /// `[width, mlp]`.
    pub down: Affine,
}

/// LiLiCorr's candidate corrector over a block's `slots` proposing rows.
///
/// Each slot keeps its `top_k` highest draft logits as candidates, in
/// descending logit order (rank 0 the best). Candidate `k` of slot `j` is
/// node `j·top_k + k` of its block; its row is the sum of its token's
/// projected embedding (`token`, or the raw embedding row when absent), the
/// slot's projected output-normed hidden row (`pass`), its projected
/// [`CORRECTOR_FEATURES`] (`feature_down · silu(feature_up ·
/// feature_norm(features))`), and the slot and rank embeddings. The nodes run
/// the corrector layers, then `output_norm`. The block's anchor state is the
/// anchor-normed `context` projection of the fused target feature that
/// selected the anchor (the draft's fusion-normed context row). Each node's
/// factor is `silu(factor_input · [x ; a ; x ∘ a])` for its output-normed row
/// `x` and the anchor state `a`; its outgoing and incoming vectors are the
/// unit-normalized (`vector_epsilon`) `outgoing` and `incoming` maps of the
/// factor, and the anchor's is the unit-normalized `anchor` map of `a`.
///
/// Candidate `c` of slot 0 scores `logit_scale · incoming(0, c) · anchor`;
/// candidate `c` of slot `j > 0` following candidate `p` of slot `j − 1`
/// scores `logit_scale · incoming(j, c) · outgoing(j − 1, p)`. Proposals walk
/// the slots greedily: each slot selects its best-scoring candidate (the
/// first on ties) after the previous slot's selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateCorrector {
    /// The corrector's node width.
    pub width: u64,
    pub top_k: u64,
    pub heads: u64,
    /// The width of the outgoing, incoming and anchor vectors.
    pub factor: u64,
    /// `[width, hidden]`; absent when `width` is the draft's hidden width and
    /// embedding rows enter the nodes directly.
    pub token: Option<Affine>,
    /// `[width, hidden]`.
    pub pass: Affine,
    /// `[width, hidden]`.
    pub context: Affine,
    /// Over [`CORRECTOR_FEATURES`] values.
    pub feature_norm: CenteredNorm,
    /// `[width, CORRECTOR_FEATURES]`.
    pub feature_up: Affine,
    /// `[width, width]`.
    pub feature_down: Affine,
    /// `[slots, width]`.
    pub slot_embedding: WeightDescriptor,
    /// `[top_k, width]`.
    pub rank_embedding: WeightDescriptor,
    /// `[heads, slots·top_k, slots·top_k]`: head `h`'s additive bias of query
    /// node `q` attending key node `k`.
    pub attention_bias: WeightDescriptor,
    pub layers: Vec<CorrectorLayer>,
    pub output_norm: RmsNorm,
    pub anchor_norm: RmsNorm,
    /// `[width, 3·width]`.
    pub factor_input: Affine,
    /// `[factor, width]` each.
    pub outgoing: Affine,
    pub incoming: Affine,
    pub anchor: Affine,
    pub vector_epsilon: f64,
    /// A positive multiplier on every score; it does not change a selection.
    pub logit_scale: f64,
}

/// A site of a corrector's affine maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorrectorSite {
    Token,
    Pass,
    Context,
    FeatureUp,
    FeatureDown,
    FactorInput,
    Outgoing,
    Incoming,
    Anchor,
    /// A corrector layer's (scope `CorrectorLayer`).
    AttentionInput,
    AttentionOutput,
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AffinePart {
    Weight,
    Bias,
}

/// The semantic role of a corrector weight within its scope (`Draft`, or
/// `CorrectorLayer(layer)` for a layer's maps and norms).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorrectorWeight {
    Affine {
        site: CorrectorSite,
        part: AffinePart,
    },
    FeatureNorm(AffinePart),
    SlotEmbedding,
    RankEmbedding,
    AttentionBias,
    OutputNorm,
    AnchorNorm,
    /// A corrector layer's.
    AttentionNorm,
    MlpNorm,
}

impl CandidateCorrector {
    /// Every corrector weight with its role, in model order.
    pub fn weights(&self) -> Vec<(WeightRole, &WeightDescriptor)> {
        fn role(scope: WeightScope, weight: CorrectorWeight) -> WeightRole {
            WeightRole {
                scope,
                kind: WeightKind::Corrector(weight),
            }
        }
        fn affine<'a>(
            out: &mut Vec<(WeightRole, &'a WeightDescriptor)>,
            scope: WeightScope,
            site: CorrectorSite,
            map: &'a Affine,
        ) {
            for (part, weight) in [(AffinePart::Weight, &map.weight), (AffinePart::Bias, &map.bias)] {
                out.push((role(scope, CorrectorWeight::Affine { site, part }), weight));
            }
        }
        let mut out = Vec::new();
        let draft = WeightScope::Draft;
        if let Some(token) = &self.token {
            affine(&mut out, draft, CorrectorSite::Token, token);
        }
        affine(&mut out, draft, CorrectorSite::Pass, &self.pass);
        affine(&mut out, draft, CorrectorSite::Context, &self.context);
        out.push((
            role(draft, CorrectorWeight::FeatureNorm(AffinePart::Weight)),
            &self.feature_norm.weight,
        ));
        out.push((
            role(draft, CorrectorWeight::FeatureNorm(AffinePart::Bias)),
            &self.feature_norm.bias,
        ));
        affine(&mut out, draft, CorrectorSite::FeatureUp, &self.feature_up);
        affine(&mut out, draft, CorrectorSite::FeatureDown, &self.feature_down);
        out.push((
            role(draft, CorrectorWeight::SlotEmbedding),
            &self.slot_embedding,
        ));
        out.push((
            role(draft, CorrectorWeight::RankEmbedding),
            &self.rank_embedding,
        ));
        out.push((
            role(draft, CorrectorWeight::AttentionBias),
            &self.attention_bias,
        ));
        for (index, layer) in self.layers.iter().enumerate() {
            let scope =
                WeightScope::CorrectorLayer(u32::try_from(index).expect("corrector layers fit u32"));
            out.push((
                role(scope, CorrectorWeight::AttentionNorm),
                &layer.attention_norm.weight,
            ));
            affine(
                &mut out,
                scope,
                CorrectorSite::AttentionInput,
                &layer.attention_input,
            );
            affine(
                &mut out,
                scope,
                CorrectorSite::AttentionOutput,
                &layer.attention_output,
            );
            out.push((role(scope, CorrectorWeight::MlpNorm), &layer.mlp_norm.weight));
            affine(&mut out, scope, CorrectorSite::Up, &layer.up);
            affine(&mut out, scope, CorrectorSite::Down, &layer.down);
        }
        out.push((
            role(draft, CorrectorWeight::OutputNorm),
            &self.output_norm.weight,
        ));
        out.push((
            role(draft, CorrectorWeight::AnchorNorm),
            &self.anchor_norm.weight,
        ));
        affine(&mut out, draft, CorrectorSite::FactorInput, &self.factor_input);
        affine(&mut out, draft, CorrectorSite::Outgoing, &self.outgoing);
        affine(&mut out, draft, CorrectorSite::Incoming, &self.incoming);
        affine(&mut out, draft, CorrectorSite::Anchor, &self.anchor);
        out
    }

    /// Candidate nodes of one block.
    pub fn nodes(&self, slots: u64) -> Result<u64, DefinitionError> {
        checked_product(&[slots, self.top_k])
    }

    /// The width of the corrector layers' feed-forward.
    pub fn mlp(&self) -> u64 {
        self.layers
            .first()
            .and_then(|layer| layer.up.weight.shape.first().copied())
            .unwrap_or(0)
    }

    fn validate(&self, hidden: u64, vocabulary: u64, slots: u64) -> Result<(), DefinitionError> {
        let width = self.width;
        if width == 0
            || self.heads == 0
            || !width.is_multiple_of(self.heads)
            || self.factor == 0
            || self.top_k == 0
            || self.top_k > vocabulary
            || slots == 0
            || self.layers.is_empty()
        {
            return Err(DefinitionError::new("draft corrector geometry"));
        }
        if !self.vector_epsilon.is_finite()
            || self.vector_epsilon <= 0.0
            || !self.logit_scale.is_finite()
            || self.logit_scale <= 0.0
            || !self.feature_norm.epsilon.is_finite()
            || self.feature_norm.epsilon <= 0.0
        {
            return Err(DefinitionError::new("draft corrector scales"));
        }
        match &self.token {
            Some(token) => token.validate(width, hidden, "draft corrector token projection")?,
            None if width == hidden => {}
            None => {
                return Err(DefinitionError::new(
                    "a draft corrector narrower than the draft projects its token embeddings",
                ));
            }
        }
        self.pass.validate(width, hidden, "draft corrector pass projection")?;
        self.context
            .validate(width, hidden, "draft corrector context projection")?;
        expect_shape(
            &self.feature_norm.weight,
            &[CORRECTOR_FEATURES],
            "draft corrector feature norm",
        )?;
        expect_shape(
            &self.feature_norm.bias,
            &[CORRECTOR_FEATURES],
            "draft corrector feature norm",
        )?;
        self.feature_up
            .validate(width, CORRECTOR_FEATURES, "draft corrector feature projection")?;
        self.feature_down
            .validate(width, width, "draft corrector feature projection")?;
        expect_shape(
            &self.slot_embedding,
            &[slots, width],
            "draft corrector slot embedding",
        )?;
        expect_shape(
            &self.rank_embedding,
            &[self.top_k, width],
            "draft corrector rank embedding",
        )?;
        let nodes = self.nodes(slots)?;
        expect_shape(
            &self.attention_bias,
            &[self.heads, nodes, nodes],
            "draft corrector attention bias",
        )?;
        let mlp = self.mlp();
        if mlp == 0 {
            return Err(DefinitionError::new("draft corrector feed-forward width"));
        }
        let joined = checked_product(&[3, width])?;
        for layer in &self.layers {
            decoder::validate_rms(&layer.attention_norm, width, "draft corrector layer norm")?;
            layer
                .attention_input
                .validate(joined, width, "draft corrector attention input")?;
            layer
                .attention_output
                .validate(width, width, "draft corrector attention output")?;
            decoder::validate_rms(&layer.mlp_norm, width, "draft corrector layer norm")?;
            layer.up.validate(mlp, width, "draft corrector up projection")?;
            layer
                .down
                .validate(width, mlp, "draft corrector down projection")?;
        }
        decoder::validate_rms(&self.output_norm, width, "draft corrector output norm")?;
        decoder::validate_rms(&self.anchor_norm, width, "draft corrector anchor norm")?;
        self.factor_input
            .validate(width, joined, "draft corrector factor input")?;
        for (map, role) in [
            (&self.outgoing, "draft corrector outgoing head"),
            (&self.incoming, "draft corrector incoming head"),
            (&self.anchor, "draft corrector anchor head"),
        ] {
            map.validate(self.factor, width, role)?;
        }
        Ok(())
    }
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
    /// Proposals follow an ordered candidate path over a convolved backbone.
    DFlash2 { selector: CandidateSelector },
    /// Proposals walk the candidate lattice a corrector scores over every
    /// mask slot of the block (LiLiCorr).
    LiLiCorr { corrector: Box<CandidateCorrector> },
}

/// The separate-draft method a draft's definition implements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DraftVariant {
    DFlash,
    DSpark,
    DFlash2,
    LiLiCorr,
}

impl fmt::Display for DraftVariant {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DFlash => "DFlash",
            Self::DSpark => "DSpark",
            Self::DFlash2 => "DFlash2",
            Self::LiLiCorr => "LiLiCorr",
        })
    }
}

impl DraftMethod {
    pub fn variant(&self) -> DraftVariant {
        match self {
            Self::DFlash => DraftVariant::DFlash,
            Self::DSpark { .. } => DraftVariant::DSpark,
            Self::DFlash2 { .. } => DraftVariant::DFlash2,
            Self::LiLiCorr { .. } => DraftVariant::LiLiCorr,
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
    /// Own full-vocabulary projection; absent means the target projection is shared.
    pub output: Option<WeightDescriptor>,
    /// Explicit draft input scale, independent of the target's input transform.
    pub embedding_scale: f64,
    /// Draft layers over the target's width. Their attention reads the
    /// injected context and the fresh block.
    pub blocks: Vec<Block>,
    /// One per draft layer: how its block rows attend the block.
    pub block_attention: Vec<BlockAttention>,
    /// The convolutions wrapping every draft sublayer of a convolved
    /// backbone.
    pub convolutions: Option<BlockConvolutions>,
    pub output_norm: RmsNorm,
    /// Explicit draft projection softcap, before any Markov/selector bias.
    /// An absent value preserves raw logits even if the target has a softcap.
    pub logit_softcap: Option<f64>,
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
    /// bidirectional layers see fewer mask rows than in training. A
    /// corrector scores every mask slot together, so its block is always
    /// whole.
    pub fn block_rows(&self, proposals: u64) -> u64 {
        if matches!(self.method, DraftMethod::LiLiCorr { .. }) {
            return self.block_size;
        }
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
        if !self.embedding_scale.is_finite()
            || self.embedding_scale <= 0.0
            || !(self.embedding_scale as f32).is_finite()
            || self.embedding_scale as f32 <= 0.0
        {
            return Err(DefinitionError::new("invalid draft embedding scale"));
        }
        if self.logit_softcap.is_some_and(|cap| {
            !cap.is_finite() || cap <= 0.0 || !(cap as f32).is_finite() || cap as f32 <= 0.0
        }) {
            return Err(DefinitionError::new("invalid draft logit softcap"));
        }
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
        if let Some(output) = &self.output {
            expect_shape(
                output,
                &[decoder.vocabulary, hidden],
                "draft output projection",
            )?;
        }
        decoder::validate_blocks(
            &self.blocks,
            &Context {
                hidden,
                vocabulary: decoder.vocabulary,
                coordinate_axes,
                residual: &ResidualForm::Single,
                per_layer: None,
                shared_sources: None,
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
            DraftMethod::DFlash2 { selector } => {
                if self.layout != BlockLayout::MaskSlots {
                    return Err(DefinitionError::new(
                        "a DFlash2 draft proposes from its mask slots",
                    ));
                }
                if self.convolutions.is_none() {
                    return Err(DefinitionError::new(
                        "a DFlash2 draft convolves its layers",
                    ));
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
            DraftMethod::LiLiCorr { corrector } => {
                if self.layout != BlockLayout::MaskSlots {
                    return Err(DefinitionError::new(
                        "a LiLiCorr draft proposes from its mask slots",
                    ));
                }
                corrector.validate(hidden, decoder.vocabulary, self.max_proposals())?;
            }
        }
        if let Some(convolutions) = &self.convolutions {
            let BlockConvolutions {
                kernel,
                group,
                layers,
            } = convolutions;
            if *kernel == 0 || *group == 0 || !hidden.is_multiple_of(*group) {
                return Err(DefinitionError::new(
                    "draft convolution groups do not divide the hidden width",
                ));
            }
            if layers.len() != self.blocks.len() {
                return Err(DefinitionError::new(
                    "a convolved draft convolves every draft layer",
                ));
            }
            let coefficients = checked_product(&[2, *kernel, hidden / *group])?;
            for layer in layers {
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
        }
        Ok(())
    }
}
