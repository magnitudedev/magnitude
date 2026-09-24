//! Numerical model-family variation described as data.
//!
//! This crate deliberately contains no loading, execution, batching, state,
//! generation, serving, chat, or template implementation. A family adapter
//! recognizes an artifact and assembles these contracts; common engine crates
//! consume them without depending on that family.

use magnitude_artifacts::{ImageProcessorConfig, PackageIdentity};
use serde::Serialize;
use std::{error, fmt};

mod inputs;
pub use inputs::{
    InputPreparationError, ModelInputAdapter, PreparedModelInput, PreparedVisionInput, TokenPlan,
    VisionSpatialControls,
};

/// Stable identity selected by a family adapter after artifact recognition.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(transparent)]
pub struct FamilyId(pub String);

/// Compact activation storage chosen by a model family.
///
/// Runtime-specific dtype objects are intentionally excluded from this
/// contract; an executor translates this value at its implementation seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationDType {
    F16,
    BF16,
}

impl ActivationDType {
    /// Bytes of one stored activation element.
    pub const fn bytes(self) -> usize {
        match self {
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// One family-bound tensor role, independent of its container encoding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WeightDescriptor {
    pub name: String,
    pub shape: Vec<u64>,
}

/// Structural location of a numerical weight. This identity is independent of
/// the model family's container name and therefore remains stable after family
/// metadata has been interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum WeightScope {
    Target,
    TargetBlock(u32),
    HeadBlock(u32),
    Vision,
    VisionPatch(u32),
    VisionBlock(u32),
}

/// Closed semantic role within a structural weight scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum WeightKind {
    Embedding,
    InputNorm,
    QueryGate,
    Key,
    Value,
    QueryNorm,
    KeyNorm,
    AttentionOutput,
    RecurrentQueryKeyValue,
    RecurrentGate,
    RecurrentAlpha,
    RecurrentBeta,
    RecurrentConvolution,
    RecurrentDecay,
    RecurrentTimeBias,
    RecurrentNorm,
    RecurrentOutput,
    FeedForwardNorm,
    DenseGate,
    DenseUp,
    DenseDown,
    Router,
    SharedRouter,
    ExpertGate,
    ExpertUp,
    ExpertDown,
    SharedGate,
    SharedUp,
    SharedDown,
    OutputNorm,
    Output,
    HeadEmbeddingNorm,
    HeadHiddenNorm,
    HeadCombine,
    PatchEmbedding,
    PatchBias,
    PositionEmbedding,
    InputNormWeight,
    InputNormBias,
    FeedForwardNormWeight,
    FeedForwardNormBias,
    NormWeight,
    NormBias,
    FusedQkvWeight,
    FusedQkvBias,
    AttentionOutputBias,
    FeedForwardUpBias,
    FeedForwardDownBias,
    MergerHidden,
    MergerHiddenBias,
    MergerOutput,
    MergerOutputBias,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct WeightRole {
    pub scope: WeightScope,
    pub kind: WeightKind,
}

/// How ordinary text positions enter the model's coordinate table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextCoordinateSemantics {
    /// Repeat the absolute token position on each spatial rotary axis.
    ReplicatedPosition,
}

/// Numerically relevant input rules supplied by a family adapter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct InputSemantics {
    pub text_coordinates: TextCoordinateSemantics,
    pub coordinate_axes: u8,
}

impl InputSemantics {
    pub fn validate(&self) -> Result<(), DefinitionError> {
        if self.coordinate_axes == 0 {
            return Err(DefinitionError::new(
                "model coordinates require at least one axis",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecurrentHeadMapping {
    Grouped,
    Tiled,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum RotarySemantics {
    /// Partial rotary embedding with pair axes selected by a repeating pattern.
    Interleaved {
        width: u64,
        base: f64,
        sections: Vec<u64>,
        axis_pattern: Vec<u8>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AttentionGeometry {
    pub heads: u64,
    pub kv_heads: u64,
    pub width: u64,
    pub rotary: RotarySemantics,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RecurrentGeometry {
    pub convolution_width: u64,
    pub key_heads: u64,
    pub value_heads: u64,
    pub width: u64,
    pub head_mapping: RecurrentHeadMapping,
}

impl RecurrentGeometry {
    pub fn channels(&self) -> Result<u64, DefinitionError> {
        self.key_heads
            .checked_mul(2)
            .and_then(|count| count.checked_add(self.value_heads))
            .and_then(|count| count.checked_mul(self.width))
            .ok_or_else(|| DefinitionError::new("recurrent dimensions overflow"))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum MixerGeometry {
    Attention(AttentionGeometry),
    Recurrent(RecurrentGeometry),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertGeometry {
    pub count: u64,
    pub selected: u64,
    pub intermediate: u64,
    pub shared_intermediate: u64,
    pub normalize_selected: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum FeedForwardGeometry {
    Dense { intermediate: u64 },
    Routed(ExpertGeometry),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BlockGeometry {
    pub mixer: MixerGeometry,
    pub feedforward: FeedForwardGeometry,
}

/// Decoder geometry interpreted by a family adapter.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecoderGeometry {
    pub activation_dtype: ActivationDType,
    pub hidden: u64,
    pub vocabulary: u64,
    pub context_limit: u64,
    pub epsilon: f64,
    pub blocks: Vec<BlockGeometry>,
}

impl DecoderGeometry {
    pub fn validate(&self) -> Result<(), DefinitionError> {
        let dimensions = [self.hidden, self.vocabulary, self.context_limit];
        if dimensions.contains(&0) || !self.epsilon.is_finite() || self.epsilon <= 0.0 {
            return Err(DefinitionError::new(
                "invalid decoder dimensions or numerical parameters",
            ));
        }
        if self.blocks.is_empty() {
            return Err(DefinitionError::new("decoder has no blocks"));
        }
        for block in &self.blocks {
            match &block.mixer {
                MixerGeometry::Attention(attention) => {
                    if attention.heads == 0
                        || attention.kv_heads == 0
                        || attention.width == 0
                        || !attention.heads.is_multiple_of(attention.kv_heads)
                    {
                        return Err(DefinitionError::new("invalid attention geometry"));
                    }
                    match &attention.rotary {
                        RotarySemantics::Interleaved {
                            width,
                            base,
                            sections,
                            axis_pattern,
                        } => {
                            let section_width = sections
                                .iter()
                                .try_fold(0u64, |sum, section| sum.checked_add(*section))
                                .and_then(|sum| sum.checked_mul(2));
                            if *width == 0
                                || !width.is_multiple_of(2)
                                || *width > attention.width
                                || !base.is_finite()
                                || *base <= 0.0
                                || section_width != Some(*width)
                                || axis_pattern.is_empty()
                            {
                                return Err(DefinitionError::new("invalid rotary geometry"));
                            }
                        }
                    }
                    checked_product(&[2, attention.heads, attention.width])?;
                }
                MixerGeometry::Recurrent(recurrent) => {
                    if recurrent.convolution_width < 2
                        || recurrent.key_heads == 0
                        || recurrent.value_heads == 0
                        || recurrent.width == 0
                        || !recurrent.value_heads.is_multiple_of(recurrent.key_heads)
                    {
                        return Err(DefinitionError::new("invalid recurrent geometry"));
                    }
                    recurrent.channels()?;
                    checked_product(&[recurrent.value_heads, recurrent.width])?;
                }
            }
            match &block.feedforward {
                FeedForwardGeometry::Dense { intermediate } if *intermediate == 0 => {
                    return Err(DefinitionError::new("invalid dense feedforward geometry"));
                }
                FeedForwardGeometry::Routed(experts)
                    if [
                        experts.count,
                        experts.selected,
                        experts.intermediate,
                        experts.shared_intermediate,
                    ]
                    .contains(&0)
                        || experts.selected > experts.count =>
                {
                    return Err(DefinitionError::new("invalid expert geometry"));
                }
                _ => {}
            }
        }
        Ok(())
    }
}

pub fn checked_product(dimensions: &[u64]) -> Result<u64, DefinitionError> {
    dimensions
        .iter()
        .try_fold(1u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| DefinitionError::new("model dimensions overflow"))
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AttentionWeights {
    pub query_gate: WeightDescriptor,
    pub key: WeightDescriptor,
    pub value: WeightDescriptor,
    pub query_norm: WeightDescriptor,
    pub key_norm: WeightDescriptor,
    pub output: WeightDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RecurrentWeights {
    pub query_key_value: WeightDescriptor,
    pub gate: WeightDescriptor,
    pub alpha: WeightDescriptor,
    pub beta: WeightDescriptor,
    pub convolution: WeightDescriptor,
    pub decay: WeightDescriptor,
    pub time_bias: WeightDescriptor,
    pub norm: WeightDescriptor,
    pub output: WeightDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MixerWeights {
    Attention(Box<AttentionWeights>),
    Recurrent(Box<RecurrentWeights>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DenseFeedForwardWeights {
    pub gate: WeightDescriptor,
    pub up: WeightDescriptor,
    pub down: WeightDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RoutedFeedForwardWeights {
    pub router: WeightDescriptor,
    pub shared_router: WeightDescriptor,
    pub expert_gate: WeightDescriptor,
    pub expert_up: WeightDescriptor,
    pub expert_down: WeightDescriptor,
    pub shared_gate: WeightDescriptor,
    pub shared_up: WeightDescriptor,
    pub shared_down: WeightDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum FeedForwardWeights {
    Dense(Box<DenseFeedForwardWeights>),
    Routed(Box<RoutedFeedForwardWeights>),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BlockWeights {
    pub input_norm: WeightDescriptor,
    pub mixer: MixerWeights,
    pub feedforward_norm: WeightDescriptor,
    pub feedforward: FeedForwardWeights,
}

/// One draft block following the target decoder. Its vocabulary projection is
/// the target's output descriptor and is therefore not duplicated here.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HeadBlock {
    pub embedding_norm: WeightDescriptor,
    pub hidden_norm: WeightDescriptor,
    pub combine: WeightDescriptor,
    pub input_norm: WeightDescriptor,
    pub attention: AttentionWeights,
    pub feedforward_norm: WeightDescriptor,
    pub feedforward: DenseFeedForwardWeights,
    pub output_norm: WeightDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HeadWeights {
    pub blocks: Vec<HeadBlock>,
}

impl HeadWeights {
    pub fn depth(&self) -> usize {
        self.blocks.len()
    }

    pub fn validate(&self, decoder: &DecoderGeometry) -> Result<(), DefinitionError> {
        if self.blocks.is_empty() {
            return Err(DefinitionError::new("draft head has no blocks"));
        }
        let attention = decoder
            .blocks
            .iter()
            .rev()
            .find_map(|block| match &block.mixer {
                MixerGeometry::Attention(attention) => Some(attention),
                MixerGeometry::Recurrent(_) => None,
            })
            .ok_or_else(|| DefinitionError::new("draft head requires attention geometry"))?;
        let query_rows = checked_product(&[2, attention.heads, attention.width])?;
        let key_value_rows = checked_product(&[attention.kv_heads, attention.width])?;
        let attention_rows = checked_product(&[attention.heads, attention.width])?;
        let combined_hidden = checked_product(&[2, decoder.hidden])?;

        for block in &self.blocks {
            for descriptor in [
                &block.embedding_norm,
                &block.hidden_norm,
                &block.input_norm,
                &block.feedforward_norm,
                &block.output_norm,
            ] {
                expect_shape(descriptor, &[decoder.hidden], "draft head norm")?;
            }
            expect_shape(
                &block.combine,
                &[decoder.hidden, combined_hidden],
                "draft head input projection",
            )?;
            let weights = &block.attention;
            expect_shape(
                &weights.query_gate,
                &[query_rows, decoder.hidden],
                "draft head query/gate projection",
            )?;
            expect_shape(
                &weights.key,
                &[key_value_rows, decoder.hidden],
                "draft head key projection",
            )?;
            expect_shape(
                &weights.value,
                &[key_value_rows, decoder.hidden],
                "draft head value projection",
            )?;
            expect_shape(
                &weights.query_norm,
                &[attention.width],
                "draft head query norm",
            )?;
            expect_shape(&weights.key_norm, &[attention.width], "draft head key norm")?;
            expect_shape(
                &weights.output,
                &[decoder.hidden, attention_rows],
                "draft head attention output",
            )?;

            let intermediate = block
                .feedforward
                .gate
                .shape
                .first()
                .copied()
                .filter(|dimension| *dimension > 0)
                .ok_or_else(|| DefinitionError::new("invalid draft head feed-forward shape"))?;
            expect_shape(
                &block.feedforward.gate,
                &[intermediate, decoder.hidden],
                "draft head feed-forward gate",
            )?;
            expect_shape(
                &block.feedforward.up,
                &[intermediate, decoder.hidden],
                "draft head feed-forward up projection",
            )?;
            expect_shape(
                &block.feedforward.down,
                &[decoder.hidden, intermediate],
                "draft head feed-forward down projection",
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionActivation {
    GeluTanh,
    GeluErf,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VisionGeometry {
    /// Compact dtype used by the vision tower's weights and activations.
    ///
    /// This belongs in the family contract rather than in the common importer:
    /// a different projector family may make a different numerical choice.
    pub activation_dtype: ActivationDType,
    pub depth: u64,
    pub hidden: u64,
    pub intermediate: u64,
    pub heads: u64,
    pub patch: u64,
    pub merge: u64,
    pub image_size: u64,
    pub output_hidden: u64,
    pub epsilon: f64,
    pub table_side: u64,
    pub temporal_patch: u64,
    pub channels: u64,
    pub block_activation: VisionActivation,
    pub merger_activation: VisionActivation,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VisionPreprocessing {
    pub processor: String,
    pub mean: [f64; 3],
    pub std: [f64; 3],
    pub min_pixels: u64,
    pub max_pixels: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RowRange {
    pub start: u64,
    pub rows: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LayerNormWeights {
    pub weight: WeightDescriptor,
    pub bias: WeightDescriptor,
}

/// A single stored QKV projection plus the logical row ranges used when it is
/// imported as three semantic projections.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FusedQkvWeights {
    pub weight: WeightDescriptor,
    pub bias: WeightDescriptor,
    pub query: RowRange,
    pub key: RowRange,
    pub value: RowRange,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VisionAttentionWeights {
    pub qkv: FusedQkvWeights,
    pub output: WeightDescriptor,
    pub output_bias: WeightDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VisionFeedForwardWeights {
    pub up: WeightDescriptor,
    pub up_bias: WeightDescriptor,
    pub down: WeightDescriptor,
    pub down_bias: WeightDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VisionBlockWeights {
    pub input_norm: LayerNormWeights,
    pub attention: VisionAttentionWeights,
    pub feedforward_norm: LayerNormWeights,
    pub feedforward: VisionFeedForwardWeights,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VisionMergerWeights {
    pub hidden: WeightDescriptor,
    pub hidden_bias: WeightDescriptor,
    pub output: WeightDescriptor,
    pub output_bias: WeightDescriptor,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VisionDescription {
    pub geometry: VisionGeometry,
    pub preprocessing: VisionPreprocessing,
    pub patch_embeddings: Vec<WeightDescriptor>,
    pub patch_bias: WeightDescriptor,
    pub position_embedding: WeightDescriptor,
    pub blocks: Vec<VisionBlockWeights>,
    pub output_norm: LayerNormWeights,
    pub merger: VisionMergerWeights,
}

/// Shared numerical description assembled by a model-family adapter.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelDefinition {
    pub family: FamilyId,
    pub artifact_identity: PackageIdentity,
    pub inputs: InputSemantics,
    pub geometry: DecoderGeometry,
    pub embedding: WeightDescriptor,
    pub output_norm: WeightDescriptor,
    pub output: WeightDescriptor,
    pub blocks: Vec<BlockWeights>,
    pub head: Option<HeadWeights>,
    pub vision: Option<VisionDescription>,
}

impl ModelDefinition {
    pub fn validate(&self) -> Result<(), DefinitionError> {
        if self.family.0.is_empty() {
            return Err(DefinitionError::new("model family identity is empty"));
        }
        self.inputs.validate()?;
        self.geometry.validate()?;
        for block in &self.geometry.blocks {
            let MixerGeometry::Attention(AttentionGeometry {
                rotary: RotarySemantics::Interleaved { axis_pattern, .. },
                ..
            }) = &block.mixer
            else {
                continue;
            };
            if axis_pattern
                .iter()
                .any(|axis| *axis >= self.inputs.coordinate_axes)
            {
                return Err(DefinitionError::new(
                    "rotary axis exceeds the declared input coordinates",
                ));
            }
        }
        if self.blocks.len() != self.geometry.blocks.len() {
            return Err(DefinitionError::new(
                "model definition does not bind every decoder block",
            ));
        }
        for (geometry, weights) in self.geometry.blocks.iter().zip(&self.blocks) {
            if !matches!(
                (&geometry.mixer, &weights.mixer),
                (MixerGeometry::Attention(_), MixerWeights::Attention(_))
                    | (MixerGeometry::Recurrent(_), MixerWeights::Recurrent(_))
            ) {
                return Err(DefinitionError::new(
                    "mixer geometry and semantic weight roles disagree",
                ));
            }
            if !matches!(
                (&geometry.feedforward, &weights.feedforward),
                (
                    FeedForwardGeometry::Dense { .. },
                    FeedForwardWeights::Dense(_)
                ) | (
                    FeedForwardGeometry::Routed(_),
                    FeedForwardWeights::Routed(_)
                )
            ) {
                return Err(DefinitionError::new(
                    "feed-forward geometry and semantic weight roles disagree",
                ));
            }
        }
        if let Some(head) = &self.head {
            head.validate(&self.geometry)?;
        }
        if let Some(vision) = &self.vision {
            vision.validate()?;
        }
        Ok(())
    }
}

impl VisionDescription {
    pub fn image_processor_config(&self) -> Result<ImageProcessorConfig, DefinitionError> {
        self.validate()?;
        let to_f32 = |values: [f64; 3]| values.map(|value| value as f32);
        let patch = usize::try_from(self.geometry.patch)
            .map_err(|_| DefinitionError::new("image patch size exceeds host domain"))?;
        let merge = usize::try_from(self.geometry.merge)
            .map_err(|_| DefinitionError::new("image merge size exceeds host domain"))?;
        let min_pixels = usize::try_from(self.preprocessing.min_pixels)
            .map_err(|_| DefinitionError::new("image minimum pixels exceed host domain"))?;
        let max_pixels = usize::try_from(self.preprocessing.max_pixels)
            .map_err(|_| DefinitionError::new("image maximum pixels exceed host domain"))?;
        let temporal_patch = usize::try_from(self.geometry.temporal_patch)
            .map_err(|_| DefinitionError::new("temporal patch size exceeds host domain"))?;
        let config = ImageProcessorConfig {
            kind: self.preprocessing.processor.clone(),
            patch,
            merge,
            temporal_patch,
            min_pixels,
            max_pixels,
            mean: to_f32(self.preprocessing.mean),
            std: to_f32(self.preprocessing.std),
        };
        config
            .validate()
            .map_err(|error| DefinitionError::new(error))?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), DefinitionError> {
        let g = &self.geometry;
        if [
            g.depth,
            g.hidden,
            g.intermediate,
            g.heads,
            g.patch,
            g.merge,
            g.image_size,
            g.output_hidden,
            g.table_side,
            g.temporal_patch,
            g.channels,
        ]
        .contains(&0)
            || !g.hidden.is_multiple_of(g.heads)
            || !g.epsilon.is_finite()
            || g.epsilon <= 0.0
            || g.channels != self.preprocessing.mean.len() as u64
            || self.preprocessing.processor.is_empty()
            || g.image_size.checked_rem(g.patch) != Some(0)
            || g.image_size / g.patch != g.table_side
            || self.blocks.len() as u64 != g.depth
            || self.patch_embeddings.len() as u64 != g.temporal_patch
            || self
                .preprocessing
                .mean
                .iter()
                .any(|value| !value.is_finite())
            || self
                .preprocessing
                .std
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
            || self.preprocessing.min_pixels == 0
            || self.preprocessing.min_pixels > self.preprocessing.max_pixels
        {
            return Err(DefinitionError::new("invalid vision description"));
        }
        let fused = g
            .hidden
            .checked_mul(3)
            .ok_or_else(|| DefinitionError::new("vision dimensions overflow"))?;
        let table_rows = checked_product(&[g.table_side, g.table_side])?;
        let merger_hidden = checked_product(&[g.hidden, g.merge, g.merge])?;
        for patch in &self.patch_embeddings {
            expect_shape(
                patch,
                &[g.hidden, g.channels, g.patch, g.patch],
                "vision patch embedding",
            )?;
        }
        expect_shape(&self.patch_bias, &[g.hidden], "vision patch bias")?;
        expect_shape(
            &self.position_embedding,
            &[table_rows, g.hidden],
            "vision position embedding",
        )?;
        for block in &self.blocks {
            validate_norm(&block.input_norm, g.hidden, "vision input norm")?;
            validate_norm(
                &block.feedforward_norm,
                g.hidden,
                "vision feed-forward norm",
            )?;
            let qkv = &block.attention.qkv;
            if qkv.weight.shape != [fused, g.hidden]
                || qkv.bias.shape != [fused]
                || qkv.query
                    != (RowRange {
                        start: 0,
                        rows: g.hidden,
                    })
                || qkv.key
                    != (RowRange {
                        start: g.hidden,
                        rows: g.hidden,
                    })
                || qkv.value
                    != (RowRange {
                        start: g.hidden * 2,
                        rows: g.hidden,
                    })
            {
                return Err(DefinitionError::new("invalid fused vision QKV split"));
            }
            expect_shape(
                &block.attention.output,
                &[g.hidden, g.hidden],
                "vision attention output",
            )?;
            expect_shape(
                &block.attention.output_bias,
                &[g.hidden],
                "vision attention output bias",
            )?;
            expect_shape(
                &block.feedforward.up,
                &[g.intermediate, g.hidden],
                "vision feed-forward up projection",
            )?;
            expect_shape(
                &block.feedforward.up_bias,
                &[g.intermediate],
                "vision feed-forward up bias",
            )?;
            expect_shape(
                &block.feedforward.down,
                &[g.hidden, g.intermediate],
                "vision feed-forward down projection",
            )?;
            expect_shape(
                &block.feedforward.down_bias,
                &[g.hidden],
                "vision feed-forward down bias",
            )?;
        }
        validate_norm(&self.output_norm, g.hidden, "vision output norm")?;
        expect_shape(
            &self.merger.hidden,
            &[merger_hidden, merger_hidden],
            "vision merger hidden projection",
        )?;
        expect_shape(
            &self.merger.hidden_bias,
            &[merger_hidden],
            "vision merger hidden bias",
        )?;
        expect_shape(
            &self.merger.output,
            &[g.output_hidden, merger_hidden],
            "vision merger output projection",
        )?;
        expect_shape(
            &self.merger.output_bias,
            &[g.output_hidden],
            "vision merger output bias",
        )?;
        Ok(())
    }
}

fn validate_norm(norm: &LayerNormWeights, hidden: u64, role: &str) -> Result<(), DefinitionError> {
    expect_shape(&norm.weight, &[hidden], role)?;
    expect_shape(&norm.bias, &[hidden], role)
}

fn expect_shape(
    descriptor: &WeightDescriptor,
    expected: &[u64],
    role: &str,
) -> Result<(), DefinitionError> {
    if descriptor.shape != expected {
        return Err(DefinitionError::new(format!(
            "invalid {role} shape for {:?}",
            descriptor.name
        )));
    }
    checked_product(expected)?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefinitionError(String);

impl DefinitionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for DefinitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl error::Error for DefinitionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_artifacts::ArtifactIdentity;

    #[test]
    fn direct_dependencies_preserve_the_family_contract_boundary() {
        let manifest = include_str!("../Cargo.toml");
        let forbidden = [
            "seismic",
            "magnitude-model-executor",
            "magnitude-generation",
            "magnitude-service",
            "magnitude-chat",
            "magnitude-templates",
        ];
        for dependency in forbidden {
            assert!(
                !manifest.lines().any(|line| {
                    let line = line.trim_start();
                    line.starts_with(dependency) && line[dependency.len()..].starts_with([' ', '='])
                }),
                "model-contracts must not depend on {dependency}"
            );
        }
    }

    fn weight(name: &str) -> WeightDescriptor {
        WeightDescriptor {
            name: name.into(),
            shape: vec![1],
        }
    }

    fn definition() -> ModelDefinition {
        let recurrent = RecurrentGeometry {
            convolution_width: 2,
            key_heads: 1,
            value_heads: 1,
            width: 1,
            head_mapping: RecurrentHeadMapping::Grouped,
        };
        ModelDefinition {
            family: FamilyId("fixture".into()),
            artifact_identity: PackageIdentity {
                target: ArtifactIdentity([0; 32]),
                projector: None,
            },
            inputs: InputSemantics {
                text_coordinates: TextCoordinateSemantics::ReplicatedPosition,
                coordinate_axes: 1,
            },
            geometry: DecoderGeometry {
                activation_dtype: ActivationDType::BF16,
                hidden: 1,
                vocabulary: 1,
                context_limit: 1,
                epsilon: 1e-5,
                blocks: vec![BlockGeometry {
                    mixer: MixerGeometry::Recurrent(recurrent),
                    feedforward: FeedForwardGeometry::Dense { intermediate: 1 },
                }],
            },
            embedding: weight("embedding"),
            output_norm: weight("output_norm"),
            output: weight("output"),
            blocks: vec![BlockWeights {
                input_norm: weight("input_norm"),
                mixer: MixerWeights::Recurrent(Box::new(RecurrentWeights {
                    query_key_value: weight("qkv"),
                    gate: weight("gate"),
                    alpha: weight("alpha"),
                    beta: weight("beta"),
                    convolution: weight("convolution"),
                    decay: weight("decay"),
                    time_bias: weight("time_bias"),
                    norm: weight("norm"),
                    output: weight("mixer_output"),
                })),
                feedforward_norm: weight("feedforward_norm"),
                feedforward: FeedForwardWeights::Dense(Box::new(DenseFeedForwardWeights {
                    gate: weight("ff_gate"),
                    up: weight("ff_up"),
                    down: weight("ff_down"),
                })),
            }],
            head: None,
            vision: None,
        }
    }

    #[test]
    fn definition_rejects_component_role_mismatches() {
        let mut model = definition();
        assert!(model.validate().is_ok());
        model.geometry.blocks[0].mixer = MixerGeometry::Attention(AttentionGeometry {
            heads: 1,
            kv_heads: 1,
            width: 2,
            rotary: RotarySemantics::Interleaved {
                width: 2,
                base: 10_000.0,
                sections: vec![1],
                axis_pattern: vec![0],
            },
        });
        assert_eq!(
            model.validate().unwrap_err().to_string(),
            "mixer geometry and semantic weight roles disagree"
        );
    }
}
