//! Qwen 3.5 dense and routed architecture contracts, ported from V3.
pub mod baseline;
pub mod dense;
pub mod decoder;
pub mod gguf;
pub mod mlx;
pub mod loading;
pub mod program;
pub mod preparation;
pub mod service;
pub mod session;
pub mod vision;
pub mod vision_runtime;
pub mod inputs;
use crate::weights::{
    descriptor::{ArtifactIdentity, WeightDescriptor},
    Error,
};
use seismic_lang::types::DType;
use serde::Serialize;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MixerKind {
    Attention,
    Recurrent,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HeadMapping {
    Grouped,
    Tiled,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExpertGeometry {
    pub count: u64,
    pub selected: u64,
    pub intermediate: u64,
    pub shared_intermediate: u64,
    pub normalize_selected: bool,
}
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Geometry {
    #[serde(serialize_with = "serialize_dtype")]
    pub activation_dtype: DType,
    pub hidden: u64,
    pub intermediate: u64,
    pub vocabulary: u64,
    pub context_limit: u64,
    pub layers: Vec<MixerKind>,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub attention_width: u64,
    pub rotary_width: u64,
    pub rotary_base: f64,
    pub rotary_sections: [u64; 4],
    pub epsilon: f64,
    pub convolution_width: u64,
    pub recurrent_key_heads: u64,
    pub recurrent_value_heads: u64,
    pub recurrent_width: u64,
    pub recurrent_head_mapping: HeadMapping,
    pub experts: Option<ExpertGeometry>,
}
fn serialize_dtype<S: serde::Serializer>(d: &DType, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(d.name())
}
fn invalid(s: impl Into<String>) -> Error {
    Error::Invalid(s.into())
}
impl Geometry {
    pub fn validate(&self) -> Result<(), Error> {
        if !matches!(self.activation_dtype, DType::F16 | DType::BF16) {
            return Err(invalid(
                "Qwen activations require a qualified 16-bit floating dtype",
            ));
        }
        let dimensions = [
            self.hidden,
            self.intermediate,
            self.vocabulary,
            self.context_limit,
            self.attention_heads,
            self.kv_heads,
            self.attention_width,
            self.rotary_width,
            self.convolution_width,
            self.recurrent_key_heads,
            self.recurrent_value_heads,
            self.recurrent_width,
        ];
        if dimensions.contains(&0)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.rotary_base.is_finite()
            || self.rotary_base <= 0.0
        {
            return Err(invalid("invalid Qwen dimensions or numerical parameters"));
        }
        if self.layers.is_empty()
            || !self.attention_heads.is_multiple_of(self.kv_heads)
            || !self
                .recurrent_value_heads
                .is_multiple_of(self.recurrent_key_heads)
        {
            return Err(invalid("invalid Qwen layer/head geometry"));
        }
        let sections = self
            .rotary_sections
            .iter()
            .try_fold(0u64, |a, n| a.checked_add(*n))
            .and_then(|s| s.checked_mul(2));
        if !self.rotary_width.is_multiple_of(2)
            || self.rotary_width > self.attention_width
            || sections != Some(self.rotary_width)
            || self.rotary_sections[3] != 0
        {
            return Err(invalid("invalid Qwen rotary width or sections"));
        }
        if self.convolution_width < 2 {
            return Err(invalid(
                "Qwen recurrent convolution requires retained history",
            ));
        }
        if let Some(e) = &self.experts {
            if [e.count, e.selected, e.intermediate, e.shared_intermediate].contains(&0)
                || e.selected > e.count
            {
                return Err(invalid("invalid Qwen expert geometry"));
            }
        }
        self.recurrent_channels()?;
        product(&[2, self.attention_heads, self.attention_width])?;
        product(&[self.recurrent_value_heads, self.recurrent_width])?;
        Ok(())
    }
    pub fn recurrent_channels(&self) -> Result<u64, Error> {
        self.recurrent_key_heads
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.recurrent_value_heads))
            .and_then(|n| n.checked_mul(self.recurrent_width))
            .ok_or_else(|| invalid("Qwen recurrent dimensions overflow"))
    }
}
fn product(dims: &[u64]) -> Result<u64, Error> {
    dims.iter()
        .try_fold(1u64, |a, n| a.checked_mul(*n))
        .ok_or_else(|| invalid("Qwen dimensions overflow"))
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
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Description {
    pub artifact_identity: ArtifactIdentity,
    pub geometry: Geometry,
    pub embedding: WeightDescriptor,
    pub output_norm: WeightDescriptor,
    pub output: WeightDescriptor,
    pub blocks: Vec<BlockWeights>,
}
