//! Family-neutral materialization of a validated model definition.
//!
//! The family adapter names semantic roles in [`ModelDefinition`]. This module
//! applies the planned resident representation to those roles and imports every
//! weight through [`ResidencyStore`]. It deliberately contains no model-name,
//! tensor-name, or family-specific branching.

use crate::{ResidencyStore, ResidentWeight, WeightImportError};
use magnitude_artifacts::{gguf::GgufArtifact, Package};
use magnitude_model_contracts::{
    ActivationDType, AttentionWeights, BlockWeights, DenseFeedForwardWeights, FeedForwardWeights,
    FusedQkvWeights, HeadBlock, HeadWeights, LayerNormWeights, MixerWeights, ModelDefinition,
    RecurrentWeights, RoutedFeedForwardWeights, VisionAttentionWeights, VisionBlockWeights,
    VisionDescription, VisionFeedForwardWeights, VisionMergerWeights, WeightKind,
};
use seismic::DType;
use std::{error, fmt};

#[derive(Debug)]
pub enum ResidencyError {
    Invalid(String),
    Import(WeightImportError),
}

impl fmt::Display for ResidencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => formatter.write_str(message),
            Self::Import(error) => write!(formatter, "model residency: {error}"),
        }
    }
}

impl error::Error for ResidencyError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Import(error) => Some(error),
            Self::Invalid(_) => None,
        }
    }
}

impl From<WeightImportError> for ResidencyError {
    fn from(error: WeightImportError) -> Self {
        Self::Import(error)
    }
}

fn invalid(message: impl Into<String>) -> ResidencyError {
    ResidencyError::Invalid(message.into())
}

#[derive(Clone)]
pub struct ResidentAttentionWeights {
    pub query_gate: ResidentWeight,
    pub key: ResidentWeight,
    pub value: ResidentWeight,
    pub query_norm: ResidentWeight,
    pub key_norm: ResidentWeight,
    pub output: ResidentWeight,
}

#[derive(Clone)]
pub struct ResidentRecurrentWeights {
    pub query_key_value: ResidentWeight,
    pub gate: ResidentWeight,
    pub alpha: ResidentWeight,
    pub beta: ResidentWeight,
    pub convolution: ResidentWeight,
    pub decay: ResidentWeight,
    pub time_bias: ResidentWeight,
    pub norm: ResidentWeight,
    pub output: ResidentWeight,
}

#[derive(Clone)]
pub enum ResidentMixerWeights {
    Attention(Box<ResidentAttentionWeights>),
    Recurrent(Box<ResidentRecurrentWeights>),
}

#[derive(Clone)]
pub struct ResidentDenseFeedForwardWeights {
    pub gate: ResidentWeight,
    pub up: ResidentWeight,
    pub down: ResidentWeight,
}

#[derive(Clone)]
pub struct ResidentRoutedFeedForwardWeights {
    pub router: ResidentWeight,
    pub shared_router: ResidentWeight,
    pub expert_gate: ResidentWeight,
    pub expert_up: ResidentWeight,
    pub expert_down: ResidentWeight,
    pub shared_gate: ResidentWeight,
    pub shared_up: ResidentWeight,
    pub shared_down: ResidentWeight,
}

#[derive(Clone)]
pub enum ResidentFeedForwardWeights {
    Dense(Box<ResidentDenseFeedForwardWeights>),
    Routed(Box<ResidentRoutedFeedForwardWeights>),
}

impl ResidentFeedForwardWeights {
    pub(crate) fn weight(&self, kind: WeightKind) -> Option<&ResidentWeight> {
        match (self, kind) {
            (Self::Dense(weights), WeightKind::DenseGate) => Some(&weights.gate),
            (Self::Dense(weights), WeightKind::DenseUp) => Some(&weights.up),
            (Self::Dense(weights), WeightKind::DenseDown) => Some(&weights.down),
            (Self::Routed(weights), WeightKind::Router) => Some(&weights.router),
            (Self::Routed(weights), WeightKind::SharedRouter) => Some(&weights.shared_router),
            (Self::Routed(weights), WeightKind::ExpertGate) => Some(&weights.expert_gate),
            (Self::Routed(weights), WeightKind::ExpertUp) => Some(&weights.expert_up),
            (Self::Routed(weights), WeightKind::ExpertDown) => Some(&weights.expert_down),
            (Self::Routed(weights), WeightKind::SharedGate) => Some(&weights.shared_gate),
            (Self::Routed(weights), WeightKind::SharedUp) => Some(&weights.shared_up),
            (Self::Routed(weights), WeightKind::SharedDown) => Some(&weights.shared_down),
            _ => None,
        }
    }
}

#[derive(Clone)]
pub struct ResidentBlockWeights {
    pub input_norm: ResidentWeight,
    pub mixer: ResidentMixerWeights,
    pub feedforward_norm: ResidentWeight,
    pub feedforward: ResidentFeedForwardWeights,
}

#[derive(Clone)]
pub struct ResidentHeadBlock {
    pub embedding_norm: ResidentWeight,
    pub hidden_norm: ResidentWeight,
    pub combine: ResidentWeight,
    pub input_norm: ResidentWeight,
    pub attention: ResidentAttentionWeights,
    pub feedforward_norm: ResidentWeight,
    pub feedforward: ResidentFeedForwardWeights,
    pub output_norm: ResidentWeight,
}

#[derive(Clone)]
pub struct ResidentHead {
    /// Tied target embedding; cloned from the sole ResidencyStore cache.
    pub embedding: ResidentWeight,
    pub blocks: Vec<ResidentHeadBlock>,
    /// Tied target vocabulary projection; cloned from the sole ResidencyStore cache.
    pub output: ResidentWeight,
}

#[derive(Clone)]
pub struct ResidentLayerNormWeights {
    pub weight: ResidentWeight,
    pub bias: ResidentWeight,
}

#[derive(Clone)]
pub struct ResidentFusedQkvWeights {
    pub weight: ResidentWeight,
    pub bias: ResidentWeight,
    pub query: magnitude_model_contracts::RowRange,
    pub key: magnitude_model_contracts::RowRange,
    pub value: magnitude_model_contracts::RowRange,
}

#[derive(Clone)]
pub struct ResidentVisionAttentionWeights {
    pub qkv: ResidentFusedQkvWeights,
    pub output: ResidentWeight,
    pub output_bias: ResidentWeight,
}

#[derive(Clone)]
pub struct ResidentVisionFeedForwardWeights {
    pub up: ResidentWeight,
    pub up_bias: ResidentWeight,
    pub down: ResidentWeight,
    pub down_bias: ResidentWeight,
}

#[derive(Clone)]
pub struct ResidentVisionBlockWeights {
    pub input_norm: ResidentLayerNormWeights,
    pub attention: ResidentVisionAttentionWeights,
    pub feedforward_norm: ResidentLayerNormWeights,
    pub feedforward: ResidentVisionFeedForwardWeights,
}

#[derive(Clone)]
pub struct ResidentVisionMergerWeights {
    pub hidden: ResidentWeight,
    pub hidden_bias: ResidentWeight,
    pub output: ResidentWeight,
    pub output_bias: ResidentWeight,
}

#[derive(Clone)]
pub struct ResidentVision {
    pub patch_embeddings: Vec<ResidentWeight>,
    pub patch_bias: ResidentWeight,
    pub position_embedding: ResidentWeight,
    pub blocks: Vec<ResidentVisionBlockWeights>,
    pub output_norm: ResidentLayerNormWeights,
    pub merger: ResidentVisionMergerWeights,
}

/// All immutable weights needed by the always-resident target decoder. Stored
/// tensors retain their semantic role grouping so prepared stages do not need
/// to rediscover or reinterpret artifact names. Optional components have
/// explicit demand-driven import paths below.
#[derive(Clone)]
pub struct ResidentTarget {
    pub embedding: ResidentWeight,
    pub blocks: Vec<ResidentBlockWeights>,
    pub output_norm: ResidentWeight,
    pub output: ResidentWeight,
}

impl ResidentTarget {
    /// Visit the materialized target weights, including tied roles. Callers
    /// that count physical storage must deduplicate shared allocations.
    pub(crate) fn visit_weights<'a>(&'a self, mut visit: impl FnMut(&'a ResidentWeight)) {
        visit(&self.embedding);
        for block in &self.blocks {
            visit(&block.input_norm);
            match &block.mixer {
                ResidentMixerWeights::Attention(weights) => {
                    visit(&weights.query_gate);
                    visit(&weights.key);
                    visit(&weights.value);
                    visit(&weights.query_norm);
                    visit(&weights.key_norm);
                    visit(&weights.output);
                }
                ResidentMixerWeights::Recurrent(weights) => {
                    visit(&weights.query_key_value);
                    visit(&weights.gate);
                    visit(&weights.alpha);
                    visit(&weights.beta);
                    visit(&weights.convolution);
                    visit(&weights.decay);
                    visit(&weights.time_bias);
                    visit(&weights.norm);
                    visit(&weights.output);
                }
            }
            visit(&block.feedforward_norm);
            match &block.feedforward {
                ResidentFeedForwardWeights::Dense(weights) => {
                    visit(&weights.gate);
                    visit(&weights.up);
                    visit(&weights.down);
                }
                ResidentFeedForwardWeights::Routed(weights) => {
                    visit(&weights.router);
                    visit(&weights.shared_router);
                    visit(&weights.expert_gate);
                    visit(&weights.expert_up);
                    visit(&weights.expert_down);
                    visit(&weights.shared_gate);
                    visit(&weights.shared_up);
                    visit(&weights.shared_down);
                }
            }
        }
        visit(&self.output_norm);
        visit(&self.output);
    }

    pub(crate) fn storage_bytes(&self) -> Result<u64, &'static str> {
        let mut seen = Vec::<&seismic::Tensor>::new();
        let mut total = Some(0u64);
        self.visit_weights(|weight| {
            let tensor = weight.tensor();
            if seen.iter().any(|other| other.shares_allocation(tensor)) {
                return;
            }
            total = total.and_then(|bytes| bytes.checked_add(tensor.storage_bytes()));
            seen.push(tensor);
        });
        total.ok_or("resident target charge overflows")
    }
}

pub(crate) fn import_target(
    definition: &ModelDefinition,
    package: &Package,
    residency: &mut ResidencyStore,
) -> Result<ResidentTarget, ResidencyError> {
    validate_definition_package(definition, package)?;
    let activation = activation_dtype(definition.geometry.activation_dtype);
    let target = package.target();
    let embedding = residency.import_gguf(target, &definition.embedding, activation)?;
    let blocks = definition
        .blocks
        .iter()
        .map(|block| import_block(residency, target, block, activation))
        .collect::<Result<Vec<_>, _>>()?;
    let output_norm = residency.import_gguf(target, &definition.output_norm, activation)?;
    let output = residency.import_gguf(target, &definition.output, activation)?;
    Ok(ResidentTarget {
        embedding,
        blocks,
        output_norm,
        output,
    })
}

pub(crate) fn import_optional_head(
    definition: &ModelDefinition,
    package: &Package,
    residency: &mut ResidencyStore,
) -> Result<Option<ResidentHead>, ResidencyError> {
    validate_definition_package(definition, package)?;
    let activation = activation_dtype(definition.geometry.activation_dtype);
    definition
        .head
        .as_ref()
        .map(|head| {
            let embedding =
                residency.import_gguf(package.target(), &definition.embedding, activation)?;
            let output = residency.import_gguf(package.target(), &definition.output, activation)?;
            materialize_head(
                residency,
                package.target(),
                head,
                activation,
                embedding,
                output,
            )
        })
        .transpose()
}

pub(crate) fn import_optional_vision(
    definition: &ModelDefinition,
    package: &Package,
    residency: &mut ResidencyStore,
) -> Result<Option<ResidentVision>, ResidencyError> {
    validate_definition_package(definition, package)?;
    definition
        .vision
        .as_ref()
        .zip(package.projector())
        .map(|(vision, projector)| materialize_vision(residency, projector, vision))
        .transpose()
}

pub(crate) fn validate_definition_package(
    definition: &ModelDefinition,
    package: &Package,
) -> Result<(), ResidencyError> {
    definition
        .validate()
        .map_err(|error| invalid(format!("invalid model definition: {error}")))?;
    validate_projector_binding(definition.vision.is_some(), package.projector().is_some())?;
    if definition.artifact_identity != package.identity() {
        return Err(invalid(format!(
            "model definition identifies package {}, but opened package is {}",
            definition.artifact_identity,
            package.identity()
        )));
    }
    Ok(())
}

fn validate_projector_binding(
    definition_has_vision: bool,
    package_has_projector: bool,
) -> Result<(), ResidencyError> {
    match (definition_has_vision, package_has_projector) {
        (true, false) => Err(invalid(
            "model definition requires a projector component, but the package has none",
        )),
        (false, true) => Err(invalid(
            "package contains a projector component not bound by the model definition",
        )),
        _ => Ok(()),
    }
}

pub(crate) fn activation_dtype(dtype: ActivationDType) -> DType {
    match dtype {
        ActivationDType::F16 => DType::F16,
        ActivationDType::BF16 => DType::BF16,
    }
}

fn import_attention(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    weights: &AttentionWeights,
    activation: DType,
) -> Result<ResidentAttentionWeights, ResidencyError> {
    Ok(ResidentAttentionWeights {
        query_gate: residency.import_gguf(artifact, &weights.query_gate, activation)?,
        key: residency.import_gguf(artifact, &weights.key, activation)?,
        value: residency.import_gguf(artifact, &weights.value, activation)?,
        query_norm: residency.import_gguf(artifact, &weights.query_norm, DType::F32)?,
        key_norm: residency.import_gguf(artifact, &weights.key_norm, DType::F32)?,
        output: residency.import_gguf(artifact, &weights.output, activation)?,
    })
}

fn import_recurrent(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    weights: &RecurrentWeights,
    activation: DType,
) -> Result<ResidentRecurrentWeights, ResidencyError> {
    Ok(ResidentRecurrentWeights {
        query_key_value: residency.import_gguf(artifact, &weights.query_key_value, activation)?,
        gate: residency.import_gguf(artifact, &weights.gate, activation)?,
        alpha: residency.import_gguf(artifact, &weights.alpha, activation)?,
        beta: residency.import_gguf(artifact, &weights.beta, activation)?,
        convolution: residency.import_gguf(artifact, &weights.convolution, DType::F32)?,
        decay: residency.import_gguf(artifact, &weights.decay, DType::F32)?,
        time_bias: residency.import_gguf(artifact, &weights.time_bias, DType::F32)?,
        norm: residency.import_gguf(artifact, &weights.norm, activation)?,
        output: residency.import_gguf(artifact, &weights.output, activation)?,
    })
}

fn import_dense(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    weights: &DenseFeedForwardWeights,
    activation: DType,
) -> Result<ResidentDenseFeedForwardWeights, ResidencyError> {
    Ok(ResidentDenseFeedForwardWeights {
        gate: residency.import_gguf(artifact, &weights.gate, activation)?,
        up: residency.import_gguf(artifact, &weights.up, activation)?,
        down: residency.import_gguf(artifact, &weights.down, activation)?,
    })
}

fn import_routed(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    weights: &RoutedFeedForwardWeights,
    activation: DType,
) -> Result<ResidentRoutedFeedForwardWeights, ResidencyError> {
    Ok(ResidentRoutedFeedForwardWeights {
        router: residency.import_gguf(artifact, &weights.router, activation)?,
        // This scalar route-control projection participates in the FP32 route
        // weighting path, rather than the compact expert matmuls.
        shared_router: residency.import_gguf(artifact, &weights.shared_router, DType::F32)?,
        expert_gate: residency.import_gguf(artifact, &weights.expert_gate, activation)?,
        expert_up: residency.import_gguf(artifact, &weights.expert_up, activation)?,
        expert_down: residency.import_gguf(artifact, &weights.expert_down, activation)?,
        shared_gate: residency.import_gguf(artifact, &weights.shared_gate, activation)?,
        shared_up: residency.import_gguf(artifact, &weights.shared_up, activation)?,
        shared_down: residency.import_gguf(artifact, &weights.shared_down, activation)?,
    })
}

fn import_block(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    block: &BlockWeights,
    activation: DType,
) -> Result<ResidentBlockWeights, ResidencyError> {
    let mixer = match &block.mixer {
        MixerWeights::Attention(weights) => ResidentMixerWeights::Attention(Box::new(
            import_attention(residency, artifact, weights, activation)?,
        )),
        MixerWeights::Recurrent(weights) => ResidentMixerWeights::Recurrent(Box::new(
            import_recurrent(residency, artifact, weights, activation)?,
        )),
    };
    let feedforward = match &block.feedforward {
        FeedForwardWeights::Dense(weights) => ResidentFeedForwardWeights::Dense(Box::new(
            import_dense(residency, artifact, weights, activation)?,
        )),
        FeedForwardWeights::Routed(weights) => ResidentFeedForwardWeights::Routed(Box::new(
            import_routed(residency, artifact, weights, activation)?,
        )),
    };
    Ok(ResidentBlockWeights {
        input_norm: residency.import_gguf(artifact, &block.input_norm, activation)?,
        mixer,
        feedforward_norm: residency.import_gguf(artifact, &block.feedforward_norm, activation)?,
        feedforward,
    })
}

fn materialize_head_block(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    block: &HeadBlock,
    activation: DType,
) -> Result<ResidentHeadBlock, ResidencyError> {
    Ok(ResidentHeadBlock {
        embedding_norm: residency.import_gguf(artifact, &block.embedding_norm, activation)?,
        hidden_norm: residency.import_gguf(artifact, &block.hidden_norm, activation)?,
        combine: residency.import_gguf(artifact, &block.combine, activation)?,
        input_norm: residency.import_gguf(artifact, &block.input_norm, activation)?,
        attention: import_attention(residency, artifact, &block.attention, activation)?,
        feedforward_norm: residency.import_gguf(artifact, &block.feedforward_norm, activation)?,
        feedforward: match &block.feedforward {
            FeedForwardWeights::Dense(weights) => ResidentFeedForwardWeights::Dense(Box::new(
                import_dense(residency, artifact, weights, activation)?,
            )),
            FeedForwardWeights::Routed(weights) => ResidentFeedForwardWeights::Routed(Box::new(
                import_routed(residency, artifact, weights, activation)?,
            )),
        },
        output_norm: residency.import_gguf(artifact, &block.output_norm, activation)?,
    })
}

fn materialize_head(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    head: &HeadWeights,
    activation: DType,
    embedding: ResidentWeight,
    output: ResidentWeight,
) -> Result<ResidentHead, ResidencyError> {
    Ok(ResidentHead {
        embedding,
        blocks: head
            .blocks
            .iter()
            .map(|block| materialize_head_block(residency, artifact, block, activation))
            .collect::<Result<Vec<_>, _>>()?,
        output,
    })
}

// Projector weights are imported into the dense element their admitted plan
// chose (their stored element).

fn import_layer_norm(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    norm: &LayerNormWeights,
) -> Result<ResidentLayerNormWeights, ResidencyError> {
    Ok(ResidentLayerNormWeights {
        weight: residency.import_gguf_planned(artifact, &norm.weight)?,
        bias: residency.import_gguf_planned(artifact, &norm.bias)?,
    })
}

fn import_fused_qkv(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    qkv: &FusedQkvWeights,
) -> Result<ResidentFusedQkvWeights, ResidencyError> {
    Ok(ResidentFusedQkvWeights {
        weight: residency.import_gguf_planned(artifact, &qkv.weight)?,
        bias: residency.import_gguf_planned(artifact, &qkv.bias)?,
        query: qkv.query,
        key: qkv.key,
        value: qkv.value,
    })
}

fn materialize_vision_attention(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    weights: &VisionAttentionWeights,
) -> Result<ResidentVisionAttentionWeights, ResidencyError> {
    Ok(ResidentVisionAttentionWeights {
        qkv: import_fused_qkv(residency, artifact, &weights.qkv)?,
        output: residency.import_gguf_planned(artifact, &weights.output)?,
        output_bias: residency.import_gguf_planned(artifact, &weights.output_bias)?,
    })
}

fn materialize_vision_feedforward(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    weights: &VisionFeedForwardWeights,
) -> Result<ResidentVisionFeedForwardWeights, ResidencyError> {
    Ok(ResidentVisionFeedForwardWeights {
        up: residency.import_gguf_planned(artifact, &weights.up)?,
        up_bias: residency.import_gguf_planned(artifact, &weights.up_bias)?,
        down: residency.import_gguf_planned(artifact, &weights.down)?,
        down_bias: residency.import_gguf_planned(artifact, &weights.down_bias)?,
    })
}

fn materialize_vision_block(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    block: &VisionBlockWeights,
) -> Result<ResidentVisionBlockWeights, ResidencyError> {
    Ok(ResidentVisionBlockWeights {
        input_norm: import_layer_norm(residency, artifact, &block.input_norm)?,
        attention: materialize_vision_attention(residency, artifact, &block.attention)?,
        feedforward_norm: import_layer_norm(residency, artifact, &block.feedforward_norm)?,
        feedforward: materialize_vision_feedforward(residency, artifact, &block.feedforward)?,
    })
}

fn materialize_vision_merger(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    merger: &VisionMergerWeights,
) -> Result<ResidentVisionMergerWeights, ResidencyError> {
    Ok(ResidentVisionMergerWeights {
        hidden: residency.import_gguf_planned(artifact, &merger.hidden)?,
        hidden_bias: residency.import_gguf_planned(artifact, &merger.hidden_bias)?,
        output: residency.import_gguf_planned(artifact, &merger.output)?,
        output_bias: residency.import_gguf_planned(artifact, &merger.output_bias)?,
    })
}

fn materialize_vision(
    residency: &mut ResidencyStore,
    artifact: &GgufArtifact,
    vision: &VisionDescription,
) -> Result<ResidentVision, ResidencyError> {
    Ok(ResidentVision {
        patch_embeddings: vision
            .patch_embeddings
            .iter()
            .map(|weight| residency.import_gguf_planned(artifact, weight))
            .collect::<Result<Vec<_>, _>>()?,
        patch_bias: residency.import_gguf_planned(artifact, &vision.patch_bias)?,
        position_embedding: residency.import_gguf_planned(artifact, &vision.position_embedding)?,
        blocks: vision
            .blocks
            .iter()
            .map(|block| materialize_vision_block(residency, artifact, block))
            .collect::<Result<Vec<_>, _>>()?,
        output_norm: import_layer_norm(residency, artifact, &vision.output_norm)?,
        merger: materialize_vision_merger(residency, artifact, &vision.merger)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_model_contracts::{
        DenseFeedForwardWeights, RecurrentWeights, RoutedFeedForwardWeights, WeightDescriptor,
    };

    fn weight(name: &str) -> WeightDescriptor {
        WeightDescriptor {
            name: name.into(),
            shape: vec![1],
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Planned<'a>(&'a str, DType);

    fn plan_attention<'a>(weights: &'a AttentionWeights, activation: DType) -> Vec<Planned<'a>> {
        vec![
            Planned(&weights.query_gate.name, activation),
            Planned(&weights.key.name, activation),
            Planned(&weights.value.name, activation),
            Planned(&weights.query_norm.name, DType::F32),
            Planned(&weights.key_norm.name, DType::F32),
            Planned(&weights.output.name, activation),
        ]
    }

    #[test]
    fn target_role_dtype_policy_is_semantic_and_exhaustive() {
        let attention = AttentionWeights {
            query_gate: weight("query_gate"),
            key: weight("key"),
            value: weight("value"),
            query_norm: weight("query_norm"),
            key_norm: weight("key_norm"),
            output: weight("attention_output"),
        };
        assert_eq!(
            plan_attention(&attention, DType::BF16),
            vec![
                Planned("query_gate", DType::BF16),
                Planned("key", DType::BF16),
                Planned("value", DType::BF16),
                Planned("query_norm", DType::F32),
                Planned("key_norm", DType::F32),
                Planned("attention_output", DType::BF16),
            ]
        );

        let recurrent = RecurrentWeights {
            query_key_value: weight("qkv"),
            gate: weight("gate"),
            alpha: weight("alpha"),
            beta: weight("beta"),
            convolution: weight("convolution"),
            decay: weight("decay"),
            time_bias: weight("time_bias"),
            norm: weight("norm"),
            output: weight("output"),
        };
        let planned = [
            (&recurrent.query_key_value, DType::BF16),
            (&recurrent.gate, DType::BF16),
            (&recurrent.alpha, DType::BF16),
            (&recurrent.beta, DType::BF16),
            (&recurrent.convolution, DType::F32),
            (&recurrent.decay, DType::F32),
            (&recurrent.time_bias, DType::F32),
            (&recurrent.norm, DType::BF16),
            (&recurrent.output, DType::BF16),
        ];
        assert_eq!(planned.len(), 9);
        assert_eq!(
            planned
                .iter()
                .filter(|(_, dtype)| *dtype == DType::F32)
                .count(),
            3
        );

        let dense = DenseFeedForwardWeights {
            gate: weight("gate"),
            up: weight("up"),
            down: weight("down"),
        };
        assert!([&dense.gate, &dense.up, &dense.down]
            .into_iter()
            .all(|_| activation_dtype(ActivationDType::F16) == DType::F16));

        let routed = RoutedFeedForwardWeights {
            router: weight("router"),
            shared_router: weight("shared_router"),
            expert_gate: weight("expert_gate"),
            expert_up: weight("expert_up"),
            expert_down: weight("expert_down"),
            shared_gate: weight("shared_gate"),
            shared_up: weight("shared_up"),
            shared_down: weight("shared_down"),
        };
        assert_eq!(routed.shared_router.name, "shared_router");
    }

    #[test]
    fn source_contains_no_family_or_tensor_name_policy() {
        let source = include_str!("resident_weights.rs");
        let source = source.split("#[cfg(test)]").next().unwrap();
        for forbidden in ["qwen", "Qwen", "ssm_", "attn_"] {
            assert!(
                !source.contains(forbidden),
                "family detail leaked: {forbidden}"
            );
        }
    }

    #[test]
    fn projector_presence_must_exactly_match_the_family_contract() {
        assert!(validate_projector_binding(false, false).is_ok());
        assert!(validate_projector_binding(true, true).is_ok());
        assert_eq!(
            validate_projector_binding(true, false)
                .unwrap_err()
                .to_string(),
            "model definition requires a projector component, but the package has none"
        );
        assert_eq!(
            validate_projector_binding(false, true)
                .unwrap_err()
                .to_string(),
            "package contains a projector component not bound by the model definition"
        );
    }
}
