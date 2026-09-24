//! Ordered checked-entry topology. A slot is the requirement and the recipe;
//! there is no second kernel-key set or broad semantic coverage class here.

use super::weights::activation_dtype;
use super::{
    planned_element, AttentionBinding, AttentionShape, DenseBinding, EmbeddingBinding, FeaturesBinding,
    HeadBinding, ReadoutBinding, RecurrentBinding, RoutedBinding, VisionBlockBinding,
    VisionMergerBinding, VisionPatchBinding, WeightPlan,
};
use crate::error::PlanError;
use magnitude_model_contracts::{
    AttentionGeometry, FeedForwardGeometry, MixerGeometry, ModelDefinition, RotarySemantics,
};
use magnitude_model_contracts::{FeedForwardWeights, MixerWeights, WeightKind, WeightScope};
use seismic::{DType, Element};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImportProgramSlot {
    Dense { source: DType, resident: DType },
    Repack { source: Element, resident: Element },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MixerProgramSlot {
    Attention(AttentionBinding),
    Recurrent(RecurrentBinding),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FeedForwardProgramSlot {
    Dense(DenseBinding),
    Routed(RoutedBinding),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TargetBlockProgramSlot {
    mixer: MixerProgramSlot,
    feed_forward: FeedForwardProgramSlot,
}

impl TargetBlockProgramSlot {
    pub fn new(mixer: MixerProgramSlot, feed_forward: FeedForwardProgramSlot) -> Self {
        Self {
            mixer,
            feed_forward,
        }
    }

    pub fn mixer(&self) -> MixerProgramSlot {
        self.mixer
    }

    pub fn feed_forward(&self) -> FeedForwardProgramSlot {
        self.feed_forward
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetProgramPlan {
    embedding: EmbeddingBinding,
    blocks: Vec<TargetBlockProgramSlot>,
    readout: ReadoutBinding,
    features: Option<FeaturesBinding>,
    selection: DType,
}

impl TargetProgramPlan {
    pub fn new(
        embedding: EmbeddingBinding,
        blocks: Vec<TargetBlockProgramSlot>,
        readout: ReadoutBinding,
        features: Option<FeaturesBinding>,
        selection: DType,
    ) -> Self {
        Self {
            embedding,
            blocks,
            readout,
            features,
            selection,
        }
    }

    pub fn embedding(&self) -> EmbeddingBinding {
        self.embedding
    }

    pub fn blocks(&self) -> &[TargetBlockProgramSlot] {
        &self.blocks
    }

    pub fn readout(&self) -> ReadoutBinding {
        self.readout
    }

    pub fn features(&self) -> Option<FeaturesBinding> {
        self.features
    }

    pub fn selection(&self) -> DType {
        self.selection
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadProgramPlan {
    blocks: Vec<HeadBinding>,
}

impl HeadProgramPlan {
    pub fn new(blocks: Vec<HeadBinding>) -> Self {
        Self { blocks }
    }

    pub fn blocks(&self) -> &[HeadBinding] {
        &self.blocks
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionProgramPlan {
    patch: VisionPatchBinding,
    blocks: Vec<VisionBlockBinding>,
    merger: VisionMergerBinding,
}

impl VisionProgramPlan {
    pub fn new(
        patch: VisionPatchBinding,
        blocks: Vec<VisionBlockBinding>,
        merger: VisionMergerBinding,
    ) -> Self {
        Self {
            patch,
            blocks,
            merger,
        }
    }

    pub fn patch(&self) -> VisionPatchBinding {
        self.patch
    }

    pub fn blocks(&self) -> &[VisionBlockBinding] {
        &self.blocks
    }

    pub fn merger(&self) -> VisionMergerBinding {
        self.merger
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateProgramPlan {
    copies: Vec<Element>,
}

impl StateProgramPlan {
    pub fn new(copies: Vec<Element>) -> Self {
        Self { copies }
    }

    pub fn copies(&self) -> &[Element] {
        &self.copies
    }
}

/// Exact program slots in execution order. The private constructor checks
/// family topology before any program factory can attest callable slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgramPlan {
    imports: Vec<ImportProgramSlot>,
    target: TargetProgramPlan,
    head: Option<HeadProgramPlan>,
    vision: Option<VisionProgramPlan>,
    state: StateProgramPlan,
}

impl ProgramPlan {
    pub(crate) fn new(
        definition: &ModelDefinition,
        imports: Vec<ImportProgramSlot>,
        target: TargetProgramPlan,
        head: Option<HeadProgramPlan>,
        vision: Option<VisionProgramPlan>,
        state: StateProgramPlan,
    ) -> Result<Self, PlanError> {
        if target.blocks.len() != definition.geometry.blocks.len()
            || target
                .blocks
                .iter()
                .zip(&definition.geometry.blocks)
                .any(|(slot, block)| {
                    !matches!(
                        (&slot.mixer, &block.mixer),
                        (MixerProgramSlot::Attention(_), MixerGeometry::Attention(_))
                            | (MixerProgramSlot::Recurrent(_), MixerGeometry::Recurrent(_))
                    ) || !matches!(
                        (&slot.feed_forward, &block.feedforward),
                        (
                            FeedForwardProgramSlot::Dense(_),
                            FeedForwardGeometry::Dense { .. }
                        ) | (
                            FeedForwardProgramSlot::Routed(_),
                            FeedForwardGeometry::Routed(_)
                        )
                    )
                })
            || head.as_ref().is_some_and(|head| {
                definition
                    .head
                    .as_ref()
                    .map(|description| description.blocks.len())
                    != Some(head.blocks.len())
            })
            || vision.as_ref().is_some_and(|vision| {
                definition
                    .vision
                    .as_ref()
                    .map(|description| description.blocks.len())
                    != Some(vision.blocks.len())
            })
        {
            return Err(PlanError::Topology(
                "program slots disagree with model topology",
            ));
        }
        Ok(Self {
            imports,
            target,
            head,
            vision,
            state,
        })
    }

    pub fn imports(&self) -> &[ImportProgramSlot] {
        &self.imports
    }

    pub fn target(&self) -> &TargetProgramPlan {
        &self.target
    }

    pub fn head(&self) -> Option<&HeadProgramPlan> {
        self.head.as_ref()
    }

    pub fn vision(&self) -> Option<&VisionProgramPlan> {
        self.vision.as_ref()
    }

    pub fn state(&self) -> &StateProgramPlan {
        &self.state
    }

    /// Recurrent prefix repair is a state transaction executed by the exact
    /// already-planned target replay slots. This projection adds no second
    /// independently attested kernel topology.
    pub fn recurrent_repair_replay(&self) -> Option<&TargetProgramPlan> {
        self.target
            .blocks
            .iter()
            .any(|block| matches!(block.mixer, MixerProgramSlot::Recurrent(_)))
            .then_some(&self.target)
    }
}

pub(super) fn derive_program_plan(
    definition: &ModelDefinition,
    target: &[WeightPlan],
    head: Option<&[WeightPlan]>,
    vision: Option<&[WeightPlan]>,
) -> Result<ProgramPlan, PlanError> {
    let lookup = |weights: &[WeightPlan], scope, kind| {
        planned_element(weights, scope, kind).map_err(PlanError::InvalidDefinition)
    };
    let mut seen_imports = HashSet::new();
    let mut imports = Vec::new();
    for weight in target
        .iter()
        .chain(head.unwrap_or_default())
        .chain(vision.unwrap_or_default())
    {
        let slot = match (weight.source.dtype(), weight.resident.dtype()) {
            (Some(source), Some(resident)) => ImportProgramSlot::Dense { source, resident },
            _ => ImportProgramSlot::Repack {
                source: weight.source,
                resident: weight.resident,
            },
        };
        if seen_imports.insert(slot) {
            imports.push(slot);
        }
    }
    let activation = activation_dtype(definition.geometry.activation_dtype);
    let active_element = Element::dense(activation);
    let embedding = EmbeddingBinding {
        table: lookup(target, WeightScope::Target, WeightKind::Embedding)?,
        activation: active_element,
    };
    let mut blocks = Vec::with_capacity(definition.blocks.len());
    for (index, block) in definition.blocks.iter().enumerate() {
        let scope = WeightScope::TargetBlock(
            u32::try_from(index)
                .map_err(|_| PlanError::Arithmetic("target block index exceeds u32"))?,
        );
        let mixer = match &block.mixer {
            MixerWeights::Attention(_) => MixerProgramSlot::Attention(AttentionBinding {
                shape: match definition.geometry.blocks.get(index).map(|block| &block.mixer) {
                    Some(MixerGeometry::Attention(geometry)) => {
                        attention_shape(definition.geometry.hidden, geometry)?
                    }
                    _ => {
                        return Err(PlanError::Topology(
                            "attention program slot geometry differs",
                        ))
                    }
                },
                norm: lookup(target, scope, WeightKind::InputNorm)?,
                query_gate: lookup(target, scope, WeightKind::QueryGate)?,
                key: lookup(target, scope, WeightKind::Key)?,
                value: lookup(target, scope, WeightKind::Value)?,
                output: lookup(target, scope, WeightKind::AttentionOutput)?,
                activation: active_element,
            }),
            MixerWeights::Recurrent(_) => {
                let Some(MixerGeometry::Recurrent(geometry)) = definition
                    .geometry
                    .blocks
                    .get(index)
                    .map(|block| &block.mixer)
                else {
                    return Err(PlanError::Topology(
                        "recurrent program slot geometry differs",
                    ));
                };
                MixerProgramSlot::Recurrent(RecurrentBinding {
                    key_heads: geometry.key_heads,
                    value_heads: geometry.value_heads,
                    width: geometry.width,
                    convolution_width: geometry.convolution_width,
                    norm: lookup(target, scope, WeightKind::InputNorm)?,
                    qkv: lookup(target, scope, WeightKind::RecurrentQueryKeyValue)?,
                    gate: lookup(target, scope, WeightKind::RecurrentGate)?,
                    alpha: lookup(target, scope, WeightKind::RecurrentAlpha)?,
                    beta: lookup(target, scope, WeightKind::RecurrentBeta)?,
                    recurrent_norm: lookup(target, scope, WeightKind::RecurrentNorm)?,
                    output: lookup(target, scope, WeightKind::RecurrentOutput)?,
                    activation: active_element,
                })
            }
        };
        let feed_forward = match &block.feedforward {
            FeedForwardWeights::Dense(_) => FeedForwardProgramSlot::Dense(DenseBinding {
                norm: lookup(target, scope, WeightKind::FeedForwardNorm)?,
                gate: lookup(target, scope, WeightKind::DenseGate)?,
                up: lookup(target, scope, WeightKind::DenseUp)?,
                down: lookup(target, scope, WeightKind::DenseDown)?,
                activation: active_element,
            }),
            FeedForwardWeights::Routed(_) => {
                let Some(FeedForwardGeometry::Routed(geometry)) = definition
                    .geometry
                    .blocks
                    .get(index)
                    .map(|block| &block.feedforward)
                else {
                    return Err(PlanError::Topology("routed program slot geometry differs"));
                };
                FeedForwardProgramSlot::Routed(RoutedBinding {
                hidden: definition.geometry.hidden,
                experts: geometry.count,
                selected: geometry.selected,
                features: geometry.intermediate,
                shared: geometry.shared_intermediate,
                norm: lookup(target, scope, WeightKind::FeedForwardNorm)?,
                router: lookup(target, scope, WeightKind::Router)?,
                expert_gate: lookup(target, scope, WeightKind::ExpertGate)?,
                expert_up: lookup(target, scope, WeightKind::ExpertUp)?,
                expert_down: lookup(target, scope, WeightKind::ExpertDown)?,
                shared_gate: lookup(target, scope, WeightKind::SharedGate)?,
                shared_up: lookup(target, scope, WeightKind::SharedUp)?,
                shared_down: lookup(target, scope, WeightKind::SharedDown)?,
                activation: active_element,
            })
            }
        };
        blocks.push(TargetBlockProgramSlot::new(mixer, feed_forward));
    }
    let readout = ReadoutBinding {
        norm: lookup(target, WeightScope::Target, WeightKind::OutputNorm)?,
        weight: lookup(target, WeightScope::Target, WeightKind::Output)?,
        activation: active_element,
    };
    let target_program = TargetProgramPlan::new(
        embedding,
        blocks,
        readout,
        Some(FeaturesBinding {
            norm: readout.norm,
            activation: active_element,
        }),
        activation,
    );
    let head_program = head
        .map(|weights| {
            let mut slots =
                Vec::with_capacity(definition.head.as_ref().map_or(0, |head| head.depth()));
            for index in 0..definition.head.as_ref().map_or(0, |head| head.depth()) {
                let scope = WeightScope::HeadBlock(
                    u32::try_from(index)
                        .map_err(|_| PlanError::Arithmetic("head block index exceeds u32"))?,
                );
                slots.push(HeadBinding {
                    attention_shape: definition
                        .geometry
                        .blocks
                        .iter()
                        .find_map(|block| match &block.mixer {
                            MixerGeometry::Attention(geometry) => Some(geometry),
                            MixerGeometry::Recurrent(_) => None,
                        })
                        .ok_or(PlanError::Topology("head requires target attention geometry"))
                        .and_then(|geometry| attention_shape(definition.geometry.hidden, geometry))?,
                    embedding_table: lookup(target, WeightScope::Target, WeightKind::Embedding)?,
                    embedding_norm: lookup(weights, scope, WeightKind::HeadEmbeddingNorm)?,
                    hidden_norm: lookup(weights, scope, WeightKind::HeadHiddenNorm)?,
                    combine: lookup(weights, scope, WeightKind::HeadCombine)?,
                    input_norm: lookup(weights, scope, WeightKind::InputNorm)?,
                    query_gate: lookup(weights, scope, WeightKind::QueryGate)?,
                    key: lookup(weights, scope, WeightKind::Key)?,
                    value: lookup(weights, scope, WeightKind::Value)?,
                    attention_output: lookup(weights, scope, WeightKind::AttentionOutput)?,
                    feedforward_norm: lookup(weights, scope, WeightKind::FeedForwardNorm)?,
                    gate: lookup(weights, scope, WeightKind::DenseGate)?,
                    up: lookup(weights, scope, WeightKind::DenseUp)?,
                    down: lookup(weights, scope, WeightKind::DenseDown)?,
                    output_norm: lookup(weights, scope, WeightKind::OutputNorm)?,
                    projection: lookup(target, WeightScope::Target, WeightKind::Output)?,
                    activation: active_element,
                });
            }
            Ok::<_, PlanError>(HeadProgramPlan::new(slots))
        })
        .transpose()?;
    let vision_program = match (definition.vision.as_ref(), vision) {
        (Some(description), Some(weights)) => {
            if description.geometry.temporal_patch != 2 || description.patch_embeddings.len() != 2 {
                return Err(PlanError::Unsupported("vision temporal patch topology"));
            }
            let active = Element::dense(activation_dtype(description.geometry.activation_dtype));
            let patch = VisionPatchBinding {
                temporal_weight_0: lookup(
                    weights,
                    WeightScope::VisionPatch(0),
                    WeightKind::PatchEmbedding,
                )?,
                temporal_weight_1: lookup(
                    weights,
                    WeightScope::VisionPatch(1),
                    WeightKind::PatchEmbedding,
                )?,
                bias: lookup(weights, WeightScope::Vision, WeightKind::PatchBias)?,
                position: lookup(weights, WeightScope::Vision, WeightKind::PositionEmbedding)?,
                activation: active,
            };
            let mut slots = Vec::with_capacity(description.blocks.len());
            for index in 0..description.blocks.len() {
                let scope = WeightScope::VisionBlock(
                    u32::try_from(index)
                        .map_err(|_| PlanError::Arithmetic("vision block index exceeds u32"))?,
                );
                slots.push(VisionBlockBinding {
                    input_norm_weight: lookup(weights, scope, WeightKind::InputNormWeight)?,
                    input_norm_bias: lookup(weights, scope, WeightKind::InputNormBias)?,
                    qkv_weight: lookup(weights, scope, WeightKind::FusedQkvWeight)?,
                    qkv_bias: lookup(weights, scope, WeightKind::FusedQkvBias)?,
                    attention_output: lookup(weights, scope, WeightKind::AttentionOutput)?,
                    attention_output_bias: lookup(weights, scope, WeightKind::AttentionOutputBias)?,
                    feedforward_norm_weight: lookup(
                        weights,
                        scope,
                        WeightKind::FeedForwardNormWeight,
                    )?,
                    feedforward_norm_bias: lookup(weights, scope, WeightKind::FeedForwardNormBias)?,
                    up: lookup(weights, scope, WeightKind::DenseUp)?,
                    up_bias: lookup(weights, scope, WeightKind::FeedForwardUpBias)?,
                    down: lookup(weights, scope, WeightKind::DenseDown)?,
                    down_bias: lookup(weights, scope, WeightKind::FeedForwardDownBias)?,
                    activation: active,
                });
            }
            let merger = VisionMergerBinding {
                output_norm_weight: lookup(weights, WeightScope::Vision, WeightKind::NormWeight)?,
                output_norm_bias: lookup(weights, WeightScope::Vision, WeightKind::NormBias)?,
                hidden: lookup(weights, WeightScope::Vision, WeightKind::MergerHidden)?,
                hidden_bias: lookup(weights, WeightScope::Vision, WeightKind::MergerHiddenBias)?,
                output: lookup(weights, WeightScope::Vision, WeightKind::MergerOutput)?,
                output_bias: lookup(weights, WeightScope::Vision, WeightKind::MergerOutputBias)?,
                activation: active,
            };
            Some(VisionProgramPlan::new(patch, slots, merger))
        }
        (_, None) => None,
        _ => {
            return Err(PlanError::Topology(
                "vision component and definition disagree",
            ))
        }
    };
    let state = StateProgramPlan::new(vec![
        Element::f32(),
        Element::f16(),
        Element::bf16(),
        Element::u32(),
    ]);
    ProgramPlan::new(
        definition,
        imports,
        target_program,
        head_program,
        vision_program,
        state,
    )
}

/// The attention kernel dimensions of one attention geometry.
fn attention_shape(hidden: u64, geometry: &AttentionGeometry) -> Result<AttentionShape, PlanError> {
    let RotarySemantics::Interleaved { width: rotary, .. } = &geometry.rotary;
    if geometry.kv_heads == 0 || geometry.heads % geometry.kv_heads != 0 || *rotary > geometry.width
    {
        return Err(PlanError::Topology("attention heads or rotary width are inconsistent"));
    }
    Ok(AttentionShape {
        hidden,
        kv_heads: geometry.kv_heads,
        group: geometry.heads / geometry.kv_heads,
        rotary_pairs: rotary / 2,
        width: geometry.width,
    })
}
