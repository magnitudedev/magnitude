//! Ordered checked-entry topology. A slot is the requirement and the recipe;
//! there is no second kernel-key set or broad semantic coverage class here.

use super::weights::{activation_dtype, ParallelBinding, PerLayerBinding, PerLayerEntryBinding};
use super::{
    planned_element, planned_scalable, AttentionBinding, DenseBinding, EmbeddingBinding,
    FeaturesBinding, HeadBinding, HeadProjection, HostTablePlan, ReadoutBinding, ReadoutHead,
    RecurrentBinding,
    RoutedBinding, ScalableWeight, WeightPlan,
};
use crate::error::PlanError;
use crate::operators::routed::GeneralRoutedBinding;
use crate::operators::short_conv::ShortConvBinding;
use crate::operators::state_space::StateSpaceBinding;
use crate::operators::vision::VisionKernel;
use crate::operators::{
    self, attention_slot, dense_slot, feed_forward_slot, mixer_slot, paired_block,
};
use magnitude_family_contracts::{
    DraftDefinition, DraftEmbedding, DraftMethod, ModelDefinition, SublayerIndex, TapPoint,
    WeightKind, WeightScope,
};
use magnitude_state::KvCodec;
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
    StateSpace(StateSpaceBinding),
    ShortConv(ShortConvBinding),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FeedForwardProgramSlot {
    Dense(DenseBinding),
    Routed(RoutedBinding),
    GeneralRouted(GeneralRoutedBinding),
    Parallel(ParallelBinding),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TargetBlockProgramSlot {
    mixer: MixerProgramSlot,
    /// Absent for a lone mixer block.
    feed_forward: Option<FeedForwardProgramSlot>,
    /// The per-layer input sublayer after the feed-forward.
    per_layer: Option<PerLayerBinding>,
}

impl TargetBlockProgramSlot {
    pub fn new(mixer: MixerProgramSlot, feed_forward: Option<FeedForwardProgramSlot>) -> Self {
        Self {
            mixer,
            feed_forward,
            per_layer: None,
        }
    }

    pub fn with_per_layer(self, per_layer: Option<PerLayerBinding>) -> Self {
        Self { per_layer, ..self }
    }

    pub fn per_layer(&self) -> Option<PerLayerBinding> {
        self.per_layer
    }

    pub fn mixer(&self) -> MixerProgramSlot {
        self.mixer
    }

    pub fn feed_forward(&self) -> Option<FeedForwardProgramSlot> {
        self.feed_forward
    }
}

/// The target taps a separate draft reads: each tap rounds the residual rows
/// the readout publishes into its column block of the draft input rows, and
/// the readout's features are their fusion (`project_rows` into F32).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TapProgramPlan {
    pub points: Vec<TapPoint>,
    pub fusion: Element,
    /// The extent of the fusion projection's accumulator-scale port.
    pub fusion_scale: u64,
    pub activation: Element,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetProgramPlan {
    embedding: EmbeddingBinding,
    blocks: Vec<TargetBlockProgramSlot>,
    readout: ReadoutBinding,
    features: Option<FeaturesBinding>,
    taps: Option<TapProgramPlan>,
    selection: DType,
    /// The per-layer entry the blocks' per-layer inputs come from.
    per_layer: Option<PerLayerEntryBinding>,
}

impl TargetProgramPlan {
    pub fn new(
        embedding: EmbeddingBinding,
        blocks: Vec<TargetBlockProgramSlot>,
        readout: ReadoutBinding,
        features: Option<FeaturesBinding>,
        taps: Option<TapProgramPlan>,
        selection: DType,
    ) -> Self {
        Self {
            embedding,
            blocks,
            readout,
            features,
            taps,
            selection,
            per_layer: None,
        }
    }

    pub fn with_per_layer(self, per_layer: Option<PerLayerEntryBinding>) -> Self {
        Self { per_layer, ..self }
    }

    /// The per-layer entry, when the model has per-layer inputs.
    pub fn per_layer(&self) -> Option<PerLayerEntryBinding> {
        self.per_layer
    }

    /// The draft taps whose fusion replaces the readout's features.
    pub fn taps(&self) -> Option<&TapProgramPlan> {
        self.taps.as_ref()
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

/// One draft layer: its attention over the fresh block, the same attention
/// injecting context K/V (its input norm the draft's fusion norm), and its
/// dense feed-forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DraftBlockBinding {
    pub attention: AttentionBinding,
    pub injection: AttentionBinding,
    pub feed_forward: DenseBinding,
}

/// DSpark's Markov table (an embedding) and its projection onto the
/// vocabulary (a `dense_output` over the slot logits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MarkovBinding {
    pub embedding: Element,
    pub projection: Element,
    /// The table's width.
    pub rank: u64,
}

/// DFlash2's candidate selector: the hidden projection (a `project_rows`
/// of the output-normed proposing rows) and its two codebooks (embedding
/// gathers of the predecessor and of every candidate).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SelectorBinding {
    pub hidden: Element,
    pub predecessor: Element,
    pub successor: Element,
    pub rank: u64,
    pub top_k: u64,
}

/// DFlash2's block pass: every draft sublayer runs unfused (normed rows,
/// half-0 convolution, plain projections, half-1 convolution plus the
/// residual), so each layer's convolution coefficient projections bind here;
/// the layers' own projection elements are their attention and dense
/// bindings'.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Dflash2Binding {
    /// Convolution taps and coefficient groups.
    pub kernel: u64,
    pub groups: u64,
    /// Per draft layer, its attention's and its feed-forward's coefficient
    /// projection elements.
    pub convolutions: Vec<[Element; 2]>,
    pub selector: SelectorBinding,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftProgramPlan {
    blocks: Vec<DraftBlockBinding>,
    /// The block's token table: the draft's own or the target's.
    embedding: EmbeddingBinding,
    output_norm: Element,
    /// The target's vocabulary projection.
    projection: Element,
    markov: Option<MarkovBinding>,
    dflash2: Option<Dflash2Binding>,
    activation: Element,
}

impl DraftProgramPlan {
    pub fn dflash2(&self) -> Option<&Dflash2Binding> {
        self.dflash2.as_ref()
    }

    pub fn blocks(&self) -> &[DraftBlockBinding] {
        &self.blocks
    }

    pub fn embedding(&self) -> EmbeddingBinding {
        self.embedding
    }

    pub fn output_norm(&self) -> Element {
        self.output_norm
    }

    pub fn projection(&self) -> Element {
        self.projection
    }

    pub fn markov(&self) -> Option<MarkovBinding> {
        self.markov
    }

    pub fn activation(&self) -> Element {
        self.activation
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

/// The distinct kernel specializations of the projector's vision program
/// (`operators::vision`), in first-use order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionProgramPlan {
    kernels: Vec<VisionKernel>,
}

impl VisionProgramPlan {
    pub(crate) fn new(kernels: Vec<VisionKernel>) -> Self {
        Self { kernels }
    }

    pub fn kernels(&self) -> &[VisionKernel] {
        &self.kernels
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
    draft: Option<DraftProgramPlan>,
    vision: Option<VisionProgramPlan>,
    state: StateProgramPlan,
}

impl ProgramPlan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        definition: &ModelDefinition,
        imports: Vec<ImportProgramSlot>,
        target: TargetProgramPlan,
        head: Option<HeadProgramPlan>,
        draft: Option<DraftProgramPlan>,
        vision: Option<VisionProgramPlan>,
        state: StateProgramPlan,
    ) -> Result<Self, PlanError> {
        if draft.as_ref().is_some_and(|draft| {
            definition
                .draft
                .as_ref()
                .map(|description| description.blocks.len())
                != Some(draft.blocks.len())
        }) || draft.is_some() != target.taps.is_some()
        {
            return Err(PlanError::Topology(
                "draft program slots disagree with the draft and its taps",
            ));
        }
        if target.blocks.len() != definition.decoder.blocks.len()
            || target
                .blocks
                .iter()
                .zip(&definition.decoder.blocks)
                .any(|(slot, block)| {
                    paired_block(block).map_or(true, |paired| !operators::slot_matches(&paired, slot))
                })
            || head.as_ref().is_some_and(|head| {
                definition
                    .head
                    .as_ref()
                    .map(|description| description.blocks.len())
                    != Some(head.blocks.len())
            })
            || vision.is_some() != definition.vision.is_some()
        {
            return Err(PlanError::Topology(
                "program slots disagree with model topology",
            ));
        }
        Ok(Self {
            imports,
            target,
            head,
            draft,
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

    /// The separate draft's program, when it drafts.
    pub fn draft(&self) -> Option<&DraftProgramPlan> {
        self.draft.as_ref()
    }

    pub fn vision(&self) -> Option<&VisionProgramPlan> {
        self.vision.as_ref()
    }

    pub fn state(&self) -> &StateProgramPlan {
        &self.state
    }
}

pub(super) fn derive_program_plan(
    definition: &ModelDefinition,
    target: &[WeightPlan],
    head: Option<&[WeightPlan]>,
    vision: Option<&[WeightPlan]>,
    host_tables: &[HostTablePlan],
    history: KvCodec,
) -> Result<ProgramPlan, PlanError> {
    operators::admit(definition, head.is_some())?;
    let lookup = |weights: &[WeightPlan], scope, kind| {
        planned_element(weights, scope, kind)
    };
    let mut seen_imports = HashSet::new();
    let mut imports = Vec::new();
    for (source, resident) in target
        .iter()
        .chain(head.unwrap_or_default())
        .chain(vision.unwrap_or_default())
        // A host-placed weight needs no import entry.
        .filter(|weight| !weight.placed_on_host())
        .map(|weight| (weight.upload, weight.resident))
    {
        let slot = match (source.dtype(), resident.dtype()) {
            (Some(source), Some(resident)) => ImportProgramSlot::Dense { source, resident },
            _ => ImportProgramSlot::Repack { source, resident },
        };
        if seen_imports.insert(slot) {
            imports.push(slot);
        }
    }
    let activation = activation_dtype(definition.decoder.activation_dtype);
    let active_element = Element::dense(activation);
    let embedding = EmbeddingBinding {
        table: lookup(target, WeightScope::Target, WeightKind::Embedding)?,
        activation: active_element,
    };
    let decoder = &definition.decoder;
    let mut blocks = Vec::with_capacity(decoder.blocks.len());
    for (index, block) in decoder.blocks.iter().enumerate() {
        let index =
            u32::try_from(index).map_err(|_| PlanError::Arithmetic("target block index exceeds u32"))?;
        let paired = paired_block(block)?;
        let [mixer, feed_forward] = paired_scopes(index, WeightScope::TargetSublayer);
        blocks.push(TargetBlockProgramSlot::new(
            mixer_slot(
                &paired,
                decoder.hidden,
                |kind| lookup(target, mixer, kind),
                active_element,
                history,
            )?,
            paired
                .feed_forward
                .map(|sublayer| {
                    feed_forward_slot(
                        &sublayer,
                        decoder.hidden,
                        feed_forward,
                        |scope, kind| lookup(target, scope, kind),
                        |scope, kind| planned_scalable(target, scope, kind),
                        active_element,
                    )
                })
                .transpose()?,
        )
        .with_per_layer(
            paired
                .per_layer
                .map(|sublayer| {
                    let scope = WeightScope::TargetSublayer(SublayerIndex {
                        block: index,
                        sublayer: 2,
                    });
                    let entry = decoder.entry.per_layer.as_ref().ok_or(PlanError::Topology(
                        "a per-layer input sublayer without a per-layer entry",
                    ))?;
                    operators::per_layer::binding(
                        sublayer,
                        entry,
                        decoder.hidden,
                        |kind| lookup(target, scope, kind),
                        active_element,
                    )
                })
                .transpose()?,
        ));
    }
    let per_layer_entry = decoder
        .entry
        .per_layer
        .as_ref()
        .map(|entry| {
            let table = host_tables
                .iter()
                .find(|table| table.role.kind == WeightKind::PerLayerTable)
                .ok_or(PlanError::Topology("a per-layer entry without its host table"))?;
            Ok::<_, PlanError>(PerLayerEntryBinding {
                hidden: decoder.hidden,
                layers: entry.layers,
                width: entry.width,
                table_source: table.source,
                table: table.resident,
                projection: lookup(target, WeightScope::Target, WeightKind::PerLayerModelProjection)?,
                norm: lookup(target, WeightScope::Target, WeightKind::PerLayerProjectionNorm)?,
                activation: active_element,
            })
        })
        .transpose()?;
    let progressive = target.iter().any(|weight| {
        matches!(weight.role.kind, WeightKind::OutputPlane(_)) && weight.role.scope == WeightScope::Target
    });
    let readout_head = if progressive {
        ReadoutHead::Progressive
    } else {
        let output = planned_scalable(target, WeightScope::Target, WeightKind::Output)?;
        ReadoutHead::Packed {
            weight: output.element,
            weight_scale: output.scale,
        }
    };
    let readout = ReadoutBinding {
        norm: lookup(target, WeightScope::Target, WeightKind::OutputNorm)?,
        head: readout_head,
        activation: active_element,
    };
    // A selected separate draft is the drafter; an embedded head otherwise.
    let separate = head.and(definition.draft.as_ref());
    let taps = separate
        .map(|draft| {
            let fusion = planned_scalable(target, WeightScope::Draft, WeightKind::DraftFusion)?;
            Ok::<_, PlanError>(TapProgramPlan {
                points: draft.taps.clone(),
                fusion: fusion.element,
                fusion_scale: fusion.scale,
                activation: active_element,
            })
        })
        .transpose()?;
    let target_program = TargetProgramPlan::new(
        embedding,
        blocks,
        readout,
        Some(FeaturesBinding {
            norm: readout.norm,
            activation: active_element,
        }),
        taps,
        activation,
    )
    .with_per_layer(per_layer_entry);
    let draft_program = separate
        .zip(head)
        .map(|(draft, weights)| {
            draft_program_plan(draft, decoder.hidden, target, weights, active_element)
        })
        .transpose()?;
    let head_program = head
        .filter(|_| separate.is_none())
        .map(|weights| {
            let head = definition
                .head
                .as_ref()
                .ok_or(PlanError::Topology("head weights without a head definition"))?;
            let mut slots = Vec::with_capacity(head.depth());
            for (index, head_block) in head.blocks.iter().enumerate() {
                let index = u32::try_from(index)
                    .map_err(|_| PlanError::Arithmetic("head block index exceeds u32"))?;
                let scope = WeightScope::HeadBlock(index);
                let paired = paired_block(&head_block.block)?;
                let [mixer, feed_forward] = paired_scopes(index, WeightScope::HeadSublayer);
                let attention = attention_slot(
                    &paired,
                    decoder.hidden,
                    |kind| lookup(weights, mixer, kind),
                    active_element,
                    KvCodec::Dense,
                )?;
                slots.push(HeadBinding {
                    embedding_table: lookup(target, WeightScope::Target, WeightKind::Embedding)?,
                    embedding_norm: lookup(weights, scope, WeightKind::HeadEmbeddingNorm)?,
                    hidden_norm: lookup(weights, scope, WeightKind::HeadHiddenNorm)?,
                    combine: lookup(weights, scope, WeightKind::HeadCombine)?,
                    attention,
                    feed_forward: feed_forward_slot(
                        &paired
                            .feed_forward
                            .ok_or(PlanError::Unsupported("draft head block without feed-forward"))?,
                        decoder.hidden,
                        feed_forward,
                        |scope, kind| lookup(weights, scope, kind),
                        // Head graphs bind no scale ports.
                        |scope, kind| {
                            lookup(weights, scope, kind)
                                .map(|element| ScalableWeight { element, scale: 0 })
                        },
                        active_element,
                    )?,
                    output_norm: lookup(weights, scope, WeightKind::OutputNorm)?,
                    projection: if progressive {
                        HeadProjection::Progressive
                    } else {
                        HeadProjection::Packed(lookup(target, WeightScope::Target, WeightKind::Output)?)
                    },
                    activation: active_element,
                });
            }
            Ok::<_, PlanError>(HeadProgramPlan::new(slots))
        })
        .transpose()?;
    let vision_program = match (definition.vision.as_ref(), vision) {
        (Some(description), Some(weights)) => {
            let program = operators::vision::vision_program(description, &|role| {
                lookup(weights, role.scope, role.kind)
            })?;
            Some(VisionProgramPlan::new(
                program.kernels().into_iter().cloned().collect(),
            ))
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
        draft_program,
        vision_program,
        state,
    )
}

/// The separate draft's program slots from its weights (`weights`, the
/// drafter's own) and the target's (the fusion norm, the target's embedding
/// and vocabulary projection).
fn draft_program_plan(
    draft: &DraftDefinition,
    hidden: u64,
    target: &[WeightPlan],
    weights: &[WeightPlan],
    activation: Element,
) -> Result<DraftProgramPlan, PlanError> {
    let lookup = |weights: &[WeightPlan], scope, kind| {
        planned_element(weights, scope, kind)
    };
    let fusion_norm = lookup(target, WeightScope::Draft, WeightKind::DraftFusionNorm)?;
    let mut blocks = Vec::with_capacity(draft.blocks.len());
    for index in 0..draft.blocks.len() {
        let paired = operators::draft::draft_block(draft, index)?;
        let block =
            u32::try_from(index).map_err(|_| PlanError::Arithmetic("draft block index exceeds u32"))?;
        let [mixer, feed_forward] = paired_scopes(block, WeightScope::DraftSublayer);
        // Draft history is dense, as a draft head's is.
        let attention = attention_slot(
            &paired,
            hidden,
            |kind| lookup(weights, mixer, kind),
            activation,
            KvCodec::Dense,
        )?;
        let dense = dense_slot(
            &paired
                .feed_forward
                .ok_or(PlanError::Unsupported("draft layer without feed-forward"))?,
            |kind| lookup(weights, feed_forward, kind),
            // DFlash and DSpark layers run the dense feed-forward graph, whose
            // entries bind scale ports; DFlash2 layers project through their
            // own entries, which bind none.
            |kind| match draft.method {
                DraftMethod::DFlash2 { .. } => lookup(weights, feed_forward, kind)
                    .map(|element| ScalableWeight { element, scale: 0 }),
                DraftMethod::DFlash | DraftMethod::DSpark { .. } => {
                    planned_scalable(weights, feed_forward, kind)
                }
            },
            activation,
        )?;
        blocks.push(DraftBlockBinding {
            attention,
            injection: AttentionBinding {
                norm: fusion_norm,
                ..attention
            },
            feed_forward: dense,
        });
    }
    Ok(DraftProgramPlan {
        blocks,
        embedding: EmbeddingBinding {
            table: match &draft.embedding {
                DraftEmbedding::Target => {
                    lookup(target, WeightScope::Target, WeightKind::Embedding)?
                }
                DraftEmbedding::Own(_) => lookup(weights, WeightScope::Draft, WeightKind::Embedding)?,
            },
            activation,
        },
        output_norm: lookup(weights, WeightScope::Draft, WeightKind::OutputNorm)?,
        // The draft's head entry binds the projection's scale port as the
        // target readout does.
        projection: planned_scalable(target, WeightScope::Target, WeightKind::Output)?.element,
        markov: match &draft.method {
            DraftMethod::DFlash | DraftMethod::DFlash2 { .. } => None,
            DraftMethod::DSpark { markov, .. } => Some(MarkovBinding {
                embedding: lookup(weights, WeightScope::Draft, WeightKind::MarkovEmbedding)?,
                projection: lookup(weights, WeightScope::Draft, WeightKind::MarkovProjection)?,
                rank: markov.rank,
            }),
        },
        dflash2: match &draft.method {
            DraftMethod::DFlash | DraftMethod::DSpark { .. } => None,
            DraftMethod::DFlash2 {
                kernel,
                group,
                convolutions,
                selector,
            } => Some(Dflash2Binding {
                kernel: *kernel,
                groups: hidden / *group,
                convolutions: (0..convolutions.len())
                    .map(|index| {
                        let block = u32::try_from(index)
                            .map_err(|_| PlanError::Arithmetic("draft block index exceeds u32"))?;
                        let [attention, feed_forward] =
                            paired_scopes(block, WeightScope::DraftSublayer);
                        Ok([
                            lookup(weights, attention, WeightKind::ConvolutionProjection)?,
                            lookup(weights, feed_forward, WeightKind::ConvolutionProjection)?,
                        ])
                    })
                    .collect::<Result<Vec<_>, PlanError>>()?,
                selector: SelectorBinding {
                    hidden: lookup(weights, WeightScope::Draft, WeightKind::SelectorHidden)?,
                    predecessor: lookup(weights, WeightScope::Draft, WeightKind::SelectorPredecessor)?,
                    successor: lookup(weights, WeightScope::Draft, WeightKind::SelectorSuccessor)?,
                    rank: selector.rank,
                    top_k: selector.top_k,
                },
            }),
        },
        activation,
    })
}

/// The weight scopes of a paired block's mixer and feed-forward sublayers.
fn paired_scopes(block: u32, scope: fn(SublayerIndex) -> WeightScope) -> [WeightScope; 2] {
    [0, 1].map(|sublayer| scope(SublayerIndex { block, sublayer }))
}


