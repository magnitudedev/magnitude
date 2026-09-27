//! The native separate-draft program (DFlash, DSpark; plan §3.8). One sealed
//! graph per class runs a draft transaction in one device submission:
//!
//! - **Injection.** For every draft layer, the layer's attention over the
//!   entry rows with the target's fused taps (widened to F32) as its hidden
//!   rows and the draft's fusion norm as its input norm, each row attending
//!   only itself: the entry K/V norm, rotary and append every attention
//!   layer uses write the context K/V at the rows' destinations. The
//!   attention's own output is not read.
//! - **Block.** When drafting, one non-causal pass over each slot's block
//!   `[anchor, mask, …]`: the raw token embedding, every draft layer's
//!   attention (over the domain's accepted and injected rows, and the whole
//!   block) and dense feed-forward, then the output norm and the target's
//!   vocabulary projection of the proposing rows and position-keyed
//!   selection. DSpark then chains its slots: slot `k`'s logits gain the
//!   Markov projection of the token before it, its selection feeds slot
//!   `k + 1`, and a confidence below the threshold declines the proposal.
//!
//! Selections are step-major: proposal `k` of slot `s` is result row
//! `k · slots + s`.

use super::{DeviceSubmission, HeadProgram};
use crate::{
    completion::CompletionWaiter,
    native::{AttestedDraft, AttestedTarget},
    operators::attention::graph::{
        self as attention_graph, attention_weights, AttentionBlock, AttentionControlPorts,
        AttentionGraphEntries, AttentionWeights, CheckedAttentionEntries,
    },
    operators::dense_ffn::graph::{self as dense_graph, CheckedDenseEntries, DenseGraphEntries},
    programs::{
        graph::draft::GraphDraft,
        graph::readout::{self, readout_softcap, shapes, write_selection_rows, SelectionPorts},
        native_constants::{
            distinct_storage_bytes, CheckedGraphFamilyResources, CheckedGraphResources,
            ConstantTensors, GraphConstant,
        },
        native_target_graph::weight,
    },
    DeviceError, DraftProgramPlan, GraphOutputTensor, HeadLaunchCore, InvariantError, ModelLoadPlan,
    NativeGraphOutputLease, NativeGraphWorkspaceLease, ResidentHead, ResidentWeight,
    ResourceLimits, StateStorePlan, SubmitError, ValidatedHeadLaunch,
};
use crate::operators::{self, Mixer};
use magnitude_batching::{row_class, TargetBatchUpload};
use magnitude_family_contracts::{
    ActivationDType, DraftDefinition, DraftEmbedding, DraftMethod, ModelDefinition, SublayerIndex,
    WeightKind, WeightRole, WeightScope,
};
use magnitude_kernels::{
    dense_output, draft_confidence, embedding_rows, readout_features_rows, readout_head_rows,
    sample_rows, shape_rows,
};
use magnitude_state::LayerRef;
use seismic::{
    BackendName, BoundNativeGraphPlan, Device, Element, NativeGraph, NativeGraphFamily,
    NativeGraphMetadata, NativeGraphPlan, NativePort, Tensor, WorkflowTensor,
};
use std::{collections::BTreeMap, rc::Rc};

fn invalid(detail: impl Into<String>) -> SubmitError {
    SubmitError::Invariant(InvariantError {
        context: "native draft program",
        detail: detail.into(),
    })
}
fn device(error: impl ToString) -> SubmitError {
    SubmitError::Device(DeviceError::Execution(error.to_string()))
}
fn i32_bytes(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values
        .into_iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}
fn activation(dtype: ActivationDType) -> Element {
    match dtype {
        ActivationDType::F16 => Element::f16(),
        ActivationDType::BF16 => Element::bf16(),
    }
}

/// DSpark's confidence threshold: a slot below it declines its proposal. 0
/// (the default of the reference service) never declines.
const CONFIDENCE_THRESHOLD: f32 = 0.0;

/// One draft graph: the entry rows' class, and when drafting the class of
/// the drafting slots and whether selection is shaped first. Every drafting
/// class drafts the load's proposals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DraftGraphClass {
    pub entry_rows: u64,
    /// 0 for an injection-only transaction.
    pub slots: u64,
    pub shaped: bool,
}

/// The admitted draft classes: every entry row class injecting only, and
/// every slot class whose blocks (`block` rows each) fit a launch drafting
/// after entry rows up to its proposal rows.
pub(crate) fn draft_graph_classes(
    limits: ResourceLimits,
    proposals: usize,
    block: u64,
) -> Result<Vec<DraftGraphClass>, String> {
    let row_classes = magnitude_batching::row_classes(limits.max_launch_rows)
        .into_iter()
        .map(|rows| rows as u64)
        .collect::<Vec<_>>();
    let max_rows = *row_classes.last().ok_or_else(|| {
        format!(
            "launch row bound {} has no row class",
            limits.max_launch_rows
        )
    })?;
    let slot_classes = magnitude_batching::row_classes(
        limits.max_launch_slots.min(limits.max_launch_rows),
    )
    .into_iter()
    .map(|slots| slots as u64)
    .filter(|slots| slots * block <= max_rows)
    .collect::<Vec<_>>();
    let mut classes = row_classes
        .iter()
        .map(|&entry_rows| DraftGraphClass {
            entry_rows,
            slots: 0,
            shaped: false,
        })
        .collect::<Vec<_>>();
    for &slots in &slot_classes {
        let entry_bound = row_class(
            (slots as usize)
                .saturating_mul(proposals + 1)
                .min(max_rows as usize),
        )
        .ok_or("draft entry row bound has no class")? as u64;
        for &entry_rows in row_classes
            .iter()
            .filter(|rows| **rows >= slots && **rows <= entry_bound)
        {
            for shaped in [false, true] {
                classes.push(DraftGraphClass {
                    entry_rows,
                    slots,
                    shaped,
                });
            }
        }
    }
    Ok(classes)
}

/// Every class's fixed geometry: the draft, each layer's history rows and
/// slab rows, the block pass's visible spans, and the proposals drafted.
pub(crate) struct DraftGeometry<'d> {
    pub definition: &'d ModelDefinition,
    pub draft: &'d DraftDefinition,
    /// Per draft layer, its history domain's rows and slab rows.
    pub layers: Vec<(u64, u32)>,
    pub segments: u64,
    pub proposals: u64,
}

impl<'d> DraftGeometry<'d> {
    pub(crate) fn new(
        definition: &'d ModelDefinition,
        state: &StateStorePlan,
        proposals: usize,
    ) -> Result<Self, String> {
        let draft = definition
            .draft
            .as_ref()
            .ok_or("draft graphs require a draft")?;
        let layers = (0..draft.blocks.len())
            .map(|index| {
                let layer = LayerRef::Head(u32::try_from(index).map_err(|_| "draft layer index")?);
                let history = state
                    .layer_history(layer)
                    .ok_or("a draft layer has no history in the draft store")?;
                Ok((
                    u64::try_from(history.store.rows).map_err(|_| "draft history rows")?,
                    history.store.slab_rows,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let segments = u64::try_from(
            state
                .max_visible_spans()?
                .checked_next_power_of_two()
                .ok_or("draft segment class overflows")?,
        )
        .map_err(|_| "draft segment class exceeds u64")?;
        Ok(Self {
            definition,
            draft,
            layers,
            segments,
            proposals: proposals as u64,
        })
    }
}

/// The draft's entries in the graph's binding form.
pub(crate) struct DraftGraphEntries<'a, G: GraphDraft + 'a> {
    pub layers: Vec<DraftLayerEntries<'a, G>>,
    pub embedding: G::Binding<'a, embedding_rows::Entry>,
    pub head: G::Binding<'a, readout_head_rows::Entry>,
    pub shape: G::Binding<'a, shape_rows::Entry>,
    pub sample: G::Binding<'a, sample_rows::Entry>,
    pub markov: Option<MarkovEntries<'a, G>>,
}

pub(crate) struct DraftLayerEntries<'a, G: GraphDraft + 'a> {
    pub attention: AttentionGraphEntries<'a, G>,
    pub injection: AttentionGraphEntries<'a, G>,
    pub dense: DenseGraphEntries<'a, G>,
}

pub(crate) struct MarkovEntries<'a, G: GraphDraft + 'a> {
    pub embedding: G::Binding<'a, embedding_rows::Entry>,
    pub projection: G::Binding<'a, dense_output::Entry>,
    pub features: G::Binding<'a, readout_features_rows::Entry>,
    pub confidence: G::Binding<'a, draft_confidence::Entry>,
}

impl<'a> DraftGraphEntries<'a, NativeGraph> {
    fn prepared(draft: &'a AttestedDraft, target: &'a AttestedTarget) -> Self {
        Self {
            layers: draft
                .blocks
                .iter()
                .map(|block| DraftLayerEntries {
                    attention: (&block.attention).into(),
                    injection: (&block.injection).into(),
                    dense: (&block.dense).into(),
                })
                .collect(),
            embedding: &draft.embedding,
            head: &draft.head,
            shape: &target.shape,
            sample: &target.sample,
            markov: draft.markov.as_ref().map(|markov| MarkovEntries {
                embedding: &markov.embedding,
                projection: &markov.projection,
                features: &markov.features,
                confidence: &markov.confidence,
            }),
        }
    }
}

/// The draft's entries as element assignments of its program plan, for the
/// metadata-only (checked) graphs.
struct CheckedDraftEntries {
    layers: Vec<(CheckedAttentionEntries, CheckedAttentionEntries, CheckedDenseEntries)>,
    embedding: [(&'static str, Element); 2],
    head: [(&'static str, Element); 3],
    selection: [(&'static str, Element); 0],
    markov: Option<CheckedMarkovEntries>,
}

struct CheckedMarkovEntries {
    embedding: [(&'static str, Element); 2],
    projection: [(&'static str, Element); 2],
    features: [(&'static str, Element); 2],
    confidence: [(&'static str, Element); 1],
}

impl CheckedDraftEntries {
    fn new(plan: &DraftProgramPlan) -> Self {
        let activation = plan.activation();
        Self {
            layers: plan
                .blocks()
                .iter()
                .map(|block| {
                    (
                        CheckedAttentionEntries::new(block.attention),
                        CheckedAttentionEntries::new(block.injection),
                        CheckedDenseEntries::new(block.feed_forward),
                    )
                })
                .collect(),
            embedding: [("EW", plan.embedding().table), ("A", activation)],
            head: [
                ("NW", plan.output_norm()),
                ("OW", plan.projection()),
                ("A", activation),
            ],
            selection: [],
            markov: plan.markov().map(|markov| CheckedMarkovEntries {
                embedding: [("EW", markov.embedding), ("A", activation)],
                projection: [("DW", markov.projection), ("A", activation)],
                features: [("NW", plan.output_norm()), ("A", activation)],
                confidence: [("A", activation)],
            }),
        }
    }

    fn entries(&self) -> Result<DraftGraphEntries<'_, NativeGraphMetadata>, String> {
        Ok(DraftGraphEntries {
            layers: self
                .layers
                .iter()
                .map(|(attention, injection, dense)| {
                    Ok(DraftLayerEntries {
                        attention: attention.entries()?,
                        injection: injection.entries()?,
                        dense: dense.entries(),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
            embedding: &self.embedding[..],
            head: &self.head[..],
            shape: &self.selection[..],
            sample: &self.selection[..],
            markov: self.markov.as_ref().map(|markov| MarkovEntries {
                embedding: &markov.embedding[..],
                projection: &markov.projection[..],
                features: &markov.features[..],
                confidence: &markov.confidence[..],
            }),
        })
    }
}

/// The storage and bound constants of the draft's graph family over
/// `classes`, from metadata alone: each class is sealed exactly, and the
/// family holds the largest of each arena.
pub(crate) fn checked_draft_family_storage(
    backend: BackendName,
    load: &ModelLoadPlan,
    plan: &DraftProgramPlan,
    geometry: &DraftGeometry<'_>,
    classes: impl IntoIterator<Item = DraftGraphClass>,
) -> Result<CheckedGraphResources, String> {
    let checked = CheckedDraftEntries::new(plan);
    let mut family = CheckedGraphFamilyResources::new();
    for class in classes {
        let parts = draft_graph(
            NativeGraphMetadata::new(backend),
            checked.entries()?,
            load,
            geometry,
            class,
        )
        .map_err(|error| format!("draft graph class {class:?}: {error}"))?;
        let storage = GraphDraft::seal(parts.plan)
            .map_err(|error| format!("draft graph class {class:?}: {error}"))?;
        family.include(storage, parts.constants);
    }
    family.finish()
}

/// One layer's per-run ports in one pass: its attention controls and its
/// history planes, in plane-descriptor order.
struct LayerPorts {
    controls: AttentionControlPorts,
    planes: Vec<NativePort>,
}

struct BlockPorts {
    /// `[slots · block, 2]` (token, status) rows.
    tokens: NativePort,
    /// The block row of each proposal, step-major.
    head_rows: NativePort,
    layers: Vec<LayerPorts>,
    /// One selection for every proposal (DFlash), or one per step (DSpark).
    selections: Vec<SelectionPorts>,
    /// DSpark: each slot's anchor, the token its first step conditions on.
    anchors: Option<NativePort>,
}

struct DraftGraphParts<P> {
    plan: P,
    /// `[entry rows, hidden]` F32 conditioning rows.
    conditioning: NativePort,
    injection: Vec<LayerPorts>,
    block: Option<BlockPorts>,
    constants: Vec<GraphConstant>,
    weights: Vec<(WeightRole, NativePort)>,
    /// Selections `[proposals · slots, 2]` when drafting; otherwise the
    /// last injection's rows, exported so the graph has a result.
    output: WorkflowTensor,
}

fn draft_sublayer(block: usize, sublayer: u32) -> Result<WeightScope, String> {
    Ok(WeightScope::DraftSublayer(SublayerIndex {
        block: u32::try_from(block).map_err(|_| "draft block index exceeds u32")?,
        sublayer,
    }))
}

fn draft_graph<'a, G: GraphDraft + 'a>(
    mut graph: G,
    entries: DraftGraphEntries<'a, G>,
    load: &ModelLoadPlan,
    geometry: &DraftGeometry<'_>,
    class: DraftGraphClass,
) -> Result<DraftGraphParts<G>, String> {
    let decoder = &geometry.definition.decoder;
    let draft = geometry.draft;
    let (hidden, vocabulary) = (decoder.hidden, decoder.vocabulary);
    let activation = activation(decoder.activation_dtype);
    let epsilon = draft.output_norm.epsilon as f32;
    if entries.layers.len() != draft.blocks.len() || geometry.layers.len() != draft.blocks.len() {
        return Err("draft entries disagree with the draft's layers".into());
    }
    let mut weights = Vec::new();
    let mut constants = Vec::new();
    let absent_scale = GraphConstant::absent_scale(&mut graph, &mut constants)?;
    let fusion_norm = weight(
        &mut graph,
        load,
        WeightScope::Draft,
        WeightKind::DraftFusionNorm,
        &mut weights,
    )?;
    // Each layer's attention weights for injection, whose input norm is the
    // fusion norm; the block pass reads the same weights with the layer's
    // own input norm.
    let mut layer_weights = Vec::with_capacity(draft.blocks.len());
    for index in 0..draft.blocks.len() {
        let paired =
            operators::draft::draft_block(draft, index).map_err(|error| error.to_string())?;
        let Mixer::Attention(attention) = paired.mixer else {
            return Err("draft layers attend".into());
        };
        let shape =
            operators::attention::shape(hidden, attention).map_err(|error| error.to_string())?;
        let scope = draft_sublayer(index, 0)?;
        let injection = attention_weights(&shape, attention, false, |kind| match kind {
            WeightKind::InputNorm => Ok(fusion_norm.clone()),
            kind => weight(&mut graph, load, scope, kind, &mut weights),
        })?;
        layer_weights.push((attention, shape, injection));
    }
    fn attention_block<'o>(
        operator: &'o magnitude_family_contracts::Attention,
        shape: crate::AttentionShape,
        rows: u64,
        segments: u64,
        (history_rows, slab_rows): (u64, u32),
        epsilon: f32,
        activation: Element,
    ) -> AttentionBlock<'o> {
        AttentionBlock {
            rows,
            segments,
            history_rows,
            slab_rows,
            shape,
            operator,
            epsilon,
            head_epsilon: epsilon,
            post_norm_epsilon: 0.0,
            post_norm_scale: 1.0,
            activation,
        }
    }

    // Injection: each entry row appends its context K/V at its destination.
    // The host writes the rows: an input of the first injection's
    // projection (every layer's projection reads the same rows).
    let (_, first_shape, _) = layer_weights.first().ok_or("a draft has no layers")?;
    let conditioning = graph.input_for(
        entries.layers[0].injection.project,
        "hidden",
        &first_shape.project_dimensions(class.entry_rows),
    )?;
    let conditioning_rows = conditioning.tensor().clone();
    let mut injection = Vec::with_capacity(draft.blocks.len());
    let mut injected = None;
    for (index, ((operator, shape, weights_of), layer)) in
        layer_weights.iter().zip(&entries.layers).enumerate()
    {
        let (mixed, state, controls) = attention_graph::attention(
            &mut graph,
            layer.injection,
            weights_of,
            &mut constants,
            &conditioning_rows,
            attention_block(
                operator,
                *shape,
                class.entry_rows,
                1,
                geometry.layers[index],
                epsilon,
                activation,
            ),
        )?;
        injection.push(LayerPorts {
            controls,
            planes: state.planes,
        });
        injected = Some(mixed);
    }
    let injected = injected.ok_or("a draft has no layers")?;
    if class.slots == 0 {
        graph.export(&injected)?;
        return Ok(DraftGraphParts {
            plan: graph,
            conditioning,
            injection,
            block: None,
            constants,
            weights,
            output: injected,
        });
    }

    // The block pass.
    let slots = class.slots;
    let rows = slots * draft.block_size;
    let proposals = geometry.proposals;
    let outputs = proposals * slots;
    let table = match &draft.embedding {
        DraftEmbedding::Target => weight(
            &mut graph,
            load,
            WeightScope::Target,
            WeightKind::Embedding,
            &mut weights,
        )?,
        DraftEmbedding::Own(_) => weight(
            &mut graph,
            load,
            WeightScope::Draft,
            WeightKind::Embedding,
            &mut weights,
        )?,
    };
    let embedding_dims = [("M", rows), ("V", vocabulary), ("D", hidden)];
    let tokens = graph.input_for(entries.embedding, "tokens", &embedding_dims)?;
    let mut residual = graph
        .enqueue(
            entries.embedding,
            &embedding_dims,
            embedding_rows::WorkflowArgs {
                table: (&table).into(),
                tokens: tokens.tensor().into(),
                scale: 1.0,
                normalize: 0,
                epsilon,
            },
        )?
        .r1;
    let mut layers = Vec::with_capacity(draft.blocks.len());
    for (index, ((operator, shape, injection_weights), layer)) in
        layer_weights.iter().zip(&entries.layers).enumerate()
    {
        let input_norm = weight(
            &mut graph,
            load,
            draft_sublayer(index, 0)?,
            WeightKind::InputNorm,
            &mut weights,
        )?;
        let block_weights = AttentionWeights {
            input_norm,
            query: injection_weights.query.clone(),
            gate: injection_weights.gate.clone(),
            key: injection_weights.key.clone(),
            value: injection_weights.value.clone(),
            query_norm: injection_weights.query_norm.clone(),
            key_norm: injection_weights.key_norm.clone(),
            output: injection_weights.output.clone(),
            post_norm: None,
        };
        let (mixed, state, controls) = attention_graph::attention(
            &mut graph,
            layer.attention,
            &block_weights,
            &mut constants,
            &residual,
            attention_block(
                operator,
                *shape,
                rows,
                geometry.segments,
                geometry.layers[index],
                epsilon,
                activation,
            ),
        )?;
        layers.push(LayerPorts {
            controls,
            planes: state.planes,
        });
        let paired =
            operators::draft::draft_block(draft, index).map_err(|error| error.to_string())?;
        let Some(operators::FeedForward::Dense(dense)) =
            paired.feed_forward.map(|sublayer| sublayer.op)
        else {
            return Err("draft layers have a dense feed-forward".into());
        };
        residual = dense_graph::dense(
            &mut graph,
            DenseGraphEntries {
                expand: layer.dense.expand,
                output: layer.dense.output,
            },
            load,
            draft_sublayer(index, 1)?,
            &mut weights,
            &mut constants,
            &mixed,
            rows,
            epsilon,
            operators::dense_ffn::activation_code(dense.up.activation()),
            0.0,
            1.0,
        )?;
    }
    let output_norm = weight(
        &mut graph,
        load,
        WeightScope::Draft,
        WeightKind::OutputNorm,
        &mut weights,
    )?;
    let projection = weight(
        &mut graph,
        load,
        WeightScope::Target,
        WeightKind::Output,
        &mut weights,
    )?;
    let head_dims = [("M", rows), ("O", outputs), ("V", vocabulary), ("D", hidden)];
    let head_rows = graph.input_for(entries.head, "out_rows", &head_dims)?;
    let logits = graph
        .enqueue(
            entries.head,
            &head_dims,
            readout_head_rows::WorkflowArgs {
                hidden: (&residual).into(),
                norm: (&output_norm).into(),
                weight: (&projection).into(),
                out_rows: head_rows.tensor().into(),
                epsilon,
                softcap: readout_softcap(decoder),
            },
        )?
        .value;
    let result = graph.local_for(
        entries.sample,
        "result",
        &[("M", outputs), ("V", vocabulary)],
    )?;
    let mut selections = Vec::new();
    let anchors = match (&draft.method, &entries.markov) {
        (DraftMethod::DFlash, None) => {
            let mut all = result.tensor().slice_leading(0, outputs);
            selections.push(readout::sample(
                &mut graph,
                entries.shape,
                entries.sample,
                vocabulary,
                (&logits).into(),
                outputs,
                class.shaped,
                (&mut all).into(),
            )?);
            None
        }
        (DraftMethod::DSpark { markov, .. }, Some(chain)) => {
            let rank = markov.rank;
            let markov_table = weight(
                &mut graph,
                load,
                WeightScope::Draft,
                WeightKind::MarkovEmbedding,
                &mut weights,
            )?;
            let markov_projection = weight(
                &mut graph,
                load,
                WeightScope::Draft,
                WeightKind::MarkovProjection,
                &mut weights,
            )?;
            let confidence_weight = weight(
                &mut graph,
                load,
                WeightScope::Draft,
                WeightKind::ConfidenceWeight,
                &mut weights,
            )?;
            let confidence_bias = weight(
                &mut graph,
                load,
                WeightScope::Draft,
                WeightKind::ConfidenceBias,
                &mut weights,
            )?;
            let memory_dims = [("M", slots), ("V", vocabulary), ("D", rank)];
            let anchors = graph.input_for(chain.embedding, "tokens", &memory_dims)?;
            for step in 0..proposals {
                let rows_of_step = GraphConstant::i32(
                    &mut graph,
                    &(step * slots..(step + 1) * slots)
                        .map(|row| i32::try_from(row).map_err(|_| "draft output row exceeds i32"))
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                let previous = match step {
                    0 => anchors.tensor().slice_leading(0, slots),
                    _ => result
                        .tensor()
                        .slice_leading((step - 1) * slots, step * slots),
                };
                let memory = graph
                    .enqueue(
                        chain.embedding,
                        &memory_dims,
                        embedding_rows::WorkflowArgs {
                            table: (&markov_table).into(),
                            tokens: (&previous).into(),
                            scale: 1.0,
                            normalize: 0,
                            epsilon,
                        },
                    )?
                    .r0;
                let biased = graph
                    .enqueue(
                        chain.projection,
                        &[("M", outputs), ("O", slots), ("H", vocabulary), ("F", rank), ("DS", 0)],
                        dense_output::WorkflowArgs {
                            residual: (&logits).into(),
                            product: (&memory).into(),
                            down_weight: (&markov_projection).into(),
                            out_rows: rows_of_step.port().tensor().into(),
                            down_scale: (&absent_scale).into(),
                        },
                    )?
                    .value;
                let mut step_result = result
                    .tensor()
                    .slice_leading(step * slots, (step + 1) * slots);
                selections.push(readout::sample(
                    &mut graph,
                    entries.shape,
                    entries.sample,
                    vocabulary,
                    (&biased).into(),
                    slots,
                    class.shaped,
                    (&mut step_result).into(),
                )?);
                let step_rows = head_rows
                    .tensor()
                    .slice_leading(step * slots, (step + 1) * slots);
                let features = graph
                    .enqueue(
                        chain.features,
                        &[("M", rows), ("O", slots), ("D", hidden)],
                        readout_features_rows::WorkflowArgs {
                            hidden: (&residual).into(),
                            norm: (&output_norm).into(),
                            out_rows: (&step_rows).into(),
                            epsilon,
                        },
                    )?
                    .value;
                graph.enqueue(
                    chain.confidence,
                    &[("S", slots), ("D", hidden), ("R", rank)],
                    draft_confidence::WorkflowArgs {
                        features: (&features).into(),
                        memory: (&memory).into(),
                        weight: (&confidence_weight).into(),
                        bias: (&confidence_bias).into(),
                        threshold: CONFIDENCE_THRESHOLD,
                        selection: (&mut step_result).into(),
                    },
                )?;
                constants.push(rows_of_step);
            }
            Some(anchors)
        }
        _ => return Err("draft method and entries disagree".into()),
    };
    let output = result.tensor().clone();
    graph.export(&output)?;
    Ok(DraftGraphParts {
        plan: graph,
        conditioning,
        injection,
        block: Some(BlockPorts {
            tokens,
            head_rows,
            layers,
            selections,
            anchors,
        }),
        constants,
        weights,
        output,
    })
}

type PreparedDraftGraph = DraftGraphParts<NativeGraphPlan>;

pub struct PreparedDraftGraphs {
    classes: BTreeMap<DraftGraphClass, PreparedDraftGraph>,
    family: NativeGraphFamily,
    /// The block pass's visible spans.
    segments: u64,
    /// Proposals every drafting class drafts.
    proposals: u64,
}

pub(crate) struct BoundDraftGraphs {
    prepared: Rc<PreparedDraftGraphs>,
    bound: BTreeMap<DraftGraphClass, BoundNativeGraphPlan>,
    constants: Vec<Tensor>,
}

impl PreparedDraftGraphs {
    pub fn binding_constant_bytes(&self) -> Result<u64, String> {
        distinct_storage_bytes(
            self.classes
                .values()
                .flat_map(|graph| graph.constants.iter()),
        )
    }

    pub(crate) fn prepare(
        target_device: &Device,
        draft: &AttestedDraft,
        target: &AttestedTarget,
        load: &ModelLoadPlan,
        geometry: &DraftGeometry<'_>,
        classes: impl IntoIterator<Item = DraftGraphClass>,
    ) -> Result<Self, SubmitError> {
        let mut prepared = BTreeMap::new();
        for class in classes {
            if prepared.contains_key(&class) {
                return Err(invalid("draft graph class is duplicated"));
            }
            let class_error = |error: String| invalid(format!("draft graph class {class:?}: {error}"));
            let parts = draft_graph(
                target_device.native_graph(),
                DraftGraphEntries::prepared(draft, target),
                load,
                geometry,
                class,
            )
            .map_err(class_error)?;
            let plan = GraphDraft::seal(parts.plan).map_err(class_error)?;
            prepared.insert(
                class,
                DraftGraphParts {
                    plan,
                    conditioning: parts.conditioning,
                    injection: parts.injection,
                    block: parts.block,
                    constants: parts.constants,
                    weights: parts.weights,
                    output: parts.output,
                },
            );
        }
        if prepared.is_empty() {
            return Err(invalid("draft graph family has no classes"));
        }
        let plans = prepared
            .values()
            .map(|graph| graph.plan.clone())
            .collect::<Vec<_>>();
        let family = NativeGraphFamily::new(&plans).map_err(device)?;
        Ok(Self {
            classes: prepared,
            family,
            segments: geometry.segments,
            proposals: geometry.proposals,
        })
    }

    pub fn family(&self) -> &NativeGraphFamily {
        &self.family
    }

    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    pub(crate) fn bind_weights(
        self: &Rc<Self>,
        resident: &ResidentHead,
    ) -> Result<BoundDraftGraphs, SubmitError> {
        let mut uploaded = ConstantTensors::new(resident.embedding.tensor().device());
        let mut bound = BTreeMap::new();
        for (class, graph) in &self.classes {
            let constants = graph
                .constants
                .iter()
                .map(|constant| Ok((constant.port(), uploaded.tensor(constant).map_err(device)?)))
                .collect::<Result<Vec<_>, SubmitError>>()?;
            let fixed = graph
                .weights
                .iter()
                .map(|(role, port)| Ok((port, resident_draft_weight(resident, *role)?.tensor())))
                .chain(constants.iter().map(|(port, tensor)| Ok((*port, tensor))))
                .collect::<Result<Vec<_>, SubmitError>>()?;
            bound.insert(*class, graph.plan.bind_static(&fixed).map_err(device)?);
        }
        Ok(BoundDraftGraphs {
            prepared: self.clone(),
            bound,
            constants: uploaded.into_tensors(),
        })
    }
}

impl BoundDraftGraphs {
    fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.constants.iter().try_fold(0u64, |bytes, tensor| {
            bytes
                .checked_add(tensor.storage_bytes())
                .ok_or("draft graph constant charge overflows")
        })
    }

    fn class(
        &self,
        class: DraftGraphClass,
    ) -> Result<(&PreparedDraftGraph, &BoundNativeGraphPlan), SubmitError> {
        let graph = self
            .prepared
            .classes
            .get(&class)
            .ok_or_else(|| invalid(format!("draft graph class {class:?} was not prepared")))?;
        let bound = self
            .bound
            .get(&class)
            .ok_or_else(|| invalid(format!("draft graph class {class:?} was not bound")))?;
        Ok((graph, bound))
    }
}

/// A draft graph's weight: the block's embedding table (the target's or the
/// draft's own), the target's projection, or a drafter weight.
fn resident_draft_weight(
    resident: &ResidentHead,
    role: WeightRole,
) -> Result<&ResidentWeight, SubmitError> {
    match (role.scope, role.kind) {
        (WeightScope::Target | WeightScope::Draft, WeightKind::Embedding) => {
            Ok(&resident.embedding)
        }
        (WeightScope::Target, WeightKind::Output) => Ok(&resident.output),
        _ => resident.weights.get(role).map_err(invalid),
    }
}

/// One pass's attention controls for one layer's history domain, padded to
/// `rows` rows and `segments` spans: padding rows attend to nothing and
/// append nowhere. The block pass reads its whole slot block fresh.
struct PassControls {
    coordinates: Vec<u8>,
    visible: Vec<u8>,
    fresh: Vec<u8>,
    destinations: Vec<u8>,
}

impl PassControls {
    fn new(
        pass: &TargetBatchUpload<'_>,
        domain: usize,
        rows: usize,
        segments: usize,
        whole_slot_fresh: bool,
    ) -> Result<Self, SubmitError> {
        let actual = pass.actual_rows;
        if actual > rows {
            return Err(invalid("draft pass has more rows than its graph class"));
        }
        let history = pass
            .histories
            .get(domain)
            .ok_or_else(|| invalid("draft pass lacks a layer's history domain"))?;
        let mut visible = vec![0_i32; rows * segments * 2];
        for (row, ranges) in history.visible[..actual].iter().enumerate() {
            let used = ranges
                .iter()
                .rposition(|range| range[1] > range[0])
                .map_or(0, |last| last + 1);
            if used > segments {
                return Err(invalid(format!(
                    "draft row attends {used} history spans; the draft admits {segments}"
                )));
            }
            for (span, [start, end]) in ranges[..used].iter().enumerate() {
                let at = (row * segments + span) * 2;
                visible[at] = *start;
                visible[at + 1] = *end;
            }
        }
        let fresh = (0..rows).map(|row| {
            if row >= actual {
                return [0, 0];
            }
            if whole_slot_fresh {
                let slot = pass.row_slots[row] as usize;
                pass.segments[slot]
            } else {
                history.fresh[row]
            }
        });
        Ok(Self {
            coordinates: i32_bytes((0..rows).flat_map(|row| {
                pass.coordinates
                    .get(row)
                    .filter(|_| row < actual)
                    .copied()
                    .unwrap_or([0; 4])
            })),
            visible: i32_bytes(visible),
            fresh: i32_bytes(fresh.flatten()),
            destinations: i32_bytes((0..rows).map(|row| {
                if row < actual {
                    history.destinations[row]
                } else {
                    -1
                }
            })),
        })
    }
}

/// Widen activation rows (bf16 or f16) to F32.
fn widen(bytes: &[u8], dtype: ActivationDType) -> Vec<u8> {
    bytes
        .chunks_exact(2)
        .flat_map(|pair| {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            let value = match dtype {
                ActivationDType::BF16 => f32::from_bits(u32::from(bits) << 16),
                ActivationDType::F16 => f16_to_f32(bits),
            };
            value.to_le_bytes()
        })
        .collect()
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 if mantissa == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

pub struct NativeDraftProgram {
    definition: ModelDefinition,
    graphs: BoundDraftGraphs,
    waiter: CompletionWaiter,
}

impl NativeDraftProgram {
    pub(crate) fn constant_bytes(&self) -> Result<u64, &'static str> {
        self.graphs.constant_bytes()
    }

    pub(crate) fn new(
        definition: ModelDefinition,
        graphs: BoundDraftGraphs,
    ) -> Result<Self, SubmitError> {
        let waiter = CompletionWaiter::spawn().map_err(device)?;
        Ok(Self {
            definition,
            graphs,
            waiter,
        })
    }

    fn queue(
        &self,
        core: &HeadLaunchCore,
        workspace: &mut NativeGraphWorkspaceLease,
        output: NativeGraphOutputLease,
    ) -> Result<(seismic::NativeGraphCompletion, Option<GraphOutputTensor>), SubmitError> {
        let decoder = &self.definition.decoder;
        let draft = self
            .definition
            .draft
            .as_ref()
            .ok_or_else(|| invalid("the draft program's definition has no draft"))?;
        let batch = core.batch();
        let entry = batch.upload();
        let block = batch.chain().next();
        let steps = batch.steps();
        let actual_slots = batch.actual_slots();
        let entry_rows = entry.class.rows();
        let (slots, shaped) = match &block {
            Some(block) => (
                row_class(actual_slots).ok_or_else(|| invalid("draft slots have no class"))?,
                block.shaping[..block.select_rows.len()].iter().any(shapes),
            ),
            None => (0, false),
        };
        let class = DraftGraphClass {
            entry_rows: entry_rows as u64,
            slots: slots as u64,
            shaped,
        };
        let (graph, bound) = self.graphs.class(class)?;
        let planes = core
            .advances()
            .first()
            .ok_or_else(|| invalid("draft batch has no state advance"))?
            .bindings()
            .history;
        // Every pass binds each layer's planes, in plane-descriptor order.
        let mut bindings = bound.bindings();
        let mut layer_domains = Vec::with_capacity(draft.blocks.len());
        for layer in 0..draft.blocks.len() {
            let layer_ref = LayerRef::Head(layer as u32);
            let layer_planes = planes
                .iter()
                .filter(|plane| plane.layer == layer_ref)
                .collect::<Vec<_>>();
            let domain = layer_planes
                .first()
                .ok_or_else(|| invalid("a draft layer has no history plane"))?
                .domain
                .0;
            layer_domains.push(domain);
            let passes = std::iter::once(&graph.injection[layer])
                .chain(graph.block.iter().map(|block| &block.layers[layer]));
            for ports in passes {
                if ports.planes.len() != layer_planes.len() {
                    return Err(invalid(
                        "draft history planes differ from the attention entry",
                    ));
                }
                for (port, plane) in ports.planes.iter().zip(&layer_planes) {
                    bindings.set(port, &plane.buffer).map_err(device)?;
                }
            }
        }
        // Conditioning rows in entry-row order, widened to F32 and padded.
        let mut conditioning = Vec::new();
        for rows in core.conditioning() {
            conditioning.extend(widen(rows.bytes(), decoder.activation_dtype));
        }
        let row_bytes = decoder.hidden as usize * 4;
        if conditioning.len() != entry.actual_rows * row_bytes {
            return Err(invalid("draft conditioning bytes differ from the entry rows"));
        }
        conditioning.resize(entry_rows * row_bytes, 0);
        let mut active = workspace.slot_mut().activate(&graph.plan).map_err(device)?;
        active
            .write_input(&graph.conditioning, &conditioning)
            .map_err(device)?;
        for (ports, &domain) in graph.injection.iter().zip(&layer_domains) {
            let controls = PassControls::new(&entry, domain, entry_rows, 1, false)?;
            write_controls(&mut active, &ports.controls, &controls)?;
        }
        if let (Some(ports), Some(block)) = (&graph.block, &block) {
            if steps as u64 != self.graphs.prepared.proposals {
                return Err(invalid("a draft batch drafts other than the load's proposals"));
            }
            let block_rows = slots * draft.block_size as usize;
            let segments = self.graphs.prepared.segments as usize;
            active
                .write_input(
                    &ports.tokens,
                    &i32_bytes((0..block_rows).flat_map(|row| {
                        [block.tokens.get(row).copied().filter(|_| row < block.actual_rows).unwrap_or(0), 0]
                    })),
                )
                .map_err(device)?;
            for (layer, &domain) in ports.layers.iter().zip(&layer_domains) {
                let controls = PassControls::new(block, domain, block_rows, segments, true)?;
                write_controls(&mut active, &layer.controls, &controls)?;
            }
            // Proposal k of slot s: graph row k · slots + s, packed
            // selection s · steps + k; padding slots repeat selection 0.
            let packed = |step: usize, slot: usize| {
                if slot < actual_slots {
                    slot * steps + step
                } else {
                    0
                }
            };
            let head_rows = (0..steps).flat_map(|step| {
                (0..slots).map(move |slot| (step, slot))
            });
            let rows = head_rows
                .map(|(step, slot)| {
                    let index = packed(step, slot);
                    let output = *block
                        .select_rows
                        .get(index)
                        .ok_or_else(|| invalid("draft selection is absent"))?;
                    block
                        .out_rows
                        .get(output as usize)
                        .copied()
                        .ok_or_else(|| invalid("draft selection has no block row"))
                })
                .collect::<Result<Vec<i32>, SubmitError>>()?;
            active
                .write_input(&ports.head_rows, &i32_bytes(rows))
                .map_err(device)?;
            let words = block.mask_words;
            match ports.selections.as_slice() {
                [selection] => {
                    let order = (0..steps)
                        .flat_map(|step| (0..slots).map(move |slot| (step, slot)))
                        .map(|(step, slot)| packed(step, slot))
                        .collect::<Vec<_>>();
                    write_selection_rows(block, &mut active, selection, &order, words)?;
                }
                per_step => {
                    for (step, selection) in per_step.iter().enumerate() {
                        let order = (0..slots)
                            .map(|slot| packed(step, slot))
                            .collect::<Vec<_>>();
                        write_selection_rows(block, &mut active, selection, &order, words)?;
                    }
                }
            }
            if let Some(anchors) = &ports.anchors {
                let anchors_of = (0..slots).flat_map(|slot| {
                    let row = slot * draft.block_size as usize;
                    [block.tokens.get(row).copied().filter(|_| slot < actual_slots).unwrap_or(0), 0]
                });
                active
                    .write_input(anchors, &i32_bytes(anchors_of))
                    .map_err(device)?;
            }
        }
        let mut output = output;
        let outputs = output
            .activate(&graph.plan)
            .map_err(SubmitError::Invariant)?;
        let (outputs, completion) = active
            .attach(bindings, outputs)
            .and_then(|ready| ready.submit())
            .map_err(device)?;
        let owner = output.publish(outputs);
        let selections = (steps > 0)
            .then(|| {
                owner
                    .tensor(&graph.output)
                    .ok_or_else(|| invalid("draft graph omitted its selections"))
            })
            .transpose()?;
        Ok((completion, selections))
    }
}

fn write_controls(
    active: &mut seismic::NativeGraphFamilyActive<'_>,
    ports: &AttentionControlPorts,
    controls: &PassControls,
) -> Result<(), SubmitError> {
    active
        .write_input(&ports.coordinates, &controls.coordinates)
        .map_err(device)?;
    active
        .write_input(&ports.visible, &controls.visible)
        .map_err(device)?;
    active
        .write_input(&ports.fresh, &controls.fresh)
        .map_err(device)?;
    active
        .write_input(&ports.destinations, &controls.destinations)
        .map_err(device)
}

impl HeadProgram for NativeDraftProgram {
    type Submission =
        DeviceSubmission<HeadLaunchCore, NativeGraphWorkspaceLease, Option<GraphOutputTensor>>;

    fn submit(
        &mut self,
        mut launch: ValidatedHeadLaunch,
    ) -> Result<Self::Submission, (SubmitError, ValidatedHeadLaunch)> {
        let queued = {
            let (core, workspace, output) = launch.execution_parts_mut();
            match output.take() {
                Some(output) => self.queue(core, workspace, output),
                None => Err(invalid("draft graph output lease is absent")),
            }
        };
        let (completion, selections) = match queued {
            Ok(queued) => queued,
            Err(error) => return Err((error, launch)),
        };
        let (core, workspace, _) = launch.into_submission_parts();
        Ok(DeviceSubmission::new(
            self.waiter.completion(vec![completion]),
            core,
            workspace,
            selections,
        ))
    }
}
